"""FedCA: compatibility-aware absorption of client LoRA updates.

Method (see utils/ca_utils.py for the pieces):

  Step 1  protection signal   the downloaded global adapter is the frozen teacher;
                              "excess" = not-true KL to it on answer positions.
  Step 2  local learning      loss = answer CE + ca_lambda * excess.
  Step 3  absorption          theta_g <- theta_g + eta * sum_i w_i (theta_i - theta_g),
                              with eta chosen per round by a federated probe:
                              each client scores candidate steps on a fixed
                              subset of its own *training* rows; the server
                              takes the step with the largest mean answer-CE
                              gain over all clients.  Because every client
                              scores the *merged* update on its own data, the
                              gain already charges cross-client interference
                              (joint compatibility, not per-update scores).
                              Optional stricter guard (ca_kappa < inf): the
                              merged excess on each client's data may not
                              exceed ca_kappa * the clients' own local excess.
                              With kappa = 1 and mostly shared (generic) local
                              changes the guard pins eta near 1, which is why
                              it is not the default.

Ablations / baselines share this code path:
  ca_lambda=0, ca_eta_mode=fixed, ca_eta=1   -> FedIT, bit for bit
  ca_lambda=0, ca_eta_mode=fixed, ca_eta=k   -> FedAvg with server learning rate k
  ca_lambda=0, ca_eta_mode=fedexp            -> FedExP adaptive extrapolation
  ca_lambda>0, ca_eta_mode=fixed, ca_eta=1   -> not-true distillation only (FedNTD-style)
  ca_lambda=0, ca_eta_mode=probe             -> server-side probe only

The global adapter stays a single rank-r LoRA: absorption is done per factor,
so no SVD or rank change is involved.  Communication beyond FedIT per round:
each client uploads 2 * |grid| + 3 scalars and needs the candidate direction,
which equals next round's download up to the scalar eta.
"""

from __future__ import annotations

import json
import os
import time

import torch

from alg.ftbase import FTBaseClient, FTBaseServer
from utils.ca_utils import (
    LoraSwap,
    absorb,
    compatibility_weights,
    fedexp_eta,
    lora_state,
    mean,
    not_true_kl,
    parse_grid,
    probe_measure,
    select_eta,
    select_probe_indices,
    valid_length,
)
from utils.time_utils import time_record
from utils.train_utils import Trainer


def _default(parser, name, value):
    """Keep a YAML-supplied value; argparse defaults would otherwise override it."""
    current = parser.get_default(name)
    return value if current is None else current


def add_args(parser):
    parser.add_argument('--ca_lambda', type=float, default=_default(parser, 'ca_lambda', 1.0),
                        help='weight of the not-true KL (excess change) regulariser in local training')
    parser.add_argument('--ca_tau', type=float, default=_default(parser, 'ca_tau', 1.0),
                        help='temperature of the not-true KL')
    parser.add_argument('--ca_eta_mode', choices=('probe', 'fixed', 'fedexp'),
                        default=_default(parser, 'ca_eta_mode', 'probe'),
                        help='how the server absorption step is chosen')
    parser.add_argument('--ca_eta', type=float, default=_default(parser, 'ca_eta', 1.0),
                        help='server step for ca_eta_mode=fixed')
    parser.add_argument('--ca_eta_grid', type=str, default=_default(parser, 'ca_eta_grid', '1,1.5,2,3,4,6,8'),
                        help='candidate steps for ca_eta_mode=probe')
    parser.add_argument('--ca_kappa', type=float, default=_default(parser, 'ca_kappa', float('inf')),
                        help='optional excess guard = kappa * mean client local excess; inf = select by gain only')
    parser.add_argument('--ca_beta', type=float, default=_default(parser, 'ca_beta', 0.0),
                        help='per-client tilt w_i ∝ p_i (mean_excess/excess_i)^beta; 0 = data-size weights')
    parser.add_argument('--ca_probe_size', type=int, default=_default(parser, 'ca_probe_size', 32),
                        help='training rows per client used to score candidate steps')
    parser.add_argument('--ca_probe_bs', type=int, default=_default(parser, 'ca_probe_bs', 8))
    parser.add_argument('--ca_probe_diag', action='store_true',
                        default=_default(parser, 'ca_probe_diag', False),
                        help='also run (and log, but not use) the probe in fixed/fedexp modes')
    parser.add_argument('--ca_save_every', type=int, default=_default(parser, 'ca_save_every', 1),
                        help='save the global adapter every N completed rounds (0 = never)')
    return parser.parse_args()


def _needs_report(args) -> bool:
    return (args.ca_eta_mode == 'probe' or float(args.ca_beta) != 0.0
            or bool(getattr(args, 'ca_probe_diag', False)))


class CompatibilityTrainer(Trainer):
    """FedIT's local training with an added not-true KL to the global teacher.

    The student forward is FedIT's unchanged forward (same padding, same
    autocast).  The teacher forward runs only over the unpadded prefix, in eval
    mode, with the round's global adapter swapped in.
    """

    def __init__(self, args, dataset, client):
        super().__init__(args=args, dataset=dataset, client=client)
        self.teacher_state = None
        self.swap = None
        self._reset_stats()
        self.last_excess = 0.0

    def _reset_stats(self):
        # Tensors, not floats: avoids a device sync on every micro-batch.
        self._ce_sum, self._kl_sum, self._count = 0, 0, 0

    def set_teacher(self, model, teacher_state):
        self.teacher_state = teacher_state
        self.swap = LoraSwap(model)   # cheap: only collects parameter references

    def train(self, model):
        self._reset_stats()
        if self.commonsense:
            continuation = getattr(self.client.server, 'experiment_run', None) or self.client.server.commonsense_run
            active = continuation.active_for(self.client)
            if active is not None:
                stats = active['fedca_statistics']
                device = next(model.parameters()).device
                self._ce_sum = stats['ce_sum'].to(device)
                self._kl_sum = stats['kl_sum'].to(device)
                self._count = stats['count']
        super().train(model)
        if self._count == 0:
            raise RuntimeError('no training batches were seen')
        self.last_excess = float(self._kl_sum) / self._count
        # Report answer CE only, so logs are comparable with FedIT.
        ce = float(self._ce_sum) / self._count
        if self.commonsense:
            self.last_train_metrics['mean_loss'] = ce
            self.last_train_metrics['train_excess'] = self.last_excess
        return ce

    def checkpoint_statistics(self):
        return {'ce_sum': torch.as_tensor(self._ce_sum).detach().cpu().clone(),
                'kl_sum': torch.as_tensor(self._kl_sum).detach().float().cpu().clone(), 'count': self._count}

    def _forward_causal_lm(self, model, batch):
        if self.commonsense:
            device = next(model.parameters()).device
            input_ids, attention_mask, labels = (batch[key].to(device) for key in ('input_ids', 'attention_mask', 'labels'))
        else:
            input_ids = torch.stack(batch['input_ids']).transpose(0, 1).to(model.device)
            attention_mask = torch.stack(batch['attention_mask']).transpose(0, 1).to(model.device)
            labels = torch.stack(batch['labels']).transpose(0, 1).to(model.device)
        lam = float(self.args.ca_lambda)
        teacher_logits = None
        if lam > 0:
            if self.teacher_state is None:
                raise RuntimeError('ca_lambda > 0 requires a teacher state')
            length = valid_length(attention_mask)
            with torch.no_grad(), self.swap.use(self.teacher_state):
                model.eval()
                with torch.autocast(device_type='cuda', dtype=torch.float16,
                                    enabled=input_ids.is_cuda):
                    teacher_logits = model(input_ids=input_ids[:, :length],
                                           attention_mask=attention_mask[:, :length]).logits
                model.train()
        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=input_ids.is_cuda):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        ce = outputs.loss
        self._ce_sum = self._ce_sum + ce.detach().float()
        self._count += 1
        if teacher_logits is None:
            return ce
        length = teacher_logits.shape[1]
        excess = not_true_kl(outputs.logits[:, :length], teacher_logits, labels[:, :length],
                             tau=float(self.args.ca_tau), compact=self.commonsense)
        self._kl_sum = self._kl_sum + excess.detach()
        return ce + lam * excess


class Client(FTBaseClient):
    def __init__(self, id, args):
        super().__init__(id, args)
        self.trainer = CompatibilityTrainer(args=args, dataset=self.dataset, client=self)
        self.probe_indices = select_probe_indices(
            len(self.dataset['train']), int(args.ca_probe_size), seed=int(args.seed) * 7919 + int(id),
        )
        self.report = None

    @time_record
    def run(self, model):
        teacher = lora_state(model)          # the downloaded global adapter
        if float(self.args.ca_lambda) > 0:
            self.trainer.set_teacher(model, teacher)
        self.last_train_loss = self.trainer.train(model)
        self.lora = {k: v.clone() for k, v in model.state_dict().items() if "lora_" in k}
        self.report = None
        if _needs_report(self.args):
            measured = self.probe(model, teacher, {'local': self.lora})
            self.report = {
                'ce_global': measured['teacher_ce'],
                'ce_local': measured['ce']['local'],
                'excess': measured['excess']['local'],
                'train_excess': self.trainer.last_excess,
                'train_ce': self.last_train_loss,
            }

    def probe(self, model, teacher_state, candidates):
        """Client-side scoring on its own fixed training subset (no test data)."""
        return probe_measure(model, self.dataset['train'], self.probe_indices, teacher_state,
                             candidates, batch_size=int(self.args.ca_probe_bs), tau=float(self.args.ca_tau),
                             dynamic_padding=bool(getattr(self.args, 'commonsense', False) or getattr(self.args, '_shared', False)))


class Server(FTBaseServer):
    def aggregate(self):
        clients = self.sampled_clients
        args = self.args
        total = sum(len(c.dataset['train']) for c in clients)
        if total <= 0:
            raise ValueError('FedCA requires sampled clients with training rows')
        data_weights = [len(c.dataset['train']) / total for c in clients]
        reports = [c.report for c in clients]
        if float(args.ca_beta) != 0.0:
            if any(r is None for r in reports):
                raise RuntimeError('ca_beta != 0 needs client compatibility reports')
            weights = compatibility_weights(data_weights, [r['excess'] for r in reports], float(args.ca_beta))
        else:
            weights = data_weights

        global_state = {k: v for k, v in self.global_lora.items()}
        client_states = [c.lora for c in clients]
        record = {'round': self.round, 'completed_rounds': self.round + 1, 'mode': args.ca_eta_mode,
                  'lambda': float(args.ca_lambda), 'weights': weights,
                  'clients': {str(c.id): {'train_ce': c.last_train_loss,
                                          'train_excess': c.trainer.last_excess,
                                          **(c.report or {})} for c in clients}}

        probe = None
        if args.ca_eta_mode == 'probe' or getattr(args, 'ca_probe_diag', False):
            probe = self._probe_steps(global_state, client_states, weights, reports)
            record['probe'] = probe

        if args.ca_eta_mode == 'fixed':
            eta = float(args.ca_eta)
        elif args.ca_eta_mode == 'fedexp':
            eta = fedexp_eta(global_state, client_states, weights)
        else:
            eta = probe['chosen']
        record['eta'] = eta

        self.global_lora = absorb(global_state, client_states, weights, eta)
        self.model.load_state_dict(self.global_lora, strict=False)
        for c in clients:
            c.report = None
        self._log(record)
        every = int(getattr(args, 'ca_save_every', 0) or 0)
        if every > 0 and (self.round + 1) % every == 0:
            self._save_completed_round(self.round + 1)
        print(f"FedCA round {self.round}: eta={eta:.3g} ({args.ca_eta_mode})"
              + (f" | probe gain {probe['gain'][str(eta)]:.4f} excess {probe['excess'][str(eta)]:.4f}"
                 f" budget {probe['budget']:.4f}" if probe and str(eta) in probe['gain'] else ''))

    # ------------------------------------------------------------------
    def _probe_steps(self, global_state, client_states, weights, reports):
        if any(r is None for r in reports):
            raise RuntimeError('probe needs client compatibility reports')
        grid = parse_grid(self.args.ca_eta_grid)
        candidates = {str(e): absorb(global_state, client_states, weights, e) for e in grid}
        start = time.time()
        per_client = {}
        for client in self.sampled_clients:
            measured = client.probe(self.model, global_state, candidates)
            per_client[str(client.id)] = measured
        self.model.load_state_dict(global_state, strict=False)
        gain = {e: mean(m['teacher_ce'] - m['ce'][str(e)] for m in per_client.values()) for e in grid}
        excess = {e: mean(m['excess'][str(e)] for m in per_client.values()) for e in grid}
        local_excess = mean(r['excess'] for r in reports)
        kappa = float(self.args.ca_kappa)
        budget = float('inf') if kappa == float('inf') else kappa * local_excess
        chosen, bound = select_eta(grid, gain, excess, budget)
        return {
            'grid': grid, 'gain': {str(e): gain[e] for e in grid}, 'excess': {str(e): excess[e] for e in grid},
            'local_excess': local_excess, 'budget': budget, 'chosen': chosen, 'budget_bound': bound,
            'per_client_ce': {cid: {'global': m['teacher_ce'], **m['ce']} for cid, m in per_client.items()},
            'seconds': time.time() - start,
        }

    def _log(self, record):
        if getattr(self, 'experiment_run', None) is not None:
            self.experiment_run.log_algorithm_round(record)
            return
        if getattr(self.args, 'commonsense', False):
            self.commonsense_run.log_algorithm_round(record)
            return
        path = os.path.join(self._adapter_path(), 'fedca_rounds.jsonl')
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'a', encoding='utf-8') as handle:
            handle.write(json.dumps(record, default=float) + '\n')

    def _save_completed_round(self, completed):
        folder = os.path.join(self._adapter_path(), f'round_{completed:03d}')
        os.makedirs(folder, exist_ok=True)
        torch.save({k: v.detach().cpu().clone() for k, v in self.global_lora.items()},
                   os.path.join(folder, 'lora_weights.pt'))
