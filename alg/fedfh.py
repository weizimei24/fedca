"""FedFH: fixed-rank federated functional harmonization of LoRA updates.

Clients train exactly as in FedIT, then upload their LoRA factors plus a
low-rank summary of each target layer's input second moment (top-k
eigenpairs, trace, (k+1)-th eigenvalue). The server returns the optimal
rank-R global LoRA of

    sum_i p_i || (M - B_i A_i) (Q_i Lam_i Q_i^T + lambda I)^{1/2} ||_F^2,

with lambda = fh_gamma * (mean eigenvalue of the layer's input moment).
fh_gamma -> infinity recovers FlexLoRA. Base weights are never modified.
"""

from __future__ import annotations

import json
import os

import torch

from alg.ftbase import FTBaseClient, FTBaseServer
from utils.fh_utils import (
    build_mean_geometry,
    collect_input_geometry,
    geometry_group,
    harmonize_layer,
    layer_lambda,
    module_name_from_lora_key,
)
from utils.time_utils import time_record


def add_args(parser):
    parser.add_argument('--fh_gamma', type=float, default=0.1,
                        help='lambda = fh_gamma * mean eigenvalue of the layer input second moment')
    parser.add_argument('--fh_k', type=int, default=32,
                        help='number of input-geometry eigenpairs uploaded per layer')
    parser.add_argument('--fh_stat_bs', type=int, default=16,
                        help='batch size of the gradient-free geometry pass')
    parser.add_argument('--fh_server_device', type=str, default='auto',
                        help="device for server aggregation: auto (model device), cuda or cpu")
    return parser.parse_args()


class Client(FTBaseClient):
    def __init__(self, id, args):
        super().__init__(id, args)
        self.geometry = None

    @time_record
    def run(self, model):
        # Local training is identical to FedIT; super().run copies client.lora.
        self.last_train_loss = self.trainer.train(model)
        self.lora = {k: v.clone() for k, v in model.state_dict().items() if "lora_" in k}
        # Geometry of the trained local model, before the server restores the
        # shared container to the global adapter.
        self.geometry = collect_input_geometry(
            model, self.dataset['train'], top_k=self.args.fh_k, batch_size=self.args.fh_stat_bs,
        )


class Server(FTBaseServer):
    def _server_device(self):
        choice = getattr(self.args, 'fh_server_device', 'auto')
        if choice == 'auto':
            return next(self.model.parameters()).device
        return torch.device(choice)

    def aggregate(self):
        clients = self.sampled_clients
        total = sum(len(client.dataset['train']) for client in clients)
        if total <= 0:
            raise ValueError('FedFH requires sampled clients with training rows')
        weights = [len(client.dataset['train']) / total for client in clients]
        rank = self.args.lora_rank
        device = self._server_device()
        dtype = torch.float64
        state = self.model.state_dict()

        # Group LoRA layers by the input tensor they read (q/v share one).
        groups: dict[str, list[str]] = {}
        for a_key in (key for key in self.global_lora if 'lora_A' in key):
            groups.setdefault(geometry_group(module_name_from_lora_key(a_key)), []).append(a_key)

        updated, layer_logs = {}, []
        with torch.no_grad():
            for group, a_keys in groups.items():
                entries = []
                for client in clients:
                    if client.geometry is None or group not in client.geometry:
                        raise KeyError(f'Client {client.id} has no geometry for {group}')
                    g = client.geometry[group]
                    entries.append({
                        "Q": g["Q"].to(device, dtype), "Lam": g["Lam"].to(device, dtype),
                        "trace": g["trace"], "tail": g["tail"],
                    })
                lam = layer_lambda(entries, weights, self.args.fh_gamma)
                inv_sqrt = build_mean_geometry(entries, weights, lam)
                coverage = sum(p * float(e["Lam"].sum()) / e["trace"] for p, e in zip(weights, entries))

                for a_key in a_keys:
                    b_key = a_key.replace('lora_A', 'lora_B')
                    b_list = [client.lora[b_key].to(device, dtype) for client in clients]
                    a_list = [client.lora[a_key].to(device, dtype) for client in clients]
                    b_g, a_g, diag = harmonize_layer(b_list, a_list, entries, weights, lam, rank, inv_sqrt)
                    if a_g.shape != state[a_key].shape or b_g.shape != state[b_key].shape:
                        raise ValueError(f'FedFH shape mismatch for {a_key}')
                    updated[a_key] = a_g.to(state[a_key].device, state[a_key].dtype)
                    updated[b_key] = b_g.to(state[b_key].device, state[b_key].dtype)
                    diag.update({"layer": module_name_from_lora_key(a_key), "lambda": lam,
                                 "topk_coverage": coverage})
                    layer_logs.append(diag)
                del inv_sqrt, entries

        missing = set(self.global_lora) - set(updated)
        if missing:
            raise KeyError(f'FedFH did not update {len(missing)} LoRA tensors, e.g. {sorted(missing)[0]}')
        self.global_lora = updated
        self.model.load_state_dict(self.global_lora, strict=False)
        for client in clients:          # free uploaded geometry
            client.geometry = None
        self._log_diagnostics(layer_logs)

    def _log_diagnostics(self, layer_logs):
        def ratio(num, den):
            return num / den if den > 0 else 0.0
        own = sum(d["own_energy"] for d in layer_logs)
        summary = {
            "round": self.round,
            "layers": len(layer_logs),
            "intrinsic_ratio": ratio(sum(d["intrinsic"] for d in layer_logs), own),
            "rank_loss_ratio": ratio(sum(d["rank_loss"] for d in layer_logs), own),
            "certificate_ratio": ratio(sum(d["certificate"] for d in layer_logs),
                                       sum(d["objective"] for d in layer_logs)),
            "mean_topk_coverage": sum(d["topk_coverage"] for d in layer_logs) / len(layer_logs),
        }
        print('FedFH aggregated {layers} layers | intrinsic/own {intrinsic_ratio:.4f} | '
              'rank-loss/own {rank_loss_ratio:.4f} | certificate/objective {certificate_ratio:.4f} | '
              'top-k coverage {mean_topk_coverage:.4f}'.format(**summary))
        path = os.path.join(self._adapter_path(), 'fh_diagnostics.jsonl')
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'a', encoding='utf-8') as handle:
            handle.write(json.dumps({"summary": summary, "layers": layer_logs}) + '\n')
