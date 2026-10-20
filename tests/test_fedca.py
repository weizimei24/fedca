"""FedCA checks on CPU with a tiny randomly initialised Qwen3 (no downloads).

    python -X utf8 -m unittest tests.test_fedca -v

Covers: the not-true KL (step 1), FedIT equivalence of the training and
aggregation paths when the method is switched off, the teacher swap, probe
scoring, step selection, FedExP, and full rounds in every server mode.
"""

import json
import math
import os
import tempfile
import unittest
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from alg import fedca
from alg.ftbase import FTBaseServer
from utils.ca_utils import (
    LoraSwap,
    absorb,
    compatibility_weights,
    fedexp_eta,
    lora_state,
    not_true_kl,
    parse_grid,
    probe_measure,
    select_eta,
    select_probe_indices,
)
from utils.train_utils import Trainer

VOCAB = 97


def tiny_peft_qwen3(seed=0):
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import Qwen3Config, Qwen3ForCausalLM
    torch.manual_seed(seed)
    config = Qwen3Config(vocab_size=VOCAB, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                         num_attention_heads=4, num_key_value_heads=2, head_dim=8,
                         max_position_embeddings=64)
    model = Qwen3ForCausalLM(config).float()
    lora = LoraConfig(task_type=TaskType.CAUSAL_LM, r=2, lora_alpha=8, lora_dropout=0.05,
                      target_modules=["q_proj", "v_proj"])
    model = get_peft_model(model, lora)
    # PEFT initialises B = 0; give B a value so the global adapter is non-trivial,
    # as it is after the first aggregation in a real run.
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "lora_B" in name:
                p.normal_(0, 0.05)
    return model


def fake_rows(num, seq=16, seed=0):
    """Right-padded rows; the last two valid tokens are supervised (answer + EOS)."""
    g = torch.Generator().manual_seed(seed)
    rows = {"input_ids": [], "attention_mask": [], "labels": []}
    for _ in range(num):
        valid = int(torch.randint(6, seq, (1,), generator=g))
        ids = torch.randint(1, VOCAB, (valid,), generator=g).tolist()
        rows["input_ids"].append(ids + [0] * (seq - valid))
        rows["attention_mask"].append([1] * valid + [0] * (seq - valid))
        rows["labels"].append([-100] * (valid - 2) + ids[-2:] + [-100] * (seq - valid))
    from datasets import Dataset
    return Dataset.from_dict(rows)


def make_args(tmp, **overrides):
    args = dict(task_type='CAUSAL_LM', bs=1, grad_accum=2, epoch=1, step=3, lr=1e-2, seed=0,
                lora_rank=2, suffix=tmp, alg='fedca', dataset='zsre', model='tiny', cn=3,
                global_test=False, ca_lambda=1.0, ca_tau=1.0, ca_eta_mode='probe', ca_eta=1.0,
                ca_eta_grid='1,2,4', ca_kappa=1.0, ca_beta=0.0, ca_probe_size=4, ca_probe_bs=2,
                ca_probe_diag=False, ca_save_every=1, epoch_unused=None)
    args.update(overrides)
    return SimpleNamespace(**args)


def build_federation(model, args, num_clients=3, client_cls=fedca.Client, server_cls=fedca.Server,
                     trainer_cls=None):
    server = object.__new__(server_cls)
    server.args, server.model, server.round = args, model, 0
    server.global_lora = lora_state(model)
    clients = []
    for cid in range(num_clients):
        client = object.__new__(client_cls)
        client.id, client.args, client.server, client.delay = cid, args, server, 1.0
        client.dataset = {"train": fake_rows(8, seed=cid + 1)}
        cls = trainer_cls or (fedca.CompatibilityTrainer if client_cls is fedca.Client else Trainer)
        client.trainer = cls(args=args, dataset=client.dataset, client=client)
        client.lora, client.last_train_loss = {}, None
        if client_cls is fedca.Client:
            client.probe_indices = select_probe_indices(8, args.ca_probe_size, seed=cid)
            client.report = None
        clients.append(client)
    server.clients = server.sampled_clients = clients
    return server, clients


def run_round(server, clients):
    for client in clients:
        client.run(server.model)
        server.model.load_state_dict(server.global_lora, strict=False)
    server.aggregate()


class NotTrueKLTest(unittest.TestCase):
    def setUp(self):
        g = torch.Generator().manual_seed(0)
        self.teacher = torch.randn(2, 6, 11, generator=g)
        self.labels = torch.full((2, 6), -100)
        self.labels[0, 3:5] = torch.tensor([4, 7])
        self.labels[1, 2:4] = torch.tensor([1, 10])

    def test_zero_for_identical_and_for_gold_only_changes(self):
        self.assertAlmostEqual(float(not_true_kl(self.teacher, self.teacher, self.labels)), 0.0, places=6)
        student = self.teacher.clone()
        # Raise the gold logit at every supervised position: the necessary change.
        for b, t in ((0, 2), (0, 3), (1, 1), (1, 2)):
            student[b, t, self.labels[b, t + 1]] += 5.0
        self.assertAlmostEqual(float(not_true_kl(student, self.teacher, self.labels)), 0.0, places=5)
        # Unsupervised positions are ignored entirely.
        student[0, 0] += torch.randn(11)
        self.assertAlmostEqual(float(not_true_kl(student, self.teacher, self.labels)), 0.0, places=5)

    def test_positive_for_non_gold_changes_and_matches_reference(self):
        g = torch.Generator().manual_seed(1)
        student = self.teacher + torch.randn(self.teacher.shape, generator=g)
        value = not_true_kl(student, self.teacher, self.labels, tau=2.0)
        self.assertGreater(float(value), 0.0)
        # Reference via boolean indexing and explicit removal of the gold token.
        terms = []
        for b in range(2):
            for t in range(5):
                y = int(self.labels[b, t + 1])
                if y == -100:
                    continue
                keep = [v for v in range(11) if v != y]
                p = F.softmax(self.teacher[b, t, keep] / 2.0, dim=-1)
                log_q = F.log_softmax(student[b, t, keep] / 2.0, dim=-1)
                terms.append((p * (p.log() - log_q)).sum() * 4.0)
        torch.testing.assert_close(value, torch.stack(terms).mean(), rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(not_true_kl(student, self.teacher, self.labels, 2.0, reduction="sum"),
                                   torch.stack(terms).sum(), rtol=1e-5, atol=1e-6)

    def test_gradient_flows_only_to_student(self):
        student = self.teacher.clone().requires_grad_(True)
        teacher = (self.teacher + 0.3).requires_grad_(True)
        not_true_kl(student, teacher, self.labels).backward()
        self.assertTrue(torch.isfinite(student.grad).all())
        self.assertGreater(float(student.grad.abs().sum()), 0.0)
        self.assertIsNone(teacher.grad)


class AbsorptionMathTest(unittest.TestCase):
    def setUp(self):
        g = torch.Generator().manual_seed(2)
        self.global_state = {"a": torch.randn(3, 4, generator=g), "b": torch.randn(4, 3, generator=g)}
        self.clients = [{k: v + torch.randn(v.shape, generator=g) for k, v in self.global_state.items()}
                        for _ in range(3)]
        self.w = [0.2, 0.3, 0.5]

    def test_eta_one_is_bitwise_fedit(self):
        out = absorb(self.global_state, self.clients, self.w, 1.0)
        for key in self.global_state:
            ref = 0
            for w, c in zip(self.w, self.clients):   # FTBaseServer.aggregate's expression
                ref = ref + c[key] * w
            self.assertTrue(torch.equal(out[key], ref))

    def test_eta_scales_the_mean_update(self):
        out = absorb(self.global_state, self.clients, self.w, 3.0)
        for key, g in self.global_state.items():
            mean = sum(w * c[key] for w, c in zip(self.w, self.clients))
            torch.testing.assert_close(out[key], g + 3.0 * (mean - g))

    def test_fedexp_limits(self):
        same = [dict(self.clients[0]) for _ in range(4)]
        self.assertEqual(fedexp_eta(self.global_state, same, [0.25] * 4), 1.0)
        # Orthogonal, equal-norm deltas: sum ||d_i||^2/N over 2||mean||^2 = N/2.
        n = 6
        g = {"x": torch.zeros(n)}
        clients = [{"x": torch.eye(n)[i]} for i in range(n)]
        self.assertAlmostEqual(fedexp_eta(g, clients, [1 / n] * n), n / 2, places=6)

    def test_compatibility_weights(self):
        self.assertEqual(compatibility_weights([0.5, 0.5], [1.0, 3.0], 0.0), [0.5, 0.5])
        w = compatibility_weights([0.5, 0.5], [1.0, 3.0], 1.0)
        self.assertAlmostEqual(sum(w), 1.0)
        self.assertGreater(w[0], w[1])

    def test_select_eta(self):
        grid = [1.0, 2.0, 4.0]
        gain = {1.0: 0.1, 2.0: 0.3, 4.0: 0.5}
        self.assertEqual(select_eta(grid, gain, {1.0: 0.1, 2.0: 0.2, 4.0: 0.9}, 0.5), (2.0, True))
        self.assertEqual(select_eta(grid, gain, {1.0: 0.1, 2.0: 0.2, 4.0: 0.3}, 0.5), (4.0, False))
        self.assertEqual(select_eta(grid, gain, {1.0: 0.9, 2.0: 0.9, 4.0: 0.9}, 0.5), (1.0, True))
        self.assertEqual(select_eta(grid, {1.0: 0.2, 2.0: 0.2, 4.0: 0.1}, {e: 0 for e in grid}, 1), (1.0, False))
        self.assertEqual(parse_grid("4, 1,2,2"), [1.0, 2.0, 4.0])
        with self.assertRaises(ValueError):
            parse_grid("1,-2")


class ModelPiecesTest(unittest.TestCase):
    def test_swap_restores_exactly_and_uses_teacher(self):
        model = tiny_peft_qwen3()
        teacher = lora_state(model)
        with torch.no_grad():
            for name, p in model.named_parameters():
                if "lora_" in name:
                    p.add_(0.1)
        student = lora_state(model)
        rows = fake_rows(2)
        ids = torch.tensor(rows["input_ids"])
        model.eval()
        with torch.no_grad(), LoraSwap(model).use(teacher):
            swapped = model(input_ids=ids).logits
        for k, v in lora_state(model).items():
            self.assertTrue(torch.equal(v, student[k]))
        model.load_state_dict(teacher, strict=False)
        with torch.no_grad():
            torch.testing.assert_close(swapped, model(input_ids=ids).logits)

    def test_trimming_right_padding_preserves_answer_logits(self):
        model = tiny_peft_qwen3().eval()
        rows = fake_rows(3, seq=16)
        ids, mask = torch.tensor(rows["input_ids"]), torch.tensor(rows["attention_mask"])
        length = int(mask.sum(1).max())
        with torch.no_grad():
            full = model(input_ids=ids, attention_mask=mask).logits
            trimmed = model(input_ids=ids[:, :length], attention_mask=mask[:, :length]).logits
        keep = mask[:, :length].bool()
        torch.testing.assert_close(full[:, :length][keep], trimmed[keep], rtol=1e-5, atol=1e-5)

    def test_probe_identity_candidate_and_state_restore(self):
        model = tiny_peft_qwen3()
        teacher = lora_state(model)
        other = {k: v + 0.05 for k, v in teacher.items()}
        model.load_state_dict(other, strict=False)
        model.train()
        rows = fake_rows(5)
        out = probe_measure(model, rows, [0, 1, 3, 4], teacher, {"same": teacher, "other": other}, batch_size=3)
        self.assertAlmostEqual(out["ce"]["same"], out["teacher_ce"], places=5)
        self.assertAlmostEqual(out["excess"]["same"], 0.0, places=6)
        self.assertGreater(out["excess"]["other"], 0.0)
        self.assertEqual(out["tokens"], 8)
        self.assertTrue(model.training)
        for k, v in lora_state(model).items():
            self.assertTrue(torch.equal(v, other[k]))

    def test_probe_indices_fixed_and_seeded(self):
        a, b = select_probe_indices(50, 8, 3), select_probe_indices(50, 8, 3)
        self.assertEqual(a, b)
        self.assertEqual(len(set(a)), 8)
        self.assertNotEqual(a, select_probe_indices(50, 8, 4))
        self.assertEqual(select_probe_indices(5, 8, 0), [0, 1, 2, 3, 4])


class FedITEquivalenceTest(unittest.TestCase):
    """Switched off (lambda=0, fixed eta=1) FedCA must be FedIT exactly."""

    def test_round_matches_fedit_bitwise(self):
        from alg import fedit
        with tempfile.TemporaryDirectory() as tmp:
            args = make_args(tmp, ca_lambda=0.0, ca_eta_mode='fixed', ca_eta=1.0, ca_save_every=0)
            s1, c1 = build_federation(tiny_peft_qwen3(), args)
            s2, c2 = build_federation(tiny_peft_qwen3(), args, client_cls=fedit.Client, server_cls=fedit.Server)
            for rnd in range(2):
                s1.round = s2.round = rnd
                run_round(s1, c1)
                run_round(s2, c2)
            for key in s2.global_lora:
                self.assertTrue(torch.equal(s1.global_lora[key], s2.global_lora[key]), key)
            for a, b in zip(c1, c2):
                self.assertAlmostEqual(a.last_train_loss, b.last_train_loss, places=6)


class RegulariserEffectTest(unittest.TestCase):
    def test_lambda_reduces_local_excess(self):
        results = {}
        for lam in (0.0, 20.0):
            with tempfile.TemporaryDirectory() as tmp:
                args = make_args(tmp, ca_lambda=lam, step=6, ca_eta_mode='fixed', ca_probe_diag=True,
                                 ca_probe_size=8, ca_save_every=0)
                server, clients = build_federation(tiny_peft_qwen3(), args, num_clients=1)
                clients[0].run(server.model)
                results[lam] = clients[0].report
        self.assertGreater(results[0.0]["excess"], 0.0)
        self.assertLess(results[20.0]["excess"], results[0.0]["excess"])
        # The constrained client still learns its answers.
        self.assertLess(results[20.0]["ce_local"], results[20.0]["ce_global"])


class EndToEndTest(unittest.TestCase):
    def _check(self, server, tmp, rounds):
        for value in server.global_lora.values():
            self.assertTrue(torch.isfinite(value).all())
        for key, value in server.global_lora.items():
            torch.testing.assert_close(server.model.state_dict()[key], value)
        log = os.path.join(server._adapter_path(), 'fedca_rounds.jsonl')
        with open(log, encoding='utf-8') as handle:
            records = [json.loads(line) for line in handle]
        self.assertEqual(len(records), rounds)
        for r in range(1, rounds + 1):
            self.assertTrue(os.path.isfile(os.path.join(server._adapter_path(), f'round_{r:03d}', 'lora_weights.pt')))
        return records

    def test_probe_mode_two_rounds(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = make_args(tmp)
            server, clients = build_federation(tiny_peft_qwen3(), args)
            for rnd in range(2):
                server.round = rnd
                run_round(server, clients)
            records = self._check(server, tmp, 2)
            for rec in records:
                probe = rec['probe']
                self.assertIn(rec['eta'], probe['grid'])
                self.assertTrue(math.isfinite(probe['budget']))
                fits = probe['excess'][str(rec['eta'])] <= probe['budget'] + 1e-12
                # Either the chosen step fits the budget, or nothing fit and the smallest step is used.
                self.assertTrue(fits or rec['eta'] == min(probe['grid']))
                self.assertEqual(set(rec['clients']), {'0', '1', '2'})

    def test_other_modes(self):
        for overrides in ({'ca_eta_mode': 'fixed', 'ca_eta': 3.0},
                          {'ca_eta_mode': 'fedexp', 'ca_lambda': 0.0},
                          {'ca_eta_mode': 'probe', 'ca_kappa': float('inf')},
                          {'ca_eta_mode': 'probe', 'ca_beta': 1.0},
                          {'ca_eta_mode': 'fixed', 'ca_eta': 2.0, 'ca_probe_diag': True}):
            with self.subTest(**overrides), tempfile.TemporaryDirectory() as tmp:
                args = make_args(tmp, **overrides)
                server, clients = build_federation(tiny_peft_qwen3(), args)
                run_round(server, clients)
                rec = self._check(server, tmp, 1)[0]
                if overrides['ca_eta_mode'] == 'fixed':
                    self.assertEqual(rec['eta'], overrides['ca_eta'])
                if overrides['ca_eta_mode'] == 'fedexp':
                    self.assertGreaterEqual(rec['eta'], 1.0)
                if overrides.get('ca_beta'):
                    self.assertNotEqual(rec['weights'], [1 / 3] * 3)


class ArgsTest(unittest.TestCase):
    def test_yaml_values_survive_add_args(self):
        import argparse
        parser = argparse.ArgumentParser()
        parser.set_defaults(ca_lambda=0.25, ca_eta_mode='fixed')
        import sys
        argv, sys.argv = sys.argv, ['prog']
        try:
            args = fedca.add_args(parser)
        finally:
            sys.argv = argv
        self.assertEqual(args.ca_lambda, 0.25)
        self.assertEqual(args.ca_eta_mode, 'fixed')
        self.assertEqual(args.ca_kappa, float('inf'))   # default: select the step by probe gain only


if __name__ == "__main__":
    unittest.main()
