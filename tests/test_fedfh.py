"""FedFH checks: closed-form limits, optimality, geometry hook, end-to-end round.

Runs on CPU with a tiny randomly initialised Qwen3; no downloads needed.
    python -X utf8 -m unittest tests.test_fedfh -v
"""

import tempfile
import unittest
from types import SimpleNamespace

import torch

from alg import fedfh, flexlora
from utils.fh_utils import (
    build_mean_geometry,
    collect_input_geometry,
    geometry_group,
    harmonize_layer,
)

D = torch.float64


def random_entries(n, k, num, seed=0, same=False):
    g = torch.Generator().manual_seed(seed)
    out = []
    for i in range(num):
        if same and out:
            out.append(dict(out[0]))
            continue
        q, _ = torch.linalg.qr(torch.randn(n, n, generator=g, dtype=D))
        lam = torch.rand(k, generator=g, dtype=D) * 4 + 0.5
        out.append({"Q": q[:, :k], "Lam": lam.sort(descending=True).values,
                    "trace": float(lam.sum()) + 0.1 * (n - k), "tail": 0.1})
    return out


def dense_reference(b_list, a_list, entries, weights, lam, rank):
    """Weighted Eckart-Young on dense matrices (methodology Theorem C)."""
    n = a_list[0].shape[1]
    sig = [(e["Q"] * e["Lam"]) @ e["Q"].T + lam * torch.eye(n, dtype=D) for e in entries]
    sbar = sum(p * s for p, s in zip(weights, sig))
    z = sum(p * b @ a @ s for p, b, a, s in zip(weights, b_list, a_list, sig))
    w, v = torch.linalg.eigh(sbar)
    y = z @ ((v * w.rsqrt()) @ v.T)
    u, s, vh = torch.linalg.svd(y)
    return (u[:, :rank] * s[:rank]) @ vh[:rank] @ ((v * w.rsqrt()) @ v.T), sig


def objective(m, b_list, a_list, sig, weights):
    return sum(p * torch.trace((m - b @ a) @ s @ (m - b @ a).T) for p, b, a, s in zip(weights, b_list, a_list, sig))


class HarmonizeMathTest(unittest.TestCase):
    def setUp(self):
        g = torch.Generator().manual_seed(1)
        self.m, self.n, self.r, self.num, self.k = 12, 20, 2, 4, 5
        self.b = [torch.randn(self.m, self.r, generator=g, dtype=D) for _ in range(self.num)]
        self.a = [torch.randn(self.r, self.n, generator=g, dtype=D) for _ in range(self.num)]
        self.w = [0.1, 0.2, 0.3, 0.4]

    def run_fh(self, entries, lam, rank):
        inv_sqrt = build_mean_geometry(entries, self.w, lam)
        return harmonize_layer(self.b, self.a, entries, self.w, lam, rank, inv_sqrt)

    def test_matches_dense_weighted_eckart_young(self):
        entries = random_entries(self.n, self.k, self.num)
        b_g, a_g, diag = self.run_fh(entries, 0.3, rank=3)
        ref, sig = dense_reference(self.b, self.a, entries, self.w, 0.3, 3)
        torch.testing.assert_close(b_g @ a_g, ref, rtol=1e-9, atol=1e-9)
        # Diagnostics: L~(M_R) = intrinsic + rank loss (Theorem C).
        self.assertAlmostEqual(float(objective(ref, self.b, self.a, sig, self.w)), diag["objective"], places=8)

    def test_large_lambda_recovers_flexlora(self):
        entries = random_entries(self.n, self.k, self.num)
        b_g, a_g, _ = self.run_fh(entries, 1e9, rank=2)
        stacked_a = torch.cat([p * a for p, a in zip(self.w, self.a)], 0)
        fb, fa = flexlora._truncated_product_svd(torch.cat(self.b, 1).double(), stacked_a, 2)
        torch.testing.assert_close(b_g @ a_g, (fb @ fa).double(), rtol=1e-5, atol=1e-5)

    def test_identical_geometry_is_weighted_truncation_of_mean(self):
        entries = random_entries(self.n, self.k, self.num, same=True)
        b_g, a_g, _ = self.run_fh(entries, 0.2, rank=2)
        sig = (entries[0]["Q"] * entries[0]["Lam"]) @ entries[0]["Q"].T + 0.2 * torch.eye(self.n, dtype=D)
        w, v = torch.linalg.eigh(sig)
        mean = sum(p * b @ a for p, b, a in zip(self.w, self.b, self.a))
        u, s, vh = torch.linalg.svd(mean @ ((v * w.sqrt()) @ v.T))
        ref = (u[:, :2] * s[:2]) @ vh[:2] @ ((v * w.rsqrt()) @ v.T)
        torch.testing.assert_close(b_g @ a_g, ref, rtol=1e-9, atol=1e-9)

    def test_output_gauge_matches_flexlora_convention(self):
        b_g, a_g, _ = self.run_fh(random_entries(self.n, self.k, self.num), 0.3, rank=3)
        torch.testing.assert_close(a_g @ a_g.T, torch.eye(3, dtype=D), rtol=0, atol=1e-10)

    def test_beats_random_rank_r_candidates(self):
        entries = random_entries(self.n, self.k, self.num, seed=3)
        b_g, a_g, _ = self.run_fh(entries, 0.05, rank=2)
        _, sig = dense_reference(self.b, self.a, entries, self.w, 0.05, 2)
        best = float(objective(b_g @ a_g, self.b, self.a, sig, self.w))
        g = torch.Generator().manual_seed(9)
        for _ in range(50):
            cand = (b_g + 0.05 * torch.randn(b_g.shape, generator=g, dtype=D)) @ (a_g + 0.05 * torch.randn(a_g.shape, generator=g, dtype=D))
            self.assertGreaterEqual(float(objective(cand, self.b, self.a, sig, self.w)), best - 1e-9)


def tiny_peft_qwen3():
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import Qwen3Config, Qwen3ForCausalLM
    torch.manual_seed(0)
    config = Qwen3Config(vocab_size=97, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                         num_attention_heads=4, num_key_value_heads=2, head_dim=8,
                         max_position_embeddings=64)
    model = Qwen3ForCausalLM(config).float()
    lora = LoraConfig(task_type=TaskType.CAUSAL_LM, r=2, lora_alpha=8, lora_dropout=0.05,
                      target_modules=["q_proj", "v_proj"])
    return get_peft_model(model, lora)


def fake_rows(num, seq=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    rows = {"input_ids": [], "attention_mask": [], "labels": []}
    for _ in range(num):
        valid = int(torch.randint(5, seq, (1,), generator=g))
        ids = torch.randint(1, 97, (valid,), generator=g).tolist()
        rows["input_ids"].append(ids + [0] * (seq - valid))
        rows["attention_mask"].append([1] * valid + [0] * (seq - valid))
        rows["labels"].append([-100] * (valid - 2) + ids[-2:] + [-100] * (seq - valid))
    from datasets import Dataset
    return Dataset.from_dict(rows)


class GeometryHookTest(unittest.TestCase):
    def test_grouping(self):
        self.assertEqual(geometry_group("m.layers.3.self_attn.q_proj"), geometry_group("m.layers.3.self_attn.v_proj"))
        self.assertNotEqual(geometry_group("m.layers.3.self_attn.q_proj"), geometry_group("m.layers.4.self_attn.q_proj"))

    def test_second_moment_matches_layernorm_hidden_states(self):
        model = tiny_peft_qwen3()
        rows = fake_rows(7)
        geo = collect_input_geometry(model, rows, top_k=32, batch_size=3)
        self.assertEqual(len(geo), 2)                          # one group per decoder layer
        self.assertTrue(model.training)                         # mode restored
        base = model.base_model.model
        acc = [torch.zeros(32, 32, dtype=D) for _ in range(2)]
        tokens = 0
        model.eval()
        with torch.no_grad():
            for i in range(len(rows)):
                ids = torch.tensor([rows[i]["input_ids"]]); mask = torch.tensor([rows[i]["attention_mask"]])
                out = model(input_ids=ids, attention_mask=mask, output_hidden_states=True)
                keep = mask[0].bool()
                tokens += int(keep.sum())
                for layer in range(2):
                    h = base.model.layers[layer].input_layernorm(out.hidden_states[layer])[0][keep].double()
                    acc[layer] += h.T @ h
        for layer, group in enumerate(sorted(geo)):
            sigma = acc[layer] / tokens
            e = geo[group]
            recon = (e["Q"].double() * e["Lam"].double()) @ e["Q"].double().T
            torch.testing.assert_close(recon, sigma, rtol=1e-4, atol=1e-5)
            self.assertAlmostEqual(e["trace"], float(torch.trace(sigma)), places=4)
            self.assertEqual(e["tokens"], tokens)


class EndToEndRoundTest(unittest.TestCase):
    def test_two_clients_one_round(self):
        model = tiny_peft_qwen3()
        global_lora = {k: v.clone() for k, v in model.state_dict().items() if "lora_" in k}
        with tempfile.TemporaryDirectory() as tmp:
            args = SimpleNamespace(task_type='CAUSAL_LM', bs=1, grad_accum=2, epoch=1, step=3, lr=1e-2,
                                   seed=0, fh_k=8, fh_stat_bs=4, fh_gamma=0.1, fh_server_device='cpu',
                                   lora_rank=2, suffix=tmp, alg='fedfh', dataset='zsre', model='tiny',
                                   cn=2, global_test=False)
            server = object.__new__(fedfh.Server)
            server.args, server.model, server.global_lora, server.round = args, model, global_lora, 0
            clients = []
            for cid in range(2):
                client = object.__new__(fedfh.Client)
                client.id, client.args, client.server, client.delay = cid, args, server, 1.0
                client.dataset = {"train": fake_rows(6, seed=cid + 1)}
                from utils.train_utils import Trainer
                client.trainer = Trainer(args=args, dataset=client.dataset, client=client)
                client.run(model)
                model.load_state_dict(global_lora, strict=False)
                clients.append(client)
            server.sampled_clients = clients
            server.aggregate()
            self.assertTrue(all(c.geometry is None for c in clients))
            changed = any(not torch.equal(server.global_lora[k], global_lora[k]) for k in global_lora)
            self.assertTrue(changed)
            for key, value in server.global_lora.items():
                self.assertTrue(torch.isfinite(value).all(), key)
                torch.testing.assert_close(model.state_dict()[key], value)


if __name__ == "__main__":
    unittest.main()
