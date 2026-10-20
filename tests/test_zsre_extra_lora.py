"""Fast aggregation/checkpoint checks without loading Qwen or training data."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from alg import fedexlora, ffalora, flexlora, ilora


A = 'layer.lora_A.default.weight'
B = 'layer.lora_B.default.weight'
W = 'layer.base_layer.weight'


class FakeModel:
    def __init__(self, a, b, w=None):
        self.values = {
            A: torch.nn.Parameter(a.clone()),
            B: torch.nn.Parameter(b.clone()),
            W: torch.nn.Parameter(torch.zeros(b.shape[0], a.shape[1]) if w is None else w.clone()),
        }

    def parameters(self):
        return self.values.values()

    def named_parameters(self):
        return self.values.items()

    def state_dict(self):
        return {key: value.detach() for key, value in self.values.items()}

    def load_state_dict(self, values, strict=False):
        for key, value in values.items():
            self.values[key].data.copy_(value)


def sample_clients(a1, b1, a2, b2, b_only=False):
    return [
        SimpleNamespace(dataset={'train': [0]}, lora={B: b1} if b_only else {A: a1, B: b1}),
        SimpleNamespace(dataset={'train': [0, 1, 2]}, lora={B: b2} if b_only else {A: a2, B: b2}),
    ]


class ExtraLoRATest(unittest.TestCase):
    def setUp(self):
        self.a1 = torch.tensor([[1., 2., 0.], [0., 1., 1.]])
        self.b1 = torch.tensor([[2., 0.], [0., 1.], [1., 0.]])
        self.a2 = torch.tensor([[1., 0., 1.], [2., 1., 0.]])
        self.b2 = torch.tensor([[0., 1.], [1., 0.], [0., 2.]])

    def test_fedex_exact_effective_update_and_checkpoint(self):
        model = FakeModel(torch.zeros_like(self.a1), torch.zeros_like(self.b1))
        server = object.__new__(fedexlora.Server)
        server.model = model
        server.global_lora = {A: torch.zeros_like(self.a1), B: torch.zeros_like(self.b1)}
        server.sampled_clients = sample_clients(self.a1, self.b1, self.a2, self.b2)
        with tempfile.TemporaryDirectory() as temporary:
            server.args = SimpleNamespace(
                suffix=temporary, alg='fedexlora', dataset='zsre', model='Qwen3-1.7B',
                cn=8, epoch=1, lr=0.0001, lora_alpha=8, lora_rank=2,
            )
            server.aggregate()
            expected = 4 * (0.25 * self.b1 @ self.a1 + 0.75 * self.b2 @ self.a2)
            effective = model.values[W].detach() + 4 * (server.global_lora[B] @ server.global_lora[A])
            torch.testing.assert_close(effective, expected)
            server.save_adapter()
            path = Path(server._adapter_path())
            saved_a_b = torch.load(path / 'lora_weights.pt', weights_only=True)
            saved_w = torch.load(path / 'merged_weights.pt', weights_only=True)
            replay = saved_w[W] + 4 * saved_a_b[B] @ saved_a_b[A]
            torch.testing.assert_close(replay, expected)

    def test_ffa_trains_only_b_and_preserves_a(self):
        model = FakeModel(self.a1, torch.zeros_like(self.b1))
        client = object.__new__(ffalora.Client)
        client.server = SimpleNamespace(round=0)
        client.delay = 1.

        class FakeTrainer:
            def train(self, current_model):
                active = {key for key, value in current_model.named_parameters() if value.requires_grad}
                assert active == {B}
                current_model.values[B].data.add_(1.)
                return 1.

        client.trainer = FakeTrainer()
        client.run(model)
        self.assertEqual(set(client.lora), {B})
        self.assertEqual(client.last_train_loss, 1.)

        server = object.__new__(ffalora.Server)
        server.model = model
        server.global_lora = {A: self.a1.clone(), B: torch.zeros_like(self.b1)}
        server.sampled_clients = sample_clients(self.a1, self.b1, self.a2, self.b2, b_only=True)
        server.aggregate()
        self.assertEqual(set(server.global_lora), {A, B})
        torch.testing.assert_close(server.global_lora[A], self.a1)
        torch.testing.assert_close(server.global_lora[B], 0.25 * self.b1 + 0.75 * self.b2)

    def test_flex_rank_truncation_matches_dense_svd(self):
        model = FakeModel(torch.zeros_like(self.a1), torch.zeros_like(self.b1))
        server = object.__new__(flexlora.Server)
        server.model = model
        server.args = SimpleNamespace(lora_rank=2, lora_alpha=8)
        server.global_lora = {A: torch.zeros_like(self.a1), B: torch.zeros_like(self.b1)}
        server.sampled_clients = sample_clients(self.a1, self.b1, self.a2, self.b2)
        server.aggregate()
        delta = 0.25 * self.b1 @ self.a1 + 0.75 * self.b2 @ self.a2
        u, s, vh = torch.linalg.svd(delta, full_matrices=False)
        expected = (u[:, :2] * s[:2]) @ vh[:2, :]
        torch.testing.assert_close(server.global_lora[B] @ server.global_lora[A], expected, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(model.values[B].detach() @ model.values[A].detach(), expected, atol=1e-5, rtol=1e-5)

    def test_ilora_qr_initialization_and_checkpoint_replay(self):
        original = torch.tensor([[1., 2., 0.], [0., 3., 1.], [1., 0., 2.]])
        model = FakeModel(torch.zeros_like(self.a1), torch.zeros_like(self.b1), original)
        server = object.__new__(ilora.Server)
        server.model = model
        server.global_lora = {A: torch.zeros_like(self.a1), B: torch.zeros_like(self.b1)}
        with tempfile.TemporaryDirectory() as temporary:
            server.args = SimpleNamespace(
                suffix=temporary, alg='ilora', dataset='zsre', model='Qwen3-1.7B',
                cn=8, epoch=1, lr=0.0001, lora_alpha=8, lora_rank=2,
            )
            server._initialize_shared_qr()
            effective = model.values[W].detach() + 4 * server.global_lora[B] @ server.global_lora[A]
            torch.testing.assert_close(effective, original, atol=1e-5, rtol=1e-5)
            torch.testing.assert_close(server.global_lora[B].T @ server.global_lora[B], torch.eye(2))

            server.sampled_clients = sample_clients(self.a1, self.b1, self.a2, self.b2)
            server.aggregate()
            server.save_adapter()
            path = Path(server._adapter_path())
            saved = torch.load(path / 'lora_weights.pt', weights_only=True)
            saved_base = torch.load(path / 'merged_weights.pt', weights_only=True)
            replay = saved_base[W] + 4 * saved[B] @ saved[A]
            current = model.values[W].detach() + 4 * server.global_lora[B] @ server.global_lora[A]
            torch.testing.assert_close(replay, current)

    def test_ilora_thin_qr_matches_dense_qr_prefix(self):
        generator = torch.Generator().manual_seed(42)
        stacked_b = torch.randn(6, 4, generator=generator)
        stacked_a = torch.randn(4, 5, generator=generator)
        new_b, new_a = ilora._product_qr(stacked_b, stacked_a, rank=2)
        q, r = torch.linalg.qr(stacked_b @ stacked_a, mode='reduced')
        torch.testing.assert_close(new_b @ new_a, q[:, :2] @ r[:2, :], atol=1e-5, rtol=1e-5)


if __name__ == '__main__':
    unittest.main()
