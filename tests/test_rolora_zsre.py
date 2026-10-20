"""Fast RoLoRA state/alternation checks without loading a language model."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from alg.rolora import Client, Server


A = 'layer.lora_A.default.weight'
B = 'layer.lora_B.default.weight'


class FakeModel:
    def __init__(self):
        self.values = {A: torch.nn.Parameter(torch.tensor([2.])),
                       B: torch.nn.Parameter(torch.tensor([0.]))}

    def parameters(self):
        return self.values.values()

    def named_parameters(self):
        return self.values.items()

    def state_dict(self):
        return {key: value.detach().clone() for key, value in self.values.items()}

    def load_state_dict(self, values, strict=False):
        for key, value in values.items():
            self.values[key].data.copy_(value)


class FakeTrainer:
    def train(self, model):
        active = [key for key, value in model.named_parameters() if value.requires_grad]
        assert len(active) == 1
        model.values[active[0]].data.add_(1.)
        return 1.


class RoLoRAZsRETest(unittest.TestCase):
    def test_alternation_aggregation_and_full_checkpoint(self):
        model = FakeModel()
        client = object.__new__(Client)
        client.server = SimpleNamespace(round=0)
        client.delay = 1.
        client.trainer = FakeTrainer()
        client.run(model)
        self.assertEqual(set(client.lora), {B})
        self.assertFalse(model.values[A].requires_grad)
        self.assertTrue(model.values[B].requires_grad)

        with tempfile.TemporaryDirectory() as temporary:
            server = object.__new__(Server)
            server.model = model
            server.global_lora = {A: torch.tensor([2.]), B: torch.tensor([0.])}
            server.round = 0
            server.sampled_clients = [
                SimpleNamespace(lora={B: torch.tensor([1.])}, dataset={'train': [1, 2]}),
                SimpleNamespace(lora={B: torch.tensor([3.])}, dataset={'train': [3, 4]}),
            ]
            server.aggregate()
            self.assertEqual(set(server.global_lora), {A, B})
            self.assertEqual(server.global_lora[A].item(), 2.)
            self.assertEqual(server.global_lora[B].item(), 2.)

            client.server.round = 1
            client.run(model)
            self.assertEqual(set(client.lora), {A})
            self.assertTrue(model.values[A].requires_grad)
            self.assertFalse(model.values[B].requires_grad)
            server.round = 1
            server.sampled_clients = [
                SimpleNamespace(lora={A: torch.tensor([4.])}, dataset={'train': [1, 2]}),
                SimpleNamespace(lora={A: torch.tensor([6.])}, dataset={'train': [3, 4]}),
            ]
            server.aggregate()
            self.assertEqual(server.global_lora[A].item(), 5.)
            self.assertEqual(server.global_lora[B].item(), 2.)

            server.args = SimpleNamespace(
                suffix=temporary, alg='rolora', dataset='zsre', model='Qwen3-1.7B',
                cn=8, epoch=1, lr=0.0001,
            )
            server.save_adapter()
            path = Path(server._adapter_path()) / 'lora_weights.pt'
            saved = torch.load(path, map_location='cpu', weights_only=True)
            self.assertEqual(set(saved), {A, B})


if __name__ == '__main__':
    unittest.main()
