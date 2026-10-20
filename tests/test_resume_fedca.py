"""Round-boundary recovery checks with tiny CPU models; no model downloads."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from tests.test_fedca import build_federation, make_args, run_round, tiny_peft_qwen3
from tools.resume_fedca import continue_training, find_checkpoint, restore_adapter


class ResumeTest(unittest.TestCase):
    def test_fedexp_continuation_matches_uninterrupted_weights(self):
        with tempfile.TemporaryDirectory() as uninterrupted, tempfile.TemporaryDirectory() as resumed:
            settings = dict(ca_lambda=0.0, ca_eta_mode="fedexp", rnd=2, sr=1.0, mode="prototype")
            args = make_args(uninterrupted, **settings)
            reference, clients = build_federation(tiny_peft_qwen3(seed=13), args)
            for index in range(2):
                reference.round = index
                run_round(reference, clients)

            args = make_args(resumed, **settings)
            first, clients = build_federation(tiny_peft_qwen3(seed=13), args)
            run_round(first, clients)
            folder = Path(first._adapter_path())
            source = (folder / "round_001/lora_weights.pt").read_bytes()
            observed_rounds = []

            def factory(args):
                server, clients = build_federation(tiny_peft_qwen3(seed=13), args)

                def run():
                    observed_rounds.append(server.round)
                    run_round(server, clients)

                server.run = run
                server.test_all = lambda: {"eval_loss": 1.0}
                return server

            continue_training(args, server_factory=factory)
            final = torch.load(folder / "lora_weights.pt", weights_only=True)
            for key, expected in reference.global_lora.items():
                self.assertTrue(torch.equal(expected, final[key]), key)
            self.assertEqual(observed_rounds, [1])
            self.assertEqual(source, (folder / "round_001/lora_weights.pt").read_bytes())
            self.assertEqual(len((folder / "fedca_rounds.jsonl").read_text().splitlines()), 2)
            state = json.loads((Path(resumed) / "resume_state.json").read_text())
            self.assertEqual(state["completed_rounds"], 2)
            self.assertEqual(state["learning_rate"], args.lr * 0.99)
            self.assertFalse((Path(resumed) / "resume_training.lock").exists())

    def test_refuses_mismatched_or_uncheckpointed_records(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            (folder / "round_001").mkdir()
            (folder / "round_001/lora_weights.pt").touch()
            record = {"round": 0, "completed_rounds": 1, "mode": "fedexp", "lambda": 0.0, "weights": [1.0]}
            log = folder / "fedca_rounds.jsonl"
            log.write_text(json.dumps(record) + "\n")
            args = SimpleNamespace(ca_eta_mode="fedexp", ca_lambda=0.0, cn=1, rnd=10)
            self.assertEqual(find_checkpoint(folder, args)[0], 1)
            args.ca_lambda = 1.0
            with self.assertRaisesRegex(ValueError, "do not match"):
                find_checkpoint(folder, args)
            args.ca_lambda = 0.0
            with log.open("a") as handle:
                handle.write(json.dumps(dict(record, round=1, completed_rounds=2)) + "\n")
            with self.assertRaisesRegex(ValueError, "disagree"):
                find_checkpoint(folder, args)

    def test_failed_round_does_not_publish_final_adapter(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = make_args(tmp, ca_lambda=0.0, ca_eta_mode="fedexp", rnd=2, sr=1.0, mode="prototype")
            server, clients = build_federation(tiny_peft_qwen3(), args)
            run_round(server, clients)

            def fail():
                raise RuntimeError("simulated failure")

            server.run = fail
            with self.assertRaisesRegex(RuntimeError, "simulated failure"):
                continue_training(args, server_factory=lambda _: server)
            self.assertFalse((Path(server._adapter_path()) / "lora_weights.pt").exists())
            self.assertFalse((Path(tmp) / "resume_training.lock").exists())
            self.assertEqual(json.loads((Path(tmp) / "resume_state.json").read_text())["status"], "failed")

    def test_rejects_invalid_tensors_before_loading_any_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = tiny_peft_qwen3()
            before = {k: v.clone() for k, v in model.state_dict().items() if "lora_" in k}
            bad = {k: v.clone() for k, v in before.items()}
            next(iter(bad.values())).flatten()[0] = float("nan")
            checkpoint = Path(tmp) / "invalid.pt"
            torch.save(bad, checkpoint)
            with self.assertRaisesRegex(ValueError, "Invalid checkpoint tensor"):
                restore_adapter(model, checkpoint)
            for key, expected in before.items():
                self.assertTrue(torch.equal(model.state_dict()[key], expected))


if __name__ == "__main__":
    unittest.main()
