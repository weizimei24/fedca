"""Continuation must match uninterrupted optimization, including dropout RNG."""
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch, MagicMock

import torch

from utils.commonsense_framework import evaluation_rounds
from utils.commonsense_eval import canonical_doc, build_task, evaluate_round
from utils.commonsense_training import (
    CheckpointStore, DOMAINS, adapter_state, factor_average, sha256, atomic_torch, encode_example, collate)

ROOT = Path(__file__).resolve().parents[1]
COMMONSENSE_DATA = ROOT / 'dataset/commonsense170k_client1000_seed42'
HAS_COMMONSENSE_DATA = all((COMMONSENSE_DATA / 'benchmarks' / f'{domain}.jsonl').is_file() for domain in DOMAINS)
HAS_QWEN_TOKENIZER = (ROOT / 'Qwen3-1.7B/tokenizer_config.json').is_file()


class TinyCausalModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(16, 8)
        self.embedding.weight.requires_grad_(False)
        self.lora_A = torch.nn.Linear(8, 4, bias=False)
        self.dropout = torch.nn.Dropout(0.3)
        self.lora_B = torch.nn.Linear(4, 16, bias=False)

    def forward(self, input_ids, attention_mask, labels=None, use_cache=False):
        logits = self.lora_B(self.dropout(self.lora_A(self.embedding(input_ids))))
        loss = None if labels is None else torch.nn.functional.cross_entropy(logits[:, :-1].reshape(-1, 16), labels[:, 1:].reshape(-1), ignore_index=-100)
        return SimpleNamespace(loss=loss, logits=logits)


class Interrupted(Exception):
    pass


class CommonsenseTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def simulate(self, directory, method, interrupt_at=None, rounds=2, step=0, epochs=1, extra_args=()):
        from main import FedSim
        from utils.options import args_parser
        clients = {cid: [{"input_ids": [1, 2, (cid+i)%14+1, 3, 4], "labels": [-100, -100, (cid+i)%14+1, 3, 4]}
                             for i in range(6)] for cid in range(8)}
        identity = 'framework-test-'+method
        prepared = {'identity': identity, 'identity_spec': {'test': method},
                    'encoded': clients, 'dataset': ROOT / 'dataset/commonsense170k_client1000_seed42',
                    'report': {'diagnostic_budget': False}}
        original_save = CheckpointStore.save
        count = 0

        def save_then_interrupt(store, payload):
            nonlocal count
            original_save(store, payload)
            count += 1
            if count == interrupt_at:
                raise Interrupted()

        calls = []
        def evaluate(*args, **kwargs):
            calls.append(args[7])
            return {'macro_acc': 0.5, 'per_task': {d: {'acc,none': 0.5} for d in DOMAINS}}

        previous = Path.cwd()
        directory.mkdir(parents=True, exist_ok=True)
        try:
            os.chdir(directory)
            argv = ['main.py', '--config', str(ROOT / f'config/config.commonsense1000_{method}.yaml'),
                    '--suffix', 'test', '--rnd', str(rounds), '--step', str(step), '--grad_accum', '2',
                    '--checkpoint_steps', '1', '--lr', '0.001', '--epoch', str(epochs), *extra_args]
            with patch('sys.argv', argv):
                args = args_parser()
            args._commonsense_prepared = prepared
            with patch('alg.ftbase.load_model', side_effect=lambda args: TinyCausalModel()), \
                 patch('alg.ftbase.get_peft_model', side_effect=lambda model, config: model), \
                 patch('alg.ftbase.load_tokenizer', return_value=SimpleNamespace(pad_token_id=0)), \
                 patch('alg.ftbase.device_config', side_effect=lambda args: [1]*args.cn), \
                 patch('main.get_logger', return_value=MagicMock()), \
                 patch('utils.commonsense_framework.evaluate_round', side_effect=evaluate), \
                 patch.object(CheckpointStore, 'save', save_then_interrupt):
                FedSim(args)
        finally:
            os.chdir(previous)
        self.last_evaluation_rounds = calls
        return CheckpointStore(directory / 'exp/test/resume', identity).load()

    def test_exact_optimizer_and_round_continuation_all_three_methods(self):
        # Covers an active optimizer update, a completed client, and the round
        # commit. Eight clients are used, so skipping pending FedIT updates and
        # keeping independent local adapters are tested through the real runner.
        for method in ("fedit", "centralized", "localonly"):
            with self.subTest(method=method), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                baseline = self.simulate(root / "full", method)
                cuts = (2, 5, 6, 34) if method != "centralized" else (2, 25, 26, 27)
                for cut in cuts:
                    directory = root / f"cut{cut}"
                    with self.assertRaises(Interrupted):
                        self.simulate(directory, method, cut)
                    recovered = self.simulate(directory, method)
                    self.assertEqual(recovered["completed_rounds"], 2)
                    for field in ("global", "local"):
                        expected, actual = baseline[field], recovered[field]
                        if field == "local":
                            for cid in expected:
                                for name in expected[cid]:
                                    self.assertTrue(torch.equal(expected[cid][name], actual[cid][name]), (method, cut, cid, name))
                        else:
                            for name in expected:
                                self.assertTrue(torch.equal(expected[name], actual[name]), (method, cut, name))
                    for rnd in (1, 2):
                        a = json.loads((root / "full/exp/test" / f"round_{rnd:03d}/training_metrics.json").read_text())
                        b = json.loads((directory / 'exp/test' / f"round_{rnd:03d}/training_metrics.json").read_text())
                        self.assertEqual(a, b)

    def test_checkpoint_commit_ignores_uncommitted_slot_and_rejects_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = CheckpointStore(root, "one")
            state = {"completed_rounds": 3, "pending": {}, "active": None, "x": torch.ones(2)}
            store.save(state)
            # A process dying before pointer replacement must not expose a
            # half-written, newer checkpoint.
            (root / "state_0.pt").write_bytes(b"incomplete next write")
            self.assertEqual(store.load()["completed_rounds"], 3)
            with self.assertRaises(ValueError):
                CheckpointStore(root, "changed-data-or-settings").load()
            meta = json.loads((root / "latest.json").read_text())
            (root / meta["file"]).write_bytes(b"corrupt committed slot")
            with self.assertRaises(ValueError):
                store.load()

    def test_continuation_recreates_private_shuffle_across_epoch_boundary(self):
        for method in ('fedit', 'centralized', 'localonly'):
            with self.subTest(method=method), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                baseline = self.simulate(root / 'full', method, epochs=2)
                cut = 26 if method == 'centralized' else 5
                with self.assertRaises(Interrupted):
                    self.simulate(root / 'resumed', method, interrupt_at=cut, epochs=2)
                recovered = self.simulate(root / 'resumed', method, epochs=2)
                for name in baseline['global']:
                    self.assertTrue(torch.equal(baseline['global'][name], recovered['global'][name]))
                for cid, state in baseline['local'].items():
                    for name in state:
                        self.assertTrue(torch.equal(state[name], recovered['local'][cid][name]))

    def test_factor_average_matches_existing_fedit_server(self):
        from alg.ftbase import FTBaseServer
        states = [{"lora_A": torch.tensor([0.123+i, -0.31*i], dtype=torch.float32)} for i in range(8)]
        sizes = [3+i for i in range(8)]
        clients = [SimpleNamespace(id=i, dataset={"train": list(range(sizes[i]))}, lora=states[i]) for i in range(8)]
        class Receiver:
            def load_state_dict(self, state, strict):
                pass
        server = SimpleNamespace(sampled_clients=clients, model=Receiver())
        FTBaseServer.aggregate(server)
        self.assertTrue(torch.equal(server.global_lora["lora_A"], factor_average(states, sizes)["lora_A"]))

    @unittest.skipUnless(HAS_COMMONSENSE_DATA, 'local Commonsense benchmark files are not available')
    def test_native_tasks_use_correct_multiple_input_winogrande(self):
        root = ROOT / "dataset/commonsense170k_client1000_seed42/benchmarks"
        for domain in DOMAINS:
            task = build_task(domain, root / f"{domain}.jsonl")
            self.assertEqual(bool(task.multiple_input), domain == "winogrande")
            doc = task.eval_docs[0]
            target = task.doc_to_text(doc) if task.multiple_input else task.doc_to_target(doc)
            scores = [(-0.1 if i == target else -20.0, False) for i in range(len(task.doc_to_choice(doc)))]
            self.assertEqual(task.process_results(doc, scores)["acc"], 1.0)
        example = {"sentence": "The _ is small.", "option1": "cat", "option2": "dog", "answer": "2"}
        doc = canonical_doc("winogrande", example)
        self.assertEqual(doc, {"query": 1, "target": "is small.", "choices": ["The cat", "The dog"]})

    def test_evaluation_interval_and_final_round(self):
        args = SimpleNamespace(eval_rounds=None, test_gap=5, rnd=30)
        self.assertEqual(evaluation_rounds(args), [5, 10, 15, 20, 25, 30])
        args.rnd = 12
        self.assertEqual(evaluation_rounds(args), [5, 10, 12])

    def test_compatibility_commands_resolve_to_original_main_and_yaml(self):
        from tools.run_commonsense import framework_command
        from utils.options import args_parser
        for method in ('fedit', 'centralized', 'localonly'):
            command = framework_command(['--method', method, '--rounds', '30', '--eval-every', '5'])
            self.assertEqual(command[0], 'main.py')
            with patch('sys.argv', command):
                args = args_parser()
            self.assertEqual((args.runtime, args.task), ('shared', 'commonsense170k'))
            self.assertEqual((args.alg, args.rnd, args.test_gap), (method, 30, 5))
            self.assertEqual(args.cn, 1 if method == 'centralized' else None)
            self.assertEqual(args.step, 0)  # Full epoch; budget derives from data.

    def test_real_fedsim_loop_honors_interval_instead_of_forcing_last_ten(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.simulate(Path(tmp), 'centralized', rounds=12, step=1)
            self.assertEqual(self.last_evaluation_rounds, [5, 10, 12])

    @unittest.skipUnless(HAS_QWEN_TOKENIZER and (COMMONSENSE_DATA / 'train/1.jsonl').is_file(),
                         'local Qwen tokenizer and Commonsense training data are not available')
    def test_real_qwen_labels_supervise_complete_answer_eos_and_no_padding(self):
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(ROOT / "Qwen3-1.7B", local_files_only=True)
        source = ROOT / "dataset/commonsense170k_client1000_seed42/train/1.jsonl"
        rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines()]
        encoded = [encode_example(tokenizer, r, 1536) for r in rows]
        longest_idx = max(range(len(encoded)), key=lambda i: len(encoded[i]["input_ids"]))
        item = encoded[longest_idx]
        self.assertEqual(len(item["input_ids"]), 1123)
        supervised = [x for x in item["labels"] if x != -100]
        self.assertEqual(supervised[-1], tokenizer.eos_token_id)
        self.assertEqual(tokenizer.decode(supervised[:-1]), rows[longest_idx]["label"])
        with self.assertRaisesRegex(ValueError, "no silent truncation"):
            encode_example(tokenizer, rows[longest_idx], 512)
        batch = collate([encoded[0], item], tokenizer.pad_token_id, torch.device("cpu"))
        self.assertTrue(torch.all(batch["labels"][batch["attention_mask"] == 0] == -100))
        self.assertEqual(batch["input_ids"].shape[1], 1123)

    @unittest.skipUnless(HAS_COMMONSENSE_DATA, 'local Commonsense benchmark files are not available')
    def test_partial_evaluation_resume_local_selection_and_full_limit_separation(self):
        # Real task objects + evaluator boundary replacement: isolate recovery
        # behavior without repeating expensive GPU inference in a unit test.
        from utils.commonsense_eval import ensure_runtime
        ensure_runtime()
        torch.manual_seed(3)
        model = TinyCausalModel()
        adapters = {str(cid): {name: torch.full_like(value, cid+0.125)
                              for name, value in adapter_state(model).items()} for cid in range(8)}
        calls = []
        should_fail = True

        class Wrapper:
            def __init__(self, **kwargs):
                pass

        def evaluator_call(**kwargs):
            nonlocal should_fail
            name = kwargs["tasks"][0].get_config("task")
            cid = DOMAINS.index(name.removeprefix("cs170k_"))
            calls.append(cid)
            self.assertTrue(all(torch.equal(t, adapters[str(cid)][k]) for k, t in adapter_state(model).items()))
            if cid == 2 and should_fail:
                should_fail = False
                raise RuntimeError("simulated evaluator failure")
            return {"results": {name: {"acc,none": 0.5}}, "n-samples": {name: {"effective": 1}}}

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            snapshot = out / "snapshot.pt"
            atomic_torch(snapshot, {"identity": "eval-test", "round": 5, "adapters": adapters})
            args = (model, None, ROOT / "dataset/commonsense170k_client1000_seed42", snapshot,
                    sha256(snapshot), out, "localonly", 5, "eval-test")
            with patch("utils.commonsense_eval.ensure_runtime", return_value=Wrapper), \
                 patch("lm_eval.evaluator.simple_evaluate", side_effect=evaluator_call):
                with self.assertRaisesRegex(RuntimeError, "simulated"):
                    evaluate_round(*args, limit=1)
                self.assertEqual(calls, [0, 1, 2])
                summary = evaluate_round(*args, limit=1)
                self.assertEqual(calls, [0, 1, 2, 2, 3, 4, 5, 6, 7])
                self.assertEqual(summary["macro_acc"], 0.5)
                before = len(calls)
                full = evaluate_round(*args, limit=None)
                self.assertEqual(len(calls) - before, 8)
                self.assertNotEqual(summary["evaluation_id"], full["evaluation_id"])
                self.assertTrue(summary["diagnostic_only"])
                self.assertFalse(full["diagnostic_only"])


if __name__ == "__main__":
    unittest.main()
