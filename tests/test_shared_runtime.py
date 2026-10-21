"""Exercise the real runner, algorithms and Trainer across dataset adapters."""
import json
import os
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from utils.dataset_adapters import DATASETS, prepare_data
from utils.experiment import same_tensors
from utils.method_state import METHODS
from utils.runtime_utils import CheckpointStore, capture_rng

ROOT = Path(__file__).resolve().parents[1]


class Interrupted(Exception):
    pass


def tiny_model(args):
    return Qwen3ForCausalLM(Qwen3Config(vocab_size=32, hidden_size=16, intermediate_size=24,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
        attention_dropout=0.2, max_position_embeddings=64, pad_token_id=0,
        bos_token_id=1, eos_token_id=2, use_cache=False))


class SharedRuntimeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def simulate(self, directory, method, task='commonsense170k', interrupt_at=None,
                 marker_cut=False, rounds=2, eval_limit=None, eval_only=False,
                 epochs=1, batch_size=1, samples=6, eval_failure=False):
        from main import FedSim
        from utils.options import args_parser
        import utils.experiment as runtime
        directory.mkdir(parents=True, exist_ok=True)
        for name in ('task0', 'task1'):
            (directory / f'{name}.jsonl').write_text('{"input_ids":"q","label":"a"}\n', encoding='utf-8')
        if task == 'commonsense170k':
            names = ['boolq', 'piqa']
        elif task == 'zsre':
            names = ['client_0', 'client_1']
        else:
            names = ['gsm8k', 'math']
        tasks = [(name, directory / f'task{cid}.jsonl', cid) for cid, name in enumerate(names)]
        encoded = {cid: [{'input_ids': [1, 4 + cid, 5 + i, 6, 2], 'labels': [-100, -100, 5+i, 6, 2]}
                         for i in range(samples)] for cid in range(2)}
        identity = 'shared-test-' + method + '-' + task
        prepared = {'identity': identity, 'identity_spec': {'test': method}, 'encoded': encoded,
                    'dataset': directory, 'adapter': DATASETS[task], 'tasks': tasks, 'report': {}}
        original_save = CheckpointStore.save
        original_json = runtime.atomic_json
        saves, calls = 0, []

        def save(store, state):
            nonlocal saves
            original_save(store, state)
            saves += 1
            if saves == interrupt_at:
                raise Interrupted()

        def write(path, payload):
            original_json(path, payload)
            if marker_cut and Path(path).name == 'complete.json' and 'round_001' in str(path):
                raise Interrupted()

        def evaluate(adapter, model, tokenizer, name, path, args):
            calls.append(name)
            if eval_failure and len(calls) == 2:
                raise Interrupted()
            random.random()
            torch.rand(3)
            metric = DATASETS[task].primary_metric
            return {'metrics': {metric: 0.5}, 'num_samples': 1}

        previous = Path.cwd()
        try:
            os.chdir(directory)
            argv = ['main.py', '--config', str(ROOT / 'config/experiments' / f'{task}.yaml'),
                '--alg', method, '--suffix', 'test', '--rnd', str(rounds), '--test_gap', '1',
                '--cn', '2', '--bs', str(batch_size), '--epoch', str(epochs),
                '--grad_accum', '2', '--checkpoint_steps', '1',
                '--lr', '0.001', '--lora_rank', '2', '--lora_alpha', '4', '--lora_dropout', '0.2']
            if method == 'fedca':
                argv += ['--ca_probe_size', '1', '--ca_eta_grid', '1,2', '--ca_kappa', '1', '--ca_beta', '0.5']
            if eval_limit:
                argv += ['--eval_limit', str(eval_limit)]
            if eval_only:
                argv += ['--eval_only']
            with patch('sys.argv', argv):
                args = args_parser()
            args._experiment_prepared = prepared
            with patch('alg.ftbase.load_model', side_effect=tiny_model), \
                 patch('alg.ftbase.load_tokenizer', return_value=SimpleNamespace(pad_token_id=0)), \
                 patch('alg.ftbase.device_config', side_effect=lambda args: [1] * args.cn), \
                 patch('main.get_logger', return_value=MagicMock()), \
                 patch('utils.experiment.evaluate_task', side_effect=evaluate), \
                 patch('utils.experiment.importlib.metadata.version', return_value='test'), \
                 patch.object(CheckpointStore, 'save', save), \
                 patch('utils.experiment.atomic_json', side_effect=write):
                simulation = FedSim(args)
                self.last_server = simulation.server
        finally:
            os.chdir(previous)
        self.last_calls = calls
        return CheckpointStore(directory / 'exp/test/resume', identity).load()

    def test_all_methods_resume_mid_client_and_completed_client(self):
        for method in METHODS:
            with self.subTest(method=method), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                reference = self.simulate(root / 'full', method)
                for cut in (2, 5, 9, 10):
                    target = root / f'cut{cut}'
                    with self.assertRaises(Interrupted):
                        self.simulate(target, method, interrupt_at=cut)
                    resumed = self.simulate(target, method)
                    self.assertTrue(same_tensors(reference['server'], resumed['server']), (method, cut))
                    self.assertTrue(same_tensors(reference['rng'], resumed['rng']), (method, cut, 'rng'))
                    for rnd in (1, 2):
                        a = json.loads((root / 'full/exp/test' / f'round_{rnd:03d}/training_metrics.json').read_text())
                        b = json.loads((target / 'exp/test' / f'round_{rnd:03d}/training_metrics.json').read_text())
                        self.assertEqual(a, b, (method, cut, 'training metrics'))

    def test_snapshot_before_checkpoint_replays_randomized_and_base_merging_aggregation(self):
        for method in ('fedmomentum', 'fedexlora'):
            with self.subTest(method=method), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                reference = self.simulate(root / 'full', method)
                with self.assertRaises(Interrupted):
                    self.simulate(root / 'cut', method, marker_cut=True)
                resumed = self.simulate(root / 'cut', method)
                self.assertTrue(same_tensors(reference['server'], resumed['server']))

    def test_same_lifecycle_for_zsre_and_math(self):
        for task in ('zsre', 'metamathqa'):
            for method in ('fedit', 'fedca', 'localonly', 'ffalora', 'fedexlora'):
                with self.subTest(task=task, method=method), tempfile.TemporaryDirectory() as tmp:
                    root = Path(tmp)
                    reference = self.simulate(root / 'full', method, task)
                    with self.assertRaises(Interrupted):
                        self.simulate(root / 'cut', method, task, interrupt_at=2)
                    resumed = self.simulate(root / 'cut', method, task)
                    self.assertTrue(same_tensors(reference['server'], resumed['server']))

    def test_old_round_evaluation_restores_base_frozen_factors_and_rng(self):
        for method in ('fedexlora', 'fedmomentum', 'ffalora', 'localonly'):
            with self.subTest(method=method), tempfile.TemporaryDirectory() as tmp:
                self.simulate(Path(tmp), method)
                server = self.last_server
                before = server.export_state()
                rng = capture_rng()
                # Force a new diagnostic evaluation of a historical snapshot.
                server.args.eval_limit = 1
                with patch('utils.experiment.evaluate_task', return_value={'metrics': {'generation_accuracy': 0.5}}), \
                     patch('utils.experiment.importlib.metadata.version', return_value='test'):
                    server.experiment_run.evaluate(1)
                self.assertTrue(same_tensors(before, server.export_state()))
                self.assertTrue(same_tensors(rng, capture_rng()))

    def test_completed_run_reuses_evaluation_and_diagnostic_cache_is_separate(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            self.simulate(path, 'fedit', eval_limit=1)
            self.simulate(path, 'fedit', eval_limit=1)
            self.assertEqual(self.last_calls, [])
            self.simulate(path, 'fedit', eval_only=True)
            self.assertEqual(len(self.last_calls), 4)

    def test_config_inheritance_and_method_defaults(self):
        from utils.options import args_parser, read_config
        config = read_config(ROOT / 'config/experiments/commonsense170k.yaml')
        self.assertEqual((config['rnd'], config['test_gap'], config['step']), (30, 5, 0))
        for method, flag, expected in [('fedrotlora', '--lam', 0.3), ('fedmomentum', '--residual_threshold', 0.8)]:
            with patch('sys.argv', ['main.py', '--config', str(ROOT / 'config/experiments/commonsense170k.yaml'),
                                    '--alg', method, flag, str(expected)]):
                args = args_parser()
            self.assertEqual(getattr(args, flag[2:]), expected)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'loop.yaml'
            path.write_text('extends: loop.yaml\n')
            with self.assertRaises(ValueError):
                read_config(path)

    def test_frozen_a_is_part_of_model_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = self.simulate(Path(tmp), 'ffalora', rounds=1)
            self.assertTrue(any('lora_A' in name for name in state['server']['adapter']))
            self.assertTrue(all(not p.requires_grad for name, p in self.last_server.model.named_parameters() if 'lora_A' in name))

    def test_multi_epoch_partial_accumulation_continuation(self):
        for method in ('fedit', 'ffalora', 'fedca'):
            with self.subTest(method=method), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                options = dict(epochs=2, batch_size=2, samples=5)
                reference = self.simulate(root / 'full', method, **options)
                for cut in (2, 3, 4):
                    target = root / f'cut{cut}'
                    with self.assertRaises(Interrupted):
                        self.simulate(target, method, interrupt_at=cut, **options)
                    resumed = self.simulate(target, method, **options)
                    self.assertTrue(same_tensors(reference['server'], resumed['server']))
                    a = json.loads((root / 'full/exp/test/round_002/training_metrics.json').read_text())
                    b = json.loads((target / 'exp/test/round_002/training_metrics.json').read_text())
                    self.assertEqual(a, b)

    def test_evaluation_failure_does_not_retrain_committed_round(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            with self.assertRaises(Interrupted):
                self.simulate(path, 'fedexlora', rounds=1, eval_failure=True)
            identity = 'shared-test-fedexlora-commonsense170k'
            store = CheckpointStore(path / 'exp/test/resume', identity)
            before = store.load()
            self.assertEqual(before['completed_rounds'], 1)
            self.simulate(path, 'fedexlora', rounds=1)
            self.assertTrue(same_tensors(before, store.load()))
            self.assertEqual(self.last_calls, ['piqa'])

    def test_math_and_zsre_encoding_supervises_eos_and_rejects_truncation(self):
        class Tokenizer:
            eos_token_id = 2
            pad_token_id = 0
            def apply_chat_template(self, *a, **kw): return [1, 4]
            def __call__(self, *a, **kw): return {'input_ids': [5, 6]}
        for adapter in DATASETS.values():
            result = adapter.encode(Tokenizer(), {'input_ids': 'question', 'label': 'answer'}, 8)
            self.assertEqual(result['labels'], [-100, -100, 5, 6, 2])
            with self.assertRaises(ValueError):
                adapter.encode(Tokenizer(), {'input_ids': 'question', 'label': 'answer'}, 4)

    def test_jsonl_preserves_unicode_separators_in_math_text(self):
        from utils.dataset_adapters import read_rows
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'math.jsonl'
            row = {'input_ids': 'a\u2028b\u0085c\u2029d', 'label': 'reasoning'}
            path.write_text(json.dumps(row, ensure_ascii=False) + '\n', encoding='utf-8')
            self.assertEqual(read_rows(path), [row])

    def test_source_identity_is_portable_across_checkout_line_endings(self):
        from utils.runtime_utils import source_sha256
        with tempfile.TemporaryDirectory() as tmp:
            a, b = Path(tmp) / 'a.py', Path(tmp) / 'b.py'
            a.write_bytes(b'a = 1\nb = 2\n')
            b.write_bytes(b'a = 1\r\nb = 2\r\n')
            self.assertEqual(source_sha256(a), source_sha256(b))

    def test_reviewed_legacy_identity_preserved_but_unrelated_changes_rejected(self):
        from utils.commonsense_fedca import preserve_baseline_identity
        from utils.runtime_utils import fingerprint
        record = json.loads((ROOT / 'utils/legacy_shared_compat.json').read_text())
        names = ['main.py', 'utils/train_utils.py', 'utils/commonsense_framework.py', 'utils/commonsense_fedca.py']
        old = {'training': {'alg': 'fedca', 'lr': 0.0001}, 'data': 'same',
               'code': {k: record['sources'][k]['before'] for k in names}}
        current = {**old, 'code': {k: record['sources'][k]['after'] for k in names}}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            (path / 'run_manifest.json').write_text(json.dumps({'identity': fingerprint(old), 'identity_spec': old}))
            self.assertEqual(preserve_baseline_identity(path, current), old)
            extended = {**current, 'code': {**current['code'],
                'utils/runtime_utils.py': record['dependencies']['utils/runtime_utils.py']}}
            self.assertEqual(preserve_baseline_identity(path, extended), old)
            changed = {**current, 'data': 'different'}
            self.assertEqual(preserve_baseline_identity(path, changed), changed)
            changed = {**current, 'code': {**current['code'], 'main.py': 'unreviewed'}}
            self.assertEqual(preserve_baseline_identity(path, changed), changed)

    def test_math_scoring_uses_pinned_normalization_and_generation_records(self):
        from utils.evaluation_adapters import math_utilities, final_math_answer, generate_math
        try:
            utils = math_utilities()
        except RuntimeError as exc:
            if 'install' not in str(exc) and 'lm_eval' not in str(exc):
                raise
            self.skipTest(str(exc))
        self.assertEqual(final_math_answer('reasoning \\boxed{\\frac{1}{2}}', 'math', utils), r'\frac{1}{2}')
        self.assertTrue(utils.is_equiv(r'\frac12', r'\frac{1}{2}'))
        self.assertIsNone(final_math_answer('no final answer', 'math', utils))
        self.assertEqual(final_math_answer('100 then #### -1,234', 'gsm8k'), '-1234')

        class Tokenizer:
            pad_token_id, eos_token_id, bos_token_id = 0, 2, 1
            padding_side = 'right'
            completion = ''
            def apply_chat_template(self, *a, **kw): return [1, 4]
            def pad(self, rows, **kw):
                return {'input_ids': torch.tensor([r['input_ids'] for r in rows]),
                        'attention_mask': torch.ones(len(rows), 2, dtype=torch.long)}
            def decode(self, *a, **kw): return self.completion

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.value = torch.nn.Parameter(torch.ones(1))
            def generate(self, input_ids, **kw):
                return torch.cat([input_ids, torch.tensor([[9, 2]] * len(input_ids))], dim=1)

        tokenizer = Tokenizer()
        options = SimpleNamespace(eval_batch_size=1, max_new_tokens=8, eval_max_length=32)
        for task, answer, text in [('gsm8k', '2', r'So \boxed{2}'), ('math', r'\frac{1}{2}', r'So \boxed{\frac12}')]:
            tokenizer.completion = text
            result = generate_math(Model(), tokenizer, [{'input_ids': 'question', 'answer': answer}], task, options)
            self.assertEqual(result['metrics']['exact_match'], 1.0)
            self.assertFalse(result['predictions'][0]['hit_max_new_tokens'])
        self.assertEqual(tokenizer.padding_side, 'right')

    def test_residual_svd_scaling_and_fixed_rank_shape(self):
        from alg.fedmomentum import Server
        from tests.test_zsre_extra_lora import FakeModel, A, B, W, sample_clients
        torch.manual_seed(11)
        clients = sample_clients(torch.randn(2, 4), torch.randn(4, 2),
                                 torch.randn(2, 4), torch.randn(4, 2))
        server = object.__new__(Server)
        server.args = SimpleNamespace(lora_rank=2, lora_alpha=6)
        server.tau = 1.0
        server.sampled_clients = clients
        server.model = FakeModel(torch.zeros(2, 4), torch.zeros(4, 2))
        server.global_lora = {A: torch.zeros(2, 4), B: torch.zeros(4, 2)}
        total = sum(len(c.dataset['train']) for c in clients)
        expected = sum(len(c.dataset['train']) / total * (c.lora[B] @ c.lora[A]) for c in clients) * 3
        server.aggregate()
        state = server.model.state_dict()
        actual = state[W] + 3 * state[B] @ state[A]
        self.assertTrue(torch.allclose(expected, actual, atol=1e-5, rtol=1e-5))
        for client in clients:
            client.lora[B].zero_()
        server.aggregate()
        self.assertEqual(server.global_lora[A].shape, (2, 4))
        self.assertEqual(server.global_lora[B].shape, (4, 2))
