"""Regression cases found during the complete shared-runtime review."""
import ast
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from transformers import BatchEncoding

from tests import test_shared_runtime as shared_support

ROOT = Path(__file__).resolve().parents[1]


class RuntimeReviewTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_gsm8k_fraction_is_not_its_denominator(self):
        from utils.evaluation_adapters import final_math_answer
        self.assertEqual(final_math_answer(r'The answer is \boxed{1/2}', 'gsm8k'), '1/2')
        self.assertEqual(final_math_answer(r'\boxed{\frac{1}{2}}', 'gsm8k'), '1/2')
        self.assertEqual(final_math_answer(r'\boxed{2.5}', 'gsm8k'), '2.5')
        self.assertIsNone(final_math_answer(r'\boxed{1/0}', 'gsm8k'))

    def test_zsre_generation_honors_limit_and_restores_tokenizer(self):
        from eval_zsre import generate_and_score

        class Tokenizer:
            padding_side = 'right'
            eos_token_id, pad_token_id, bos_token_id = 2, 0, 1
            def apply_chat_template(self, *args, **kw): return [1, 4]
            def pad(self, values, **kw):
                return BatchEncoding({'input_ids': torch.tensor(values['input_ids']),
                    'attention_mask': torch.tensor(values['attention_mask'])})
            def batch_decode(self, values, **kw): return ['answer'] * len(values)

        class Model:
            device = 'cpu'
            def eval(self): pass
            def generate(self, input_ids, generation_config, **kw):
                self.limit = generation_config.max_new_tokens
                return torch.cat([input_ids, torch.full((len(input_ids), self.limit), 7)], dim=1)

        model, tokenizer = Model(), Tokenizer()
        metrics, predictions = generate_and_score(model, tokenizer,
            [{'input_ids': 'question', 'label': 'answer'}], 1, max_new_tokens=3)
        self.assertEqual(model.limit, 3)
        self.assertEqual(metrics['max_tokens_rate'], 100.0)
        self.assertTrue(predictions[0]['hit_max_new_tokens'])
        self.assertEqual(tokenizer.padding_side, 'right')

    def test_zsre_cli_keeps_32_token_default_and_accepts_overrides(self):
        from utils.options import build_parser
        from eval_zsre import generation_metadata
        config = str(ROOT / 'config/config.zsre_fedit.yaml')
        with patch('sys.argv', ['eval_zsre.py', '--config', config]):
            args = build_parser(max_new_tokens_default=32).parse_args()
        self.assertEqual(args.max_new_tokens, 32)
        with patch('sys.argv', ['eval_zsre.py', '--config', config, '--max-new-tokens', '7']):
            args = build_parser(max_new_tokens_default=32).parse_args()
        self.assertEqual(args.max_new_tokens, 7)
        self.assertEqual(generation_metadata(SimpleNamespace(eos_token_id=2), max_new_tokens=7)['max_new_tokens'], 7)

    def test_zero_svd_update_preserves_a_learning_direction(self):
        from alg.fedmomentum import Server
        from tests.test_zsre_extra_lora import FakeModel, A, B
        server = object.__new__(Server)
        server.args = SimpleNamespace(lora_rank=1, lora_alpha=2)
        server.tau = 0.9999
        a, b = torch.ones(1, 4), torch.zeros(4, 1)
        server.model = FakeModel(a, b)
        server.global_lora = {A: a.clone(), B: b.clone()}
        server.sampled_clients = [SimpleNamespace(dataset={'train': [0]}, lora={A: a.clone(), B: b.clone()})]
        server.aggregate()
        updated_a = server.global_lora[A].detach().clone().requires_grad_()
        updated_b = server.global_lora[B].detach().clone().requires_grad_()
        ((updated_b @ updated_a - 1) ** 2).sum().backward()
        self.assertGreater(torch.count_nonzero(updated_b.grad).item(), 0)

    def test_localonly_last_client_interrupt_preserves_round_rng(self):
        from utils.experiment import same_tensors
        helper = shared_support.SharedRuntimeTest()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            reference = helper.simulate(root / 'full', 'localonly', rounds=1)
            with self.assertRaises(shared_support.Interrupted):
                helper.simulate(root / 'cut', 'localonly', rounds=1, interrupt_at=9)
            resumed = helper.simulate(root / 'cut', 'localonly', rounds=1)
            self.assertTrue(same_tensors(reference['rng'], resumed['rng']))

    def test_modified_evaluation_cache_is_rejected(self):
        helper = shared_support.SharedRuntimeTest()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            helper.simulate(root, 'fedit', rounds=1)
            path = next((root / 'exp/test/evaluation/round_001').glob('*/boolq.json'))
            record = json.loads(path.read_text())
            record['metrics']['acc,none'] = 1.0
            path.write_text(json.dumps(record))
            with self.assertRaisesRegex(ValueError, 'cache'):
                helper.simulate(root, 'fedit', rounds=1, eval_only=True)

    def test_progress_recovers_after_metadata_write_interruption(self):
        helper = shared_support.SharedRuntimeTest()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            helper.simulate(root, 'fedit', rounds=1)
            progress = root / 'exp/test/progress.json'
            progress.unlink()
            helper.simulate(root, 'fedit', rounds=1)
            self.assertEqual(json.loads(progress.read_text())['completed_rounds'], 1)

    def test_eval_only_preserves_original_training_target(self):
        from utils.experiment import ExperimentRun
        helper = shared_support.SharedRuntimeTest()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            helper.simulate(root, 'fedit', rounds=1)
            server = helper.last_server
            server.args.eval_only = True
            server.args.rnd = 30
            server.args.suffix = str(root / 'exp/test')
            ExperimentRun(server.args, server)
            progress = json.loads((root / 'exp/test/progress.json').read_text())
            self.assertEqual(progress['requested_rounds'], 1)

    def test_model_tokenizer_assets_are_part_of_identity(self):
        from utils.model_utils import local_model_fingerprint
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'config.json').write_text('{}')
            (root / 'model.safetensors').write_bytes(b'weights')
            (root / 'added_tokens.json').write_text('{"a":1}')
            before = local_model_fingerprint(root)
            (root / 'added_tokens.json').write_text('{"a":2}')
            after = local_model_fingerprint(root)
            self.assertNotEqual(before, after)
            (root / 'model.safetensors.index.json').write_text(json.dumps({'weight_map': {'a': '../elsewhere'}}))
            with self.assertRaisesRegex(ValueError, 'shard'):
                local_model_fingerprint(root)

    def test_nested_model_shards_are_supported(self):
        from utils.model_utils import local_model_fingerprint
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'config.json').write_text('{}')
            (root / 'weights').mkdir()
            (root / 'weights/part.safetensors').write_bytes(b'weights')
            (root / 'model.safetensors.index.json').write_text(json.dumps({'weight_map': {'a': 'weights/part.safetensors'}}))
            self.assertIn('weights/part.safetensors', local_model_fingerprint(root))

    def test_dataset_identity_paths_are_portable(self):
        from utils.dataset_adapters import prepare_data
        class Tokenizer:
            eos_token_id, pad_token_id = 2, 0
            def apply_chat_template(self, *a, **kw): return [1, 4]
            def __call__(self, *a, **kw): return {'input_ids': [5]}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for split in ('train', 'test'):
                (root / split).mkdir()
                (root / split / '0.jsonl').write_text('{"client_id":0,"input_ids":"q","label":"a"}\n')
            args = SimpleNamespace(task='zsre', dataset=str(root), alg='fedit', cn=None, max_length=16)
            data = prepare_data(args, Tokenizer())
            self.assertIn('train/0.jsonl', data['data_files'])
            self.assertTrue(all('\\' not in name for name in data['data_files']))

    def test_invalid_yaml_runtime_and_integer_parameters_fail_early(self):
        from utils.experiment import prepare_experiment, use_shared_runtime
        from utils.options import args_parser
        with self.assertRaisesRegex(ValueError, 'runtime'):
            use_shared_runtime(SimpleNamespace(runtime='sharedd'))
        with self.assertRaisesRegex(ValueError, 'shared'):
            use_shared_runtime(SimpleNamespace(runtime='legacy', task='zsre'))
        with patch('sys.argv', ['main.py', '--config', str(ROOT / 'config/experiments/zsre.yaml')]):
            original = args_parser()
        for key, value in [('bs', True), ('epoch', 0.5), ('step', 1.5), ('eval_limit', True), ('resume', 'invalid')]:
            args = SimpleNamespace(**vars(original))
            setattr(args, key, value)
            with self.subTest(key=key), self.assertRaises(ValueError):
                prepare_experiment(args)

    def test_legacy_launchers_point_to_existing_configs(self):
        for name in ('run_zsre_sequence.py', 'run_zsre_centralized.py'):
            tree = ast.parse((ROOT / name).read_text())
            paths = [node.value for node in ast.walk(tree) if isinstance(node, ast.Constant)
                     and isinstance(node.value, str) and 'config.zsre_' in node.value and node.value.endswith('.yaml')]
            self.assertTrue(paths)
            for value in paths:
                self.assertTrue((ROOT / value).is_file(), (name, value))

    def test_incomplete_sharded_model_is_rejected_during_preflight(self):
        from utils.experiment import prepare_experiment
        from utils.dataset_adapters import DATASETS
        from utils.options import args_parser
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'config.json').write_text('{"model_type":"qwen3"}')
            (root / 'model-00001-of-00002.safetensors').write_bytes(b'first shard')
            (root / 'model.safetensors.index.json').write_text(json.dumps({'weight_map': {
                'a': 'model-00001-of-00002.safetensors', 'b': 'model-00002-of-00002.safetensors'}}))
            with patch('sys.argv', ['main.py', '--config', str(ROOT / 'config/experiments/zsre.yaml'),
                                   '--model', str(root), '--cn', '1']):
                args = args_parser()
            prepared = {'adapter': DATASETS['zsre'], 'encoded': {0: [{}]}, 'data_files': {}, 'tasks': [], 'audit': {}}
            with patch('transformers.AutoTokenizer.from_pretrained', return_value=object()), \
                 patch('utils.experiment.prepare_data', return_value=prepared), \
                 patch('utils.experiment.validate_evaluator'):
                with self.assertRaisesRegex(ValueError, 'shard|missing'):
                    prepare_experiment(args)
