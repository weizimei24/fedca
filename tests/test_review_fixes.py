"""Portable regression checks for the 2026-10-21 result-integrity review."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from tests import test_shared_runtime as runtime_support
from utils.dataset_adapters import DATASETS, DOMAINS
from utils.evaluation_adapters import commonsense_answer, final_math_answer, generate_math, generate_commonsense
from utils.runtime_utils import CheckpointStore, source_sha256

ROOT = Path(__file__).resolve().parents[1]


class Tokenizer:
    eos_token_id, pad_token_id, bos_token_id = 2, 7, 1
    padding_side = 'right'
    completion = ''

    def apply_chat_template(self, messages, **kwargs):
        self.last_messages = messages
        return [1, 4]

    def pad(self, rows, **kwargs):
        return {'input_ids': torch.tensor([row['input_ids'] for row in rows]),
                'attention_mask': torch.ones(len(rows), 2, dtype=torch.long)}

    def decode(self, tokens, **kwargs):
        return self.completion


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.value = torch.nn.Parameter(torch.ones(1))

    def generate(self, input_ids, **kwargs):
        self.generation_config = kwargs['generation_config']
        return torch.cat([input_ids, torch.tensor([[9, 2]] * len(input_ids))], dim=1)


class ReviewFixesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_training_format_math_answers_with_optional_colon(self):
        grader = SimpleNamespace(last_boxed_only_string=lambda _: None)
        for text in ('The answer is: 18', 'The answer is 18', 'The final answer is : 18', 'Final answer: 18'):
            self.assertEqual(final_math_answer(text, 'gsm8k'), '18')
        self.assertEqual(final_math_answer(r'The answer is: \frac{1}{2}', 'gsm8k'), '1/2')
        self.assertEqual(final_math_answer(r'The answer is: \frac{1}{2}', 'math', grader), r'\frac{1}{2}')
        self.assertIsNone(final_math_answer('The answer is: not a number', 'gsm8k'))

    def test_math_generation_uses_the_same_user_content_as_training(self):
        tokenizer = Tokenizer()
        tokenizer.completion = 'The answer is: 18'
        options = SimpleNamespace(eval_batch_size=1, max_new_tokens=8, eval_max_length=32)
        result = generate_math(Model(), tokenizer, [{'input_ids': 'original question', 'answer': '18'}], 'gsm8k', options)
        self.assertEqual(tokenizer.last_messages, [{'role': 'user', 'content': 'original question'}])
        self.assertEqual(result['metrics']['exact_match'], 1.0)
        self.assertEqual(tokenizer.padding_side, 'right')

    def test_commonsense_answer_matching_rejects_substrings_and_ambiguity(self):
        labels = ['solution1', 'solution2']
        self.assertEqual(commonsense_answer('Reasoning about solution2.\nThe correct answer is solution1.', labels), 'solution1')
        self.assertIsNone(commonsense_answer('solution10', labels))
        self.assertIsNone(commonsense_answer('The correct answer is solution1 or solution2', labels))
        self.assertIsNone(commonsense_answer('No answer', labels))
        self.assertEqual(commonsense_answer('the correct answer is FALSE', ['true', 'false']), 'false')

    def test_commonsense_generation_uses_training_prompt_and_small_answer_budget(self):
        tokenizer, model = Tokenizer(), Model()
        tokenizer.completion = 'the correct answer is solution2'
        options = SimpleNamespace(eval_batch_size=1, eval_max_length=128, commonsense_max_new_tokens=24)
        result = generate_commonsense(model, tokenizer, [{'input_ids': 'Choose the solution',
            'choice_labels': ['solution1', 'solution2'], 'answer': 'solution2'}], options)
        self.assertEqual(result['metrics']['generation_accuracy'], 1.0)
        self.assertEqual(tokenizer.last_messages, [{'role': 'user', 'content': 'Choose the solution'}])
        self.assertEqual(model.generation_config.max_new_tokens, 24)
        self.assertEqual(tokenizer.padding_side, 'right')

    def test_commonsense_tasks_keep_generation_and_native_files_distinct(self):
        tasks = DATASETS['commonsense170k'].tasks(Path('processed'))
        self.assertEqual(len(tasks), 16)
        for cid, domain in enumerate(DOMAINS):
            self.assertIn((domain, Path('processed/test') / f'{cid}.jsonl', cid), tasks)
            self.assertIn((f'native_{domain}', Path('processed/benchmarks') / f'{domain}.jsonl', cid), tasks)
        self.assertEqual(DATASETS['commonsense170k'].primary_metric, 'generation_accuracy')

    def test_mutable_base_is_not_rewritten_in_active_checkpoints(self):
        helper = runtime_support.SharedRuntimeTest()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaises(runtime_support.Interrupted):
                helper.simulate(root, 'fedexlora', interrupt_at=2)
            pointer = json.loads((root / 'exp/test/resume/latest.json').read_text())
            state = torch.load(root / 'exp/test/resume' / pointer['file'], weights_only=True)
            self.assertEqual(state['server']['base'], {})
            self.assertEqual(state['_tensor_references'][0]['file'], 'initial/state.pt')
            store = CheckpointStore(root / 'exp/test/resume', 'shared-test-fedexlora-commonsense170k')
            self.assertTrue(store.load()['server']['base'])
            initial = root / 'exp/test/initial/state.pt'
            original = initial.read_bytes()
            initial.write_bytes(original + b'corrupt')
            with self.assertRaisesRegex(ValueError, 'reference'):
                store.load()

    def test_round_checkpoint_references_the_committed_round_base(self):
        helper = runtime_support.SharedRuntimeTest()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            helper.simulate(root, 'fedmomentum', rounds=1)
            pointer = json.loads((root / 'exp/test/resume/latest.json').read_text())
            state = torch.load(root / 'exp/test/resume' / pointer['file'], weights_only=True)
            self.assertEqual(state['server']['base'], {})
            self.assertEqual(state['_tensor_references'][0]['file'], 'round_001/state.pt')

    def test_gpu_ordinal_is_not_part_of_the_training_identity(self):
        from utils.options import args_parser
        from utils.experiment import prepare_experiment
        config = ROOT / 'config/experiments/zsre.yaml'
        with patch('sys.argv', ['main.py', '--config', str(config)]):
            original = args_parser()
        prepared = {'adapter': DATASETS['zsre'], 'encoded': {0: [{}]}, 'tasks': [], 'data_files': {}, 'audit': {}}
        identities = []
        for device in (0, 7):
            args = copy.copy(original)
            args.device = device
            with patch('transformers.AutoTokenizer.from_pretrained', return_value=object()), \
                 patch('utils.experiment.prepare_data', return_value=copy.deepcopy(prepared)), \
                 patch('utils.experiment.validate_evaluator'), \
                 patch('utils.model_utils.local_model_fingerprint', return_value={'weights': 'same'}), \
                 patch('utils.experiment.importlib.metadata.version', return_value='test'):
                identities.append(prepare_experiment(args)['identity'])
        self.assertEqual(*identities)

    def test_config_is_required_even_when_an_old_root_config_exists(self):
        from utils.options import build_parser
        with patch('sys.argv', ['main.py']):
            with self.assertRaises(SystemExit) as result:
                build_parser()
        self.assertEqual(result.exception.code, 2)

    def test_legacy_registry_current_hashes_are_newline_independent(self):
        record = json.loads((ROOT / 'utils/legacy_shared_compat.json').read_text())
        for name, pair in record['sources'].items():
            self.assertEqual(pair['after'], source_sha256(ROOT / name), name)
        for name, digest in record['dependencies'].items():
            self.assertEqual(digest, source_sha256(ROOT / name), name)
        source = (ROOT / 'utils/train_utils.py').read_text(encoding='utf-8-sig').encode()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'source.py'
            for data in (source, source.replace(b'\n', b'\r\n')):
                path.write_bytes(data)
                self.assertEqual(source_sha256(path), hashlib.sha256(source).hexdigest())

    def test_legacy_identity_survives_lf_and_crlf_checkout_but_rejects_dependency_edits(self):
        import utils.commonsense_fedca as compat
        from utils.runtime_utils import fingerprint
        record = json.loads((ROOT / 'utils/legacy_shared_compat.json').read_text())
        old = {'training': {'alg': 'fedca'}, 'code': {name: pair['before'] for name, pair in record['sources'].items()}}
        current = {'training': old['training'], 'code': {name: pair['after'] for name, pair in record['sources'].items()}}
        current['code'].update(record['dependencies'])
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / 'output'
            output.mkdir()
            (output / 'run_manifest.json').write_text(json.dumps({'identity': fingerprint(old), 'identity_spec': old}))
            (root / 'utils').mkdir()
            (root / 'utils/legacy_shared_compat.json').write_text(json.dumps(record))
            for newline in (b'\n', b'\r\n'):
                for name in record['dependencies']:
                    (root / name).write_bytes((ROOT / name).read_text(encoding='utf-8-sig').encode().replace(b'\n', newline))
                with patch.object(compat, '__file__', str(root / 'utils/commonsense_fedca.py')):
                    self.assertEqual(compat.preserve_baseline_identity(output, current), old)
            with (root / 'utils/runtime_utils.py').open('ab') as stream:
                stream.write(b'\n# unrelated edit\n')
            with patch.object(compat, '__file__', str(root / 'utils/commonsense_fedca.py')):
                self.assertEqual(compat.preserve_baseline_identity(output, current), current)

    def test_fedca_dynamic_probe_uses_the_configured_padding_id(self):
        from datasets import Dataset
        from utils.ca_utils import _batches
        rows = Dataset.from_list([{'input_ids': [1, 2], 'labels': [-100, 2]},
                                  {'input_ids': [1, 3, 2], 'labels': [-100, 3, 2]}])
        ids, mask, labels = next(_batches(rows, [0, 1], 2, 'cpu', True, pad_token_id=7))
        self.assertEqual(ids[0, -1].item(), 7)
        self.assertEqual(mask[0, -1].item(), 0)
        self.assertEqual(labels[0, -1].item(), -100)

    def test_fedmomentum_small_core_aggregation_does_not_consume_rng(self):
        from alg.fedmomentum import Server
        from tests.test_zsre_extra_lora import FakeModel, A, B, sample_clients
        torch.manual_seed(19)
        server = object.__new__(Server)
        server.args = SimpleNamespace(lora_rank=2, lora_alpha=4)
        server.tau = 1.0
        server.sampled_clients = sample_clients(torch.randn(2, 4), torch.randn(4, 2),
                                                torch.randn(2, 4), torch.randn(4, 2))
        server.model = FakeModel(torch.zeros(2, 4), torch.zeros(4, 2))
        server.global_lora = {A: torch.zeros(2, 4), B: torch.zeros(4, 2)}
        before = torch.get_rng_state().clone()
        server.aggregate()
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
