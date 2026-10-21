"""Base evaluation reuses production scoring and never needs a LoRA adapter."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from tools.eval_base import cached_result, check_encoding, write_predictions
from utils.runtime_utils import atomic_json, fingerprint, sha256


class BaseEvaluationTest(unittest.TestCase):
    def test_real_base_math_answer_presentation_and_units(self):
        from utils.evaluation_adapters import final_math_answer
        examples = {'Final Answer: **3 bolts**.': '3',
            'Final Answer: **$694**': '694',
            r'Final Answer: \boxed{60\%}': '60',
            r'Final Answer: \boxed{\$10}': '10',
            'Final Answer:\nEach train covers **230 miles** in total.': '230',
            'Final Answer:\nTerry spends **$75.00** on yogurt over 30 days.': '75.00',
            'Final Answer: 1/2 liters': '1/2',
            'The answer is 2.\nThe final answer is: 3': '3'}
        for text, expected in examples.items():
            with self.subTest(text=text):
                self.assertEqual(final_math_answer(text, 'gsm8k'), expected)
        for text in ('Final answer: 1 or 2', 'Final answer: 1 + 2',
                     'Final answer: 1/0 liters', r'Final answer: \sqrt{4}',
                     'Final answer: **1** or **2**', 'Final answer: **1** or 2',
                     'Final answer: **1** + 2', 'Final answer: not 3',
                     r'Final answer: \boxed{12'):
            with self.subTest(text=text):
                self.assertIsNone(final_math_answer(text, 'gsm8k'))

    def test_encoding_audit_requires_masked_prompt_and_supervised_eos(self):
        tokenizer = SimpleNamespace(eos_token_id=2)
        prepared = {'encoded': {0: [{'input_ids':[1,4,5,2], 'labels':[-100,-100,5,2]}]}}
        self.assertEqual(check_encoding(prepared, tokenizer)['training_rows_checked'], 1)
        for labels in ([-100,-100,5,-100], [1,4,5,2], [-100,4,-100,2]):
            prepared['encoded'][0][0]['labels'] = labels
            with self.assertRaises(ValueError):
                check_encoding(prepared, tokenizer)

    def test_prediction_files_contain_prompt_output_and_extracted_answer(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'predictions/task.jsonl'
            rows = [{'input_ids':'question', 'choice_labels':['answer1','answer2']}]
            predictions = [{'prediction':'the correct answer is answer1', 'extracted_answer':'answer1','target':'answer1'}]
            write_predictions(path, rows, predictions)
            record = json.loads(path.read_text(encoding='utf-8'))
            self.assertEqual(record['prompt'], 'question')
            self.assertEqual(record['choice_labels'], ['answer1','answer2'])
            self.assertEqual(record['extracted_answer'], 'answer1')
            with self.assertRaises(ValueError):
                write_predictions(path, rows, [])

    def test_cache_rejects_changed_predictions_and_result_body(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            prediction = output/'predictions/task.jsonl'
            write_predictions(prediction, [{'input_ids':'q'}], [{'prediction':'a','target':'a'}])
            result = {'cache_key':'key','metrics':{'exact_match':1},
                'predictions_file':'predictions/task.jsonl','predictions_sha256':sha256(prediction)}
            result['cache_sha256'] = fingerprint(result)
            path = output/'task.json'
            atomic_json(path,result)
            self.assertEqual(cached_result(path,'key',output),result)
            with self.assertRaises(ValueError):
                cached_result(path,'other-key',output)
            prediction.write_text('{}\n')
            with self.assertRaisesRegex(ValueError,'prediction'):
                cached_result(path,'key',output)
            result['metrics']['exact_match']=0
            atomic_json(path,result)
            with self.assertRaisesRegex(ValueError,'cache'):
                cached_result(path,'key',output)

    def test_generation_cannot_inherit_sampling_from_model_defaults(self):
        import torch
        from transformers import GenerationConfig
        from transformers.generation.utils import GenerationMixin
        from tests.test_review_fixes import Model, Tokenizer
        from utils.evaluation_adapters import generate_commonsense, generate_math

        class SamplingDefaultModel(Model):
            def __init__(self):
                super().__init__()
                self.generation_config = GenerationConfig(do_sample=True, temperature=.6, top_p=.95,
                                                          transformers_version='4.51.0')

            def generate(self, input_ids, generation_config, use_model_defaults=None, **kwargs):
                self.effective, _ = GenerationMixin._prepare_generation_config(
                    self, generation_config, use_model_defaults=use_model_defaults)
                return torch.cat([input_ids, torch.tensor([[9,2]]*len(input_ids))],dim=1)

        args = SimpleNamespace(eval_batch_size=1, max_new_tokens=8, eval_max_length=32,
                               commonsense_max_new_tokens=8)
        tokenizer = Tokenizer()
        model = SamplingDefaultModel()
        tokenizer.completion = 'answer1'
        generate_commonsense(model, tokenizer, [{'input_ids':'q','answer':'answer1',
            'choice_labels':['answer1','answer2']}], args)
        self.assertFalse(model.effective.do_sample)
        tokenizer.completion = 'The answer is: 18'
        generate_math(model, tokenizer, [{'input_ids':'q','answer':'18'}], 'gsm8k', args)
        self.assertFalse(model.effective.do_sample)
