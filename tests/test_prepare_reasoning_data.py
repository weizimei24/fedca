"""Offline checks for deterministic, non-destructive reasoning preparation."""

import json
import tempfile
import unittest
from pathlib import Path

from tools.prepare_reasoning_data import boxed_answer, choice_data, fingerprint, math_fingerprint, iter_array, sample_indices


class ReasoningPreparationTest(unittest.TestCase):
    def test_stream_array_across_unicode_and_chunk_boundaries(self):
        rows = [{"query": "中文 braces { and escaped \" text " * 20, "n": i} for i in range(5)]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "rows.json"
            path.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
            self.assertEqual(list(iter_array(path, chunk_size=7)), rows)
            path.write_text("[]\n", encoding="utf-8")
            self.assertEqual(list(iter_array(path, chunk_size=1)), [])

    def test_stream_rejects_truncation_trailing_comma_and_extra_data(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "rows.json"
            for bad in ('[{"query": "cut off', '[{"x":1},]', '[{"x":1}] garbage', '[{"x":1} {"x":2}]'):
                path.write_text(bad, encoding="utf-8")
                with self.subTest(bad=bad), self.assertRaises(ValueError):
                    list(iter_array(path, chunk_size=3))

    def test_variable_choice_count_and_original_label_are_preserved(self):
        row = {"instruction": "Question\nAnswer1: a Answer2: b Answer3: c Answer4: d Answer5: e\nAnswer format: answer1/answer2/answer3/answer4/answer5", "answer": "answer5"}
        question, options, labels, target = choice_data(row)
        self.assertEqual(question, "Question")
        self.assertEqual(options, ["a", "b", "c", "d", "e"])
        self.assertEqual(labels[target], "answer5")
        with self.assertRaises(ValueError):
            choice_data(dict(row, answer="answer6"))

    def test_boolean_choices_are_not_forced_into_abcd(self):
        row = {"instruction": "Please answer: is this true?\nAnswer format: true/false", "answer": "false"}
        _, options, labels, target = choice_data(row)
        self.assertEqual(options[target], "false")
        self.assertEqual(labels, ["true", "false"])

    def test_balanced_math_gold_extraction(self):
        self.assertEqual(boxed_answer(r"first \boxed{1}, then \boxed{\frac{2}{\sqrt{3}}}"), r"\frac{2}{\sqrt{3}}")
        self.assertEqual(boxed_answer(r"\fbox{\{1,2\}}"), r"\{1,2\}")
        self.assertEqual(boxed_answer(r"our answer is $\boxed 2$."), "2")
        self.assertIsNone(boxed_answer(r"\boxed{unfinished"))

    def test_sampling_is_reproducible_capped_and_without_replacement(self):
        first = sample_indices(10000, 4000, 42, 0)
        self.assertEqual(first, sample_indices(10000, 4000, 42, 0))
        self.assertNotEqual(first, sample_indices(10000, 4000, 42, 1))
        self.assertEqual(len(set(first)), 4000)
        self.assertEqual(sample_indices(1119, 4000, 42, 0), list(range(1119)))

    def test_fingerprints_normalize_only_for_comparison(self):
        self.assertEqual(fingerprint("A\n B", "Correct"), fingerprint("a b", "correct"))
        self.assertNotEqual(fingerprint("a", "b c"), fingerprint("a b", "c"))
        self.assertNotEqual(math_fingerprint(r"\Pi + X"), math_fingerprint(r"\pi + x"))
        self.assertEqual(math_fingerprint("X\n + 1"), math_fingerprint("X + 1"))


if __name__ == "__main__":
    unittest.main()
