"""FedCA keeps its protection/probe algorithm and complete continuation state."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import torch

from tests.test_commonsense import CommonsenseTest, Interrupted
from utils.ca_utils import not_true_kl
from utils.commonsense_training import atomic_json, fingerprint
from utils.commonsense_fedca import BASELINE_BEFORE, BASELINE_AFTER, preserve_baseline_identity


class FedCACommonsenseTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_full_fedca_continuation_keeps_reports_stats_and_probe_eta(self):
        helper = CommonsenseTest()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            extra = ('--ca_beta', '1', '--ca_kappa', '1')
            baseline = helper.simulate(root / 'full', 'fedca', extra_args=extra)
            for cut in (2, 5, 6, 34):
                directory = root / f'cut{cut}'
                with self.assertRaises(Interrupted):
                    helper.simulate(directory, 'fedca', interrupt_at=cut, extra_args=extra)
                recovered = helper.simulate(directory, 'fedca', extra_args=extra)
                for name in baseline['global']:
                    self.assertTrue(torch.equal(baseline['global'][name], recovered['global'][name]), (cut, name))
                for rnd in (1, 2):
                    before = json.loads((root / f'full/exp/test/round_{rnd:03d}/fedca_metrics.json').read_text())
                    after = json.loads((directory / f'exp/test/round_{rnd:03d}/fedca_metrics.json').read_text())
                    self.assertEqual(before['eta'], after['eta'])
                    self.assertEqual(before['clients'], after['clients'])
                    before['probe'].pop('seconds'); after['probe'].pop('seconds')
                    self.assertEqual(before['probe'], after['probe'])

    def test_fedca_off_is_exactly_framework_fedit(self):
        helper = CommonsenseTest()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fedit = helper.simulate(root / 'fedit', 'fedit')
            off = helper.simulate(root / 'off', 'fedca', extra_args=('--ca_lambda', '0', '--ca_eta_mode', 'fixed', '--ca_eta', '1'))
            for name in fedit['global']:
                self.assertTrue(torch.equal(fedit['global'][name], off['global'][name]), name)

    def test_compact_kl_matches_loss_and_gradient_without_prompt_distributions(self):
        torch.manual_seed(4)
        student = torch.randn(2, 8, 17, requires_grad=True)
        teacher = torch.randn_like(student)
        labels = torch.full((2, 8), -100)
        labels[0, 4:7] = torch.tensor([1, 2, 3]); labels[1, 3:5] = torch.tensor([7, 8])
        a = not_true_kl(student, teacher, labels, tau=1.7)
        ga = torch.autograd.grad(a, student, retain_graph=True)[0]
        b = not_true_kl(student, teacher, labels, tau=1.7, compact=True)
        gb = torch.autograd.grad(b, student)[0]
        torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(ga, gb, atol=1e-6, rtol=1e-6)

    def test_only_the_known_fedca_extension_reuses_existing_baseline_identity(self):
        old = {'training': {'alg': 'fedit'}, 'model': 'same', 'data': 'same',
               'code': {**BASELINE_BEFORE, 'utils/train_utils.py': 'unchanged'}}
        current = copy.deepcopy(old)
        current['code'].update(BASELINE_AFTER)
        with tempfile.TemporaryDirectory() as tmp:
            atomic_json(Path(tmp) / 'run_manifest.json', {'identity_spec': old, 'identity': fingerprint(old)})
            self.assertEqual(preserve_baseline_identity(Path(tmp), current), old)
            changed_model = copy.deepcopy(current); changed_model['model'] = 'different'
            self.assertEqual(preserve_baseline_identity(Path(tmp), changed_model), changed_model)
            changed_code = copy.deepcopy(current); changed_code['code']['utils/train_utils.py'] = 'different'
            self.assertEqual(preserve_baseline_identity(Path(tmp), changed_code), changed_code)


if __name__ == '__main__':
    unittest.main()
