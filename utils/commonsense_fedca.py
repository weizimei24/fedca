"""FedCA-specific state and safe compatibility with the three existing baselines."""
from __future__ import annotations
import math
from pathlib import Path

from utils.commonsense_framework import CommonsenseRun, read_json
from utils.commonsense_training import atomic_json, cpu_tree, fingerprint
from utils.commonsense_eval import json_safe

# Filled with the exact before/after digests of this FedCA-only extension. It
# accepts no arbitrary code changes, model/data changes or parameter changes.
BASELINE_BEFORE = {
    'main.py': 'a0bca0ea6fda1cdf727748fcf2583cbce9332e24cfa5d13cd21030fc2c67ca5a',
    'utils/options.py': 'a813d682416c1ba3b301567cf1ebaec7e6969d7cacafbf2603391b147fd655db',
    'utils/commonsense_framework.py': 'a1c29f6a0c3255a2514893eff364872ba819ffde1e2c8ab19ef3efb8fea2ec10',
}
BASELINE_AFTER = {'main.py': '8a08cd63c742fff2706a49bfba3637e36cc569ce2aff058040dbba3be0ca3850', 'utils/options.py': '73a8fc3dec045133c58388b3f41b8e21700fab0672a934b077b5b8d9110e60d4', 'utils/commonsense_framework.py': 'b813e7ec3e46cac88aeb642351e994944ea919b2312ba8f36f63849b5255939b'}


def preserve_baseline_identity(output, current):
    manifest = Path(output) / 'run_manifest.json'
    if not manifest.exists():
        return current
    saved = read_json(manifest)
    old = saved.get('identity_spec', {})
    if fingerprint(old) != saved.get('identity') or set(old) != set(current):
        return current
    if any(old[key] != current[key] for key in current if key != 'code'):
        return current
    # The legacy route retains the same optimizer/encoding/commit semantics.
    # Accept only exact reviewed pre/post source pairs; never ignore hashes.
    root = Path(__file__).resolve().parents[1]
    compatibility = root / 'utils/legacy_shared_compat.json'
    reviewed = {}
    dependencies = {}
    if compatibility.exists():
        from utils.runtime_utils import source_sha256
        record = read_json(compatibility)
        if all((root / path).is_file() and source_sha256(root / path) == digest
               for path, digest in record['dependencies'].items()):
            reviewed = record['sources']
            dependencies = record['dependencies']
    extra = set(current['code']) - set(old['code'])
    if set(old['code']) - set(current['code']) or any(
            dependencies.get(name) != current['code'][name] for name in extra):
        return current
    for name, digest in current['code'].items():
        if name in extra:
            continue  # Exact new dependency digest verified above.
        previous = old['code'][name]
        if previous == digest:
            continue
        if (BASELINE_BEFORE.get(name), BASELINE_AFTER.get(name)) == (previous, digest):
            continue
        pair = reviewed.get(name, {})
        allowed_before = {pair.get('before'), *pair.get('compatible_before', [])}
        if pair.get('before') == BASELINE_AFTER.get(name):
            allowed_before.add(BASELINE_BEFORE.get(name))
        if digest != pair.get('after') or previous not in allowed_before:
            return current
    print('Existing legacy checkpoint matches reviewed compatibility fingerprints; preserving its identity.', flush=True)
    return old


def validate_fedca(args):
    from utils.ca_utils import parse_grid
    if not math.isfinite(args.ca_lambda) or args.ca_lambda < 0 or not math.isfinite(args.ca_tau) or args.ca_tau <= 0:
        raise ValueError('FedCA requires finite ca_lambda>=0 and ca_tau>0')
    if args.ca_probe_size <= 0 or args.ca_probe_bs <= 0 or args.ca_kappa < 0 or math.isnan(float(args.ca_kappa)):
        raise ValueError('FedCA requires positive probe size/batch size and nonnegative kappa')
    if not math.isfinite(args.ca_beta) or not math.isfinite(args.ca_eta) or args.ca_eta <= 0:
        raise ValueError('FedCA requires finite beta and a positive eta')
    parse_grid(args.ca_eta_grid)


class CommonsenseFedCARun(CommonsenseRun):
    def checkpoint_active(self, client, active):
        active['fedca_statistics'] = client.trainer.checkpoint_statistics()
        super().checkpoint_active(client, active)

    def commit_client(self, client):
        # Report and statistics commit atomically WITH the completed update.
        # They are needed for beta weights, excess budgets and server probes.
        self.state['pending'][str(client.id)] = {'adapter': cpu_tree(client.lora),
            'metrics': client.trainer.last_train_metrics, 'training_time': float(client.training_time),
            'report': cpu_tree(client.report), 'last_excess': client.trainer.last_excess}
        self.state['active'] = None
        self.store.save(self.state)

    def restore_completed_client(self, client):
        if not super().restore_completed_client(client):
            return False
        saved = self.state['pending'][str(client.id)]
        client.report = saved['report']
        client.trainer.last_excess = saved['last_excess']
        return True

    def log_algorithm_round(self, record):
        payload = json_safe(record)
        if record.get('probe') and math.isinf(record['probe']['budget']):
            payload['probe']['budget_unbounded'] = True
        atomic_json(self.output / f"round_{self.server.round+1:03d}" / 'fedca_metrics.json', payload)
