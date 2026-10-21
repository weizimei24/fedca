"""One lifecycle for every registered dataset and federated method.

The existing FedSim/Server/Client/Trainer still own training. This module owns
durable commits and evaluation scheduling, not an alternative optimizer loop.
"""
import importlib.metadata
import json
import math
from pathlib import Path

import torch

from utils.dataset_adapters import prepare_data
from utils.evaluation_adapters import validate_evaluator, evaluate_task
from utils.method_state import method_spec, validate_method, active_extras, client_extras, restore_client_extras, restore_tensors
from utils.runtime_utils import (CheckpointStore, atomic_json, atomic_torch, cpu_tree,
    fingerprint, sha256, source_sha256, capture_rng, restore_rng, evaluation_rounds)

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = 'fedsim_shared_v1'


def use_shared_runtime(args):
    runtime = getattr(args, 'runtime', 'legacy')
    if runtime not in ('legacy', 'shared'):
        raise ValueError(f'Invalid runtime: {runtime!r}; expected shared or legacy')
    if runtime == 'legacy' and getattr(args, 'task', ''):
        raise ValueError('Dataset task adapters require runtime: shared')
    return runtime == 'shared'


def normalized(value):
    if isinstance(value, float) and not math.isfinite(value):
        if math.isnan(value):
            raise ValueError('NaN configuration value')
        return 'inf' if value > 0 else '-inf'
    return value


def prepare_experiment(args):
    if hasattr(args, '_experiment_prepared'):
        return args._experiment_prepared
    if args.mode != 'prototype' or args.task_type != 'CAUSAL_LM' or float(args.sr) != 1.0:
        raise ValueError('Shared runtime currently requires prototype, CAUSAL_LM and sr=1.0')
    if args.global_test or args.cross_domain_qa:
        raise ValueError('Shared runtime obtains evaluation tasks from the dataset adapter')
    for key in ('rnd', 'test_gap', 'bs', 'grad_accum', 'epoch', 'checkpoint_steps',
        'max_length', 'eval_max_length', 'eval_batch_size', 'max_new_tokens', 'commonsense_max_new_tokens', 'lora_rank', 'lora_alpha'):
        value = getattr(args, key)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f'{key} must be a positive integer')
    if isinstance(args.step, bool) or not isinstance(args.step, int) or args.step < 0:
        raise ValueError('step must be a nonnegative integer')
    if not math.isfinite(args.lr) or args.lr <= 0 or not 0 <= args.lora_dropout < 1:
        raise ValueError('Invalid training budget, lr or dropout')
    if args.eval_limit is not None and (isinstance(args.eval_limit, bool)
            or not isinstance(args.eval_limit, int) or args.eval_limit <= 0):
        raise ValueError('eval_limit must be a positive integer')
    if args.resume not in ('auto', 'never'):
        raise ValueError('resume must be auto or never')
    if args.commonsense_eval not in ('generation', 'native'):
        raise ValueError('commonsense_eval must be generation or native')
    evaluation_rounds(args)
    spec = method_spec(args.alg)
    validate_method(args)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    prepared = prepare_data(args, tokenizer)
    validate_evaluator(prepared['adapter'], prepared['tasks'])
    settings = {k: getattr(args, k) for k in ('alg', 'cn', 'sr', 'seed', 'deterministic', 'task_type',
        'bs', 'grad_accum', 'epoch', 'step', 'lr', 'lora_rank', 'lora_alpha', 'lora_dropout',
        'max_length', 'gradient_checkpointing')}
    settings['method_parameters'] = {k: normalized(getattr(args, k)) for k in spec.parameters if hasattr(args, k)}
    settings.update(lr_decay=0.99, dtype='float16', encoding=prepared['adapter'].encoding)
    from utils.model_utils import local_model_fingerprint
    model_files = local_model_fingerprint(args.model)
    sources = ['main.py', 'alg/ftbase.py', 'alg/base.py', f'alg/{args.alg}.py',
        'utils/experiment.py', 'utils/runtime_utils.py', 'utils/dataset_adapters.py',
        'utils/method_state.py', 'utils/train_utils.py', 'utils/data_utils.py',
        'utils/model_utils.py', 'utils/seed_utils.py', 'utils/options.py']
    if args.alg in ('fedca', 'fedsvd', 'fedfh', 'fedmomentum'):
        sources += {'fedca': ['utils/ca_utils.py'], 'fedsvd': ['alg/flexlora.py'],
                    'fedfh': ['utils/fh_utils.py', 'alg/flexlora.py'],
                    'fedmomentum': ['alg/flexlora.py']}[args.alg]
    if prepared['adapter'].name == 'zsre':
        sources += ['utils/zsre_utils.py']
    identity_spec = {'protocol': PROTOCOL, 'training': settings,
        'task': prepared['adapter'].name, 'data_files': prepared['data_files'],
        'model_files': model_files,
        'packages': {name: importlib.metadata.version(name) for name in
                     ('torch', 'transformers', 'tokenizers', 'peft', 'numpy', 'datasets', 'accelerate')},
        'code': {name: source_sha256(ROOT / name) for name in sorted(set(sources))}}
    sizes = ([sum(len(v) for v in prepared['encoded'].values())] if args.alg == 'centralized'
             else [len(v) for v in prepared['encoded'].values()])
    full_updates = [math.ceil(math.ceil(size / args.bs) / args.grad_accum) * args.epoch for size in sizes]
    prepared.update(identity=fingerprint(identity_spec), identity_spec=identity_spec,
        report={'protocol': PROTOCOL, 'task': prepared['adapter'].name, 'method': args.alg,
                'clients': prepared['audit'], 'rounds': args.rnd, 'eval_at': evaluation_rounds(args),
                'full_epoch_optimizer_updates': full_updates,
                'diagnostic_budget': bool(args.step and any(args.step < n for n in full_updates)),
                'train_rows': sum(len(v) for v in prepared['encoded'].values())})
    args._experiment_prepared = prepared
    return prepared


def same_tensors(left, right):
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and torch.equal(left, right)
    if isinstance(left, dict):
        return isinstance(right, dict) and set(left) == set(right) and all(same_tensors(left[k], right[k]) for k in left)
    if isinstance(left, (list, tuple)):
        return isinstance(right, type(left)) and len(left) == len(right) and all(same_tensors(a, b) for a, b in zip(left, right))
    return left == right


class ExperimentRun:
    def __init__(self, args, server):
        self.args, self.server = args, server
        self.prepared = prepare_experiment(args)
        self.identity = self.prepared['identity']
        self.output = Path(args.suffix).resolve()
        self.store = CheckpointStore(self.output / 'resume', self.identity)
        manifest = self.output / 'run_manifest.json'
        if manifest.exists() and json.loads(manifest.read_text(encoding='utf-8'))['identity'] != self.identity:
            raise ValueError('Experiment identity differs; use the matching code/config or a new --suffix')
        self.state = self.store.load()
        if args.resume == 'never' and (manifest.exists() or self.state is not None):
            raise ValueError('resume=never requires a new output directory')
        if self.state is None and (manifest.exists() or any(self.output.glob('round_*/complete.json'))
                                   or (self.output / 'adapter').exists()):
            raise ValueError('Existing experiment has no shared continuation state; use a new --suffix, '
                             'or its original legacy runtime. Refusing to retrain into old output.')
        if args.eval_only and self.state is None:
            raise ValueError('eval_only requires an existing shared checkpoint')
        if self.state is None:
            self.state = {'format': PROTOCOL, 'completed_rounds': 0, 'pending': {}, 'active': None,
                          'server': server.export_state(), 'rng': capture_rng(), 'wall_clock_time': 0.0,
                          'requested_rounds': args.rnd}
            self._set_base_reference(initial=True)
            self._save_checkpoint()
        else:
            self._set_base_reference()
        if self.state['format'] != PROTOCOL or self.completed_rounds > args.rnd:
            raise ValueError('Incompatible checkpoint format or rnd below completed rounds')
        if not manifest.exists():
            atomic_json(manifest, {'identity': self.identity, 'identity_spec': self.prepared['identity_spec'],
                                  'readiness': self.prepared['report'], 'initial_rounds': args.rnd})
        original_rounds = json.loads(manifest.read_text(encoding='utf-8'))['initial_rounds']
        self.state.setdefault('requested_rounds', max(original_rounds, self.completed_rounds))
        if not args.eval_only:
            self.state['requested_rounds'] = args.rnd
        server.restore_state(self.state['server'])
        restore_rng(self.state['rng'])
        server.wall_clock_time = self.state['wall_clock_time']
        self.write_progress()
        print(f'Shared continuation: {self.completed_rounds} completed rounds, '
              f'{len(self.state["pending"])} completed clients in next round', flush=True)

    def _set_base_reference(self, initial=False):
        self._base_reference = None
        if not self.state['server']['base']:
            return
        if self.completed_rounds:
            folder = self.output / f'round_{self.completed_rounds:03d}'
            marker = json.loads((folder / 'complete.json').read_text(encoding='utf-8'))
            path, digest = folder / marker['file'], marker['sha256']
            if marker['identity'] != self.identity or sha256(path) != digest:
                raise ValueError('Committed base snapshot identity/checksum differs')
        else:
            path = self.output / 'initial/state.pt'
            payload = {'identity': self.identity, 'round': 0, 'server': self.state['server']}
            if path.exists():
                previous = torch.load(path, map_location='cpu', weights_only=True)
                if not same_tensors(previous, payload):
                    raise ValueError('Initial base snapshot differs')
            elif initial:
                atomic_torch(path, payload)
            else:
                # Migration of a complete pre-optimization checkpoint: its
                # verified embedded base provides the initial immutable blob.
                atomic_torch(path, payload)
            digest = sha256(path)
        self._base_reference = {'file': path.relative_to(self.output).as_posix(), 'sha256': digest,
            'source_keys': ['server', 'base'], 'destination_keys': ['server', 'base']}

    def _save_checkpoint(self):
        payload = self.state
        if self._base_reference is not None:
            payload = {**self.state, 'server': {**self.state['server'], 'base': {}},
                       '_tensor_references': [self._base_reference]}
        self.store.save(payload)

    @property
    def completed_rounds(self):
        return self.state['completed_rounds']

    def write_progress(self):
        atomic_json(self.output / 'progress.json', {'completed_rounds': self.completed_rounds,
            'requested_rounds': self.state['requested_rounds'], 'identity': self.identity, 'test_gap': self.args.test_gap})

    def active_for(self, client):
        active = self.state['active']
        if active and (active['client_id'] != client.id or active['round_index'] != self.server.round):
            raise ValueError('Active checkpoint client/round differs')
        return active

    def checkpoint_active(self, client, active):
        active.update(active_extras(client))
        self.state['active'] = active
        self._save_checkpoint()

    def commit_client(self, client):
        self.state['pending'][str(client.id)] = {'adapter': cpu_tree(client.lora),
            'metrics': cpu_tree(client.trainer.last_train_metrics), 'training_time': float(client.training_time),
            'extras': client_extras(client)}
        if self.args.alg == 'localonly':
            self.state['server']['local'][client.id] = cpu_tree(client.lora)
        self.state['active'] = None
        self.state['rng'] = capture_rng()
        self._save_checkpoint()

    def restore_completed_client(self, client):
        saved = self.state['pending'].get(str(client.id))
        if saved is None:
            return False
        params = dict(self.server.model.named_parameters())
        client.lora = {k: v.to(params[k].device).clone() for k, v in saved['adapter'].items()}
        client.last_train_loss = saved['metrics']['mean_loss']
        client.trainer.last_train_metrics = saved['metrics']
        client.training_time = saved['training_time']
        restore_client_extras(client, saved['extras'])
        if self.args.alg == 'localonly':
            self.server.client_loras[client.id] = {k: v.clone() for k, v in client.lora.items()}
        print(f'Client {client.id}: reusing committed update', flush=True)
        return True

    def before_aggregate(self):
        # Includes randomized SVD/probing. Replay after a pre-round-commit crash
        # must start with exactly the RNG state following the final client.
        if self.state['pending']:
            restore_rng(self.state['rng'])

    def commit_round(self):
        rnd = self.server.round + 1
        if set(self.state['pending']) != {str(c.id) for c in self.server.clients}:
            raise RuntimeError('Cannot commit a round with missing clients')
        saved = self.server.export_state()
        directory = self.output / f'round_{rnd:03d}'
        path, marker = directory / 'state.pt', directory / 'complete.json'
        payload = {'identity': self.identity, 'round': rnd, 'method': self.args.alg, 'server': saved}
        if marker.exists():
            meta = json.loads(marker.read_text(encoding='utf-8'))
            if meta['identity'] != self.identity or meta['sha256'] != sha256(path):
                raise ValueError('Existing round snapshot identity/checksum differs')
            previous = torch.load(path, map_location='cpu', weights_only=True)
            if not same_tensors(previous, payload):
                raise ValueError('Reconstructed aggregation differs from committed snapshot')
        else:
            atomic_torch(path, payload)
            atomic_json(marker, {'identity': self.identity, 'round': rnd, 'file': path.name, 'sha256': sha256(path)})
        atomic_json(directory / 'training_metrics.json', {'round': rnd, 'method': self.args.alg,
            'clients': [self.state['pending'][str(c.id)]['metrics'] for c in self.server.clients]})
        self.state.update(completed_rounds=rnd, pending={}, active=None, server=saved,
                          rng=capture_rng(), wall_clock_time=self.server.wall_clock_time)
        self._set_base_reference()
        self._save_checkpoint()
        self.write_progress()

    def log_algorithm_round(self, record):
        from utils.commonsense_eval import json_safe
        atomic_json(self.output / f'round_{self.server.round+1:03d}' / 'algorithm_metrics.json', json_safe(record))

    def evaluate(self, round_number):
        directory = self.output / f'round_{round_number:03d}'
        meta = json.loads((directory / 'complete.json').read_text(encoding='utf-8'))
        path = directory / meta['file']
        if meta['identity'] != self.identity or sha256(path) != meta['sha256']:
            raise ValueError('Evaluation snapshot checksum/identity differs')
        snapshot = torch.load(path, map_location='cpu', weights_only=True)
        if snapshot['identity'] != self.identity or snapshot['round'] != round_number:
            raise ValueError('Evaluation snapshot metadata differs')
        adapter = self.prepared['adapter']
        settings = {'identity': self.identity, 'round': round_number, 'snapshot_sha256': meta['sha256'],
            'protocol': adapter.evaluator, 'eval_limit': self.args.eval_limit,
            'batch_size': self.args.eval_batch_size, 'max_length': self.args.eval_max_length,
            'max_new_tokens': self.args.max_new_tokens, 'seed': self.args.seed,
            'primary_metric': adapter.primary_metric,
            'commonsense_max_new_tokens': self.args.commonsense_max_new_tokens,
            'implementation': {name: source_sha256(ROOT / name) for name in
                ('utils/evaluation_adapters.py', 'utils/commonsense_eval.py', 'eval_zsre.py', 'utils/zsre_utils.py')}}
        settings['evaluator_packages'] = {name: importlib.metadata.version(name) for name in adapter.evaluator_packages}
        evaluation_id = fingerprint(settings)
        output = self.output / 'evaluation' / f'round_{round_number:03d}' / evaluation_id[:16]
        before, rng, training = self.server.export_state(), capture_rng(), self.server.model.training
        results = {}
        try:
            for name, data_path, cid in self.prepared['tasks']:
                key = fingerprint({**settings, 'task': name, 'data_sha256': sha256(data_path)})
                dest = output / f'{name}.json'
                if dest.exists():
                    result = json.loads(dest.read_text(encoding='utf-8'))
                    checksum = fingerprint({k: v for k, v in result.items() if k != 'cache_sha256'})
                    if result.get('cache_key') != key or result.get('cache_sha256') != checksum:
                        raise ValueError(f'Evaluation cache differs: {dest}')
                else:
                    self.server.restore_state(snapshot['server'])
                    if self.args.alg == 'localonly':
                        restore_tensors(self.server.model, snapshot['server']['local'][cid])
                    # Evaluation must not consume training or aggregation RNG.
                    from utils.seed_utils import seed_rngs
                    seed_rngs(self.args.seed)
                    result = {'cache_key': key, 'settings': settings, 'task': name,
                              'client_id': cid if self.args.alg == 'localonly' else None,
                              **evaluate_task(adapter, self.server.model, self.server.clients[0].tokenizer,
                                              name, data_path, self.args)}
                    result['cache_sha256'] = fingerprint(result)
                    atomic_json(dest, result)
                results[name] = result
            metric = adapter.primary_metric
            primary_results = [result for result in results.values() if metric in result['metrics']]
            if not primary_results:
                raise ValueError(f'No evaluation task provides primary metric {metric}')
            native_results = [result for name, result in results.items() if name.startswith('native_')]
            generation_results = [result for result in results.values() if 'generation_accuracy' in result['metrics']]
            supplementary = {}
            if native_results:
                supplementary['native_mc_raw_accuracy'] = sum(r['metrics']['acc,none'] for r in native_results) / len(native_results)
            if generation_results and metric != 'generation_accuracy':
                supplementary['generation_accuracy'] = sum(r['metrics']['generation_accuracy'] for r in generation_results) / len(generation_results)
            summary = {'settings': settings, 'evaluation_id': evaluation_id,
                'diagnostic_only': self.args.eval_limit is not None or self.prepared['report'].get('diagnostic_budget', False),
                'primary_metric_units': 'fraction',
                'primary_metric': metric, 'macro_average': sum(r['metrics'][metric] for r in primary_results) / len(primary_results),
                'supplementary_metrics': supplementary,
                'per_task': {name: result['metrics'] for name, result in results.items()}}
            atomic_json(output / 'summary.json', summary)
            pointer = 'latest_full.json' if self.args.eval_limit is None else 'latest_diagnostic.json'
            atomic_json(output.parent / pointer, {'summary': str((output / 'summary.json').relative_to(self.output)), **summary})
            return {f'macro_{metric}': summary['macro_average'],
                    **{f'macro_{key}': value for key, value in supplementary.items()}}
        finally:
            self.server.restore_state(before)
            restore_rng(rng)
            self.server.model.train(training)

    def recover_evaluations(self, eval_only=False):
        rounds = evaluation_rounds(self.args)
        if eval_only and any(rnd > self.completed_rounds for rnd in rounds):
            raise ValueError('Requested evaluation round is not completed')
        for rnd in rounds:
            if rnd <= self.completed_rounds:
                self.evaluate(rnd)

    def mark_complete(self):
        atomic_json(self.output / 'completed.json', {'completed_rounds': self.completed_rounds,
            'requested_rounds': self.state['requested_rounds'],
            'evaluation_rounds': evaluation_rounds(self.args), 'identity': self.identity,
            'diagnostic_only': self.args.eval_limit is not None or self.prepared['report'].get('diagnostic_budget', False),
            'protocol': PROTOCOL})
