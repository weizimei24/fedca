"""Commonsense data and checkpoint lifecycle for the existing FedSim framework.

Algorithms still own sample/local_run/aggregate, and Trainer owns optimization.
This module supplies validated data, durable state and harness evaluation only.
"""
from __future__ import annotations

import importlib.metadata
import json
import math
from pathlib import Path

import torch

from utils.commonsense_training import (DOMAINS, TRAIN_PROTOCOL, CheckpointStore, adapter_state,
    atomic_json, atomic_torch, cpu_tree, encode_example, fingerprint, load_adapter, sha256)
from utils.commonsense_eval import build_task, ensure_runtime, evaluate_round

ROOT = Path(__file__).resolve().parents[1]
FRAMEWORK_PROTOCOL = 'fedsim_commonsense_v1'


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def evaluation_rounds(args):
    if args.eval_rounds:
        rounds = sorted(set(int(x) for x in args.eval_rounds.split(',')))
        if not rounds or min(rounds) < 1 or max(rounds) > args.rnd:
            raise ValueError('eval_rounds must be within 1..rnd')
        return rounds
    return sorted(set(range(args.test_gap, args.rnd+1, args.test_gap)) | {args.rnd})


def validate_args(args):
    if args.alg not in ('fedit', 'centralized', 'localonly', 'fedca') or args.mode != 'prototype':
        raise ValueError('Commonsense supports prototype fedit/centralized/localonly/fedca')
    expected = 1 if args.alg == 'centralized' else 8
    if args.cn != expected or float(args.sr) != 1.0:
        raise ValueError(f'{args.alg} requires cn={expected}, sr=1.0')
    if args.task_type != 'CAUSAL_LM' or args.cross_domain_qa or args.global_test:
        raise ValueError('Commonsense requires CAUSAL_LM and uses its own canonical server evaluation')
    for name in ('rnd', 'test_gap', 'bs', 'grad_accum', 'epoch', 'checkpoint_steps', 'max_length',
                 'eval_max_length', 'eval_batch_size', 'lora_rank', 'lora_alpha'):
        if getattr(args, name) <= 0:
            raise ValueError(f'{name} must be positive')
    if args.step < 0 or not math.isfinite(args.lr) or args.lr <= 0 or not 0 <= args.lora_dropout < 1:
        raise ValueError('Invalid step/lr/LoRA dropout; step=0 means the complete configured epochs')
    if args.eval_limit is not None and args.eval_limit <= 0:
        raise ValueError('eval_limit must be positive')
    if args.eval_rounds and not args.eval_only:
        raise ValueError('eval_rounds requires eval_only')
    evaluation_rounds(args)


def verify_data(dataset):
    manifest = read_json(dataset / 'manifest.json')
    clients = manifest['clients']
    if (manifest['experiment'] != 'commonsense170k' or
            [c['partition'] for c in clients] != list(DOMAINS) or
            [c['client_id'] for c in clients] != list(range(8)) or
            any(c['selected'] != 1000 for c in clients)):
        raise ValueError('Expected eight named Commonsense clients with exactly 1000 rows each')
    for name, expected in manifest['files'].items():
        path = dataset / name
        if path.stat().st_size != expected['bytes'] or sha256(path) != expected['sha256']:
            raise ValueError(f'Dataset checksum mismatch: {path}')
    rows, seen = {}, set()
    for cid, domain in enumerate(DOMAINS):
        samples = [json.loads(line) for line in (dataset / 'train' / f'{cid}.jsonl').read_text(encoding='utf-8').splitlines()]
        if len(samples) != 1000:
            raise ValueError(f'Client {cid} must have exactly 1000 rows')
        for row in samples:
            if row['client_id'] != cid or row['source_dataset'] != domain or row['source_split'] != 'train':
                raise ValueError('Invalid client/source provenance')
            if row['sample_id'] in seen:
                raise ValueError('Repeated sample ID')
            seen.add(row['sample_id'])
        rows[cid] = samples
    return manifest, rows


def model_fingerprint(path):
    files = sorted(set(path.glob('*.safetensors')) | set(path.glob('pytorch_model*.bin')))
    if not files:
        raise ValueError(f'No local model weights found: {path}')
    files += [path / name for name in ('config.json', 'tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json')
              if (path / name).exists()]
    return {file.name: sha256(file) for file in files}


def prepare_framework(args):
    """Preflight before model allocation; clients consume the same cached rows."""
    if hasattr(args, '_commonsense_prepared'):
        return args._commonsense_prepared
    validate_args(args)
    from transformers import AutoTokenizer
    ensure_runtime()
    dataset, model_path = (ROOT / 'dataset' / args.dataset).resolve(), Path(args.model).resolve()
    manifest, raw = verify_data(dataset)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    encoded, audit = {}, {}
    print('Commonsense: validating source files, full training text and harness tasks ...', flush=True)
    for cid, domain in enumerate(DOMAINS):
        encoded[cid] = [encode_example(tokenizer, row, args.max_length) for row in raw[cid]]
        lengths = [len(row['input_ids']) for row in encoded[cid]]
        task = build_task(domain, dataset / 'benchmarks' / f'{domain}.jsonl')
        audit[domain] = {'train_rows': len(encoded[cid]), 'max_training_tokens': max(lengths),
                         'above_old_512_limit': sum(n > 512 for n in lengths),
                         'evaluation_rows': len(task.eval_docs), 'multiple_input': bool(task.multiple_input)}
        print(f'Client {cid} {domain}: {len(encoded[cid])} training rows, max {max(lengths)} tokens', flush=True)
    settings = {key: getattr(args, key) for key in ('alg', 'cn', 'sr', 'mode', 'seed', 'deterministic',
                'task_type', 'bs', 'grad_accum', 'epoch', 'step', 'lr', 'lora_rank', 'lora_alpha',
                'lora_dropout', 'max_length', 'gradient_checkpointing', 'device')}
    settings.update({'lr_decay': 0.99, 'dtype': 'float16', 'encoding': TRAIN_PROTOCOL})
    if args.alg == 'fedca':
        from utils.commonsense_fedca import validate_fedca
        validate_fedca(args)
        settings['fedca'] = {name: getattr(args, name) for name in ('ca_lambda', 'ca_tau', 'ca_eta_mode',
                            'ca_eta', 'ca_eta_grid', 'ca_beta', 'ca_probe_size', 'ca_probe_bs', 'ca_probe_diag')}
        settings['fedca']['ca_kappa'] = 'inf' if math.isinf(float(args.ca_kappa)) else float(args.ca_kappa)
    source_files = ('main.py', 'alg/ftbase.py', 'alg/fedit.py', 'alg/centralized.py', 'alg/localonly.py',
                    'utils/train_utils.py', 'utils/data_utils.py', 'utils/model_utils.py', 'utils/seed_utils.py', 'utils/options.py',
                    'utils/commonsense_framework.py', 'utils/commonsense_training.py', 'utils/runtime_utils.py')
    if args.alg == 'fedca':
        source_files += ('alg/fedca.py', 'utils/ca_utils.py', 'utils/commonsense_fedca.py')
    print('Commonsense: checking base-model and framework fingerprints ...', flush=True)
    identity_spec = {'protocol': FRAMEWORK_PROTOCOL, 'training': settings,
                     'dataset_manifest_sha256': sha256(dataset / 'manifest.json'),
                     'model_files': model_fingerprint(model_path),
                     'packages': {name: importlib.metadata.version(name) for name in ('torch', 'transformers', 'peft', 'numpy')},
                     'code': {name: sha256(ROOT / name) for name in source_files}}
    from utils.commonsense_fedca import preserve_baseline_identity
    identity_spec = preserve_baseline_identity(Path(args.suffix), identity_spec)
    identity = fingerprint(identity_spec)
    per_client = math.ceil(math.ceil((8000 if args.alg == 'centralized' else 1000)/args.bs)/args.grad_accum)
    updates = min(args.step, per_client*args.epoch) if args.step else per_client*args.epoch
    report = {'identity': identity, 'framework': 'main.FedSim -> alg -> FTBaseClient -> utils.train_utils.Trainer',
              'training': settings, 'clients': audit, 'rounds': args.rnd, 'eval_at': evaluation_rounds(args),
              'train_rows_per_complete_epoch': 8000, 'optimizer_updates_per_round': updates*args.cn,
              'diagnostic_budget': bool(args.step and args.step < per_client*args.epoch), 'training_started': False}
    prepared = {'identity': identity, 'identity_spec': identity_spec, 'dataset': dataset,
                'encoded': encoded, 'tokenizer': tokenizer, 'report': report}
    args._commonsense_prepared = prepared
    return prepared


def load_client_data(args, idx):
    from datasets import Dataset
    prepared = prepare_framework(args)
    rows = ([row for cid in range(8) for row in prepared['encoded'][cid]]
            if args.alg == 'centralized' else prepared['encoded'][idx])
    return {'train': Dataset.from_list(rows)}


class CommonsenseRun:
    """Checkpoint hooks used by the real algorithms and the real Trainer."""
    def __init__(self, args, server):
        self.args, self.server = args, server
        self.prepared = prepare_framework(args)
        self.identity = self.prepared['identity']
        self.output = Path(args.suffix).resolve()
        self.store = CheckpointStore(self.output / 'resume', self.identity)
        self.state = self.store.load()
        manifest_path = self.output / 'run_manifest.json'
        if args.resume == 'never' and (self.state is not None or manifest_path.exists()):
            raise ValueError('resume=never requires a new suffix')
        if self.state is None and (manifest_path.exists() or any(self.output.glob('round_*/complete.json'))):
            raise ValueError('Saved output exists but its continuation checkpoint is missing')
        if args.eval_only and self.state is None:
            raise ValueError('No continuation state exists for eval_only')
        if self.state is None:
            initial = adapter_state(server.model)
            self.state = {'format': FRAMEWORK_PROTOCOL, 'completed_rounds': 0, 'pending': {}, 'active': None,
                          'global': initial, 'local': {str(c.id): cpu_tree(initial) for c in server.clients}
                          if args.alg == 'localonly' else {}, 'wall_clock_time': 0.0}
            self.store.save(self.state)
        if self.state['completed_rounds'] > args.rnd:
            raise ValueError('rnd is smaller than the completed training rounds')
        if not manifest_path.exists():
            atomic_json(manifest_path, {'identity': self.identity, 'framework': FRAMEWORK_PROTOCOL,
                'identity_spec': self.prepared['identity_spec'], 'readiness': self.prepared['report'],
                'dataset': str(self.prepared['dataset']), 'initial_rounds': args.rnd})
        self.server.global_lora = self._on_device(self.state['global'])
        if args.alg == 'localonly':
            self.server.client_loras = {int(cid): self._on_device(value) for cid, value in self.state['local'].items()}
        self.server.model.load_state_dict(self.server.global_lora, strict=False)
        self.server.wall_clock_time = self.state['wall_clock_time']
        print(f"Framework continuation: {self.state['completed_rounds']} completed rounds, "
              f"{len(self.state['pending'])} completed clients in next round", flush=True)

    @property
    def completed_rounds(self):
        return self.state['completed_rounds']

    def _on_device(self, state):
        params = dict(self.server.model.named_parameters())
        return {key: value.to(params[key].device).clone() for key, value in state.items()}

    def active_for(self, client):
        active = self.state['active']
        if active is not None and (active['client_id'] != client.id or active['round_index'] != self.server.round):
            raise ValueError('Active checkpoint client/round does not match scheduled training')
        return active

    def checkpoint_active(self, client, active):
        self.state['active'] = active
        self.store.save(self.state)

    def restore_completed_client(self, client):
        saved = self.state['pending'].get(str(client.id))
        if saved is None:
            return False
        client.lora = self._on_device(saved['adapter'])
        client.last_train_loss = saved['metrics']['mean_loss']
        client.trainer.last_train_metrics = saved['metrics']
        client.training_time = saved['training_time']
        if self.args.alg == 'localonly':
            self.server.client_loras[client.id] = self._on_device(saved['adapter'])
        print(f'Client {client.id}: reuse completed update from checkpoint', flush=True)
        return True

    def commit_client(self, client):
        adapter = cpu_tree(client.lora)
        self.state['pending'][str(client.id)] = {'adapter': adapter,
            'metrics': client.trainer.last_train_metrics, 'training_time': float(client.training_time)}
        if self.args.alg == 'localonly':
            self.state['local'][str(client.id)] = adapter
        self.state['active'] = None
        self.store.save(self.state)

    def commit_round(self):
        rnd = self.server.round + 1
        if set(self.state['pending']) != {str(c.id) for c in self.server.clients}:
            raise RuntimeError('Cannot commit a round with missing client updates')
        # Aggregation has ALREADY happened in alg.fedit/alg.centralized.Server.run.
        # The Local-only algorithm deliberately performs no aggregation.
        self.state['global'] = cpu_tree(self.server.global_lora)
        if self.args.alg == 'localonly':
            self.state['local'] = {str(cid): cpu_tree(value) for cid, value in self.server.client_loras.items()}
        adapters = self.state['local'] if self.args.alg == 'localonly' else {'global': self.state['global']}
        directory = self.output / f'round_{rnd:03d}'
        path, marker = directory / 'adapters.pt', directory / 'complete.json'
        payload = {'identity': self.identity, 'round': rnd, 'method': self.args.alg, 'adapters': adapters}
        if marker.exists():
            meta = read_json(marker)
            if meta['identity'] != self.identity or meta['round'] != rnd or sha256(path) != meta['sha256']:
                raise ValueError('Existing round snapshot mismatch')
            prior = torch.load(path, map_location='cpu', weights_only=True)['adapters']
            if set(prior) != set(adapters) or any(set(prior[cid]) != set(value) or
                    any(not torch.equal(tensor, prior[cid][key]) for key, tensor in value.items())
                    for cid, value in adapters.items()):
                raise ValueError('Reconstructed round differs from existing committed snapshot')
        else:
            atomic_torch(path, payload)
            atomic_json(marker, {'identity': self.identity, 'round': rnd, 'method': self.args.alg,
                                  'file': 'adapters.pt', 'sha256': sha256(path)})
        metrics = [self.state['pending'][str(c.id)]['metrics'] for c in self.server.clients]
        atomic_json(directory / 'training_metrics.json', {'round': rnd, 'method': self.args.alg,
                    'clients': metrics, 'train_examples': sum(m['examples'] for m in metrics)})
        self.state['completed_rounds'] = rnd
        self.state['wall_clock_time'] = self.server.wall_clock_time
        self.state['pending'], self.state['active'] = {}, None
        self.store.save(self.state)
        atomic_json(self.output / 'progress.json', {'completed_rounds': rnd, 'requested_rounds': self.args.rnd,
                                                   'identity': self.identity, 'test_gap': self.args.test_gap})

    def evaluate(self, round_number):
        directory = self.output / f'round_{round_number:03d}'
        meta = read_json(directory / 'complete.json')
        try:
            summary = evaluate_round(self.server.model, self.server.clients[0].tokenizer,
                self.prepared['dataset'], directory / 'adapters.pt', meta['sha256'], self.output,
                self.args.alg, round_number, self.identity, batch_size=self.args.eval_batch_size,
                max_length=self.args.eval_max_length, limit=self.args.eval_limit, seed=self.args.seed)
            return {'macro_acc': summary['macro_acc'],
                    **{domain+'_acc': row['acc,none'] for domain, row in summary['per_task'].items()}}
        finally:
            # Evaluation may load an OLDER snapshot. Restore the latest server
            # state before the next algorithm.local_run starts its first client.
            self.server.model.load_state_dict(self.server.global_lora, strict=False)

    def recover_evaluations(self, eval_only=False):
        rounds = evaluation_rounds(self.args)
        if eval_only:
            missing = [rnd for rnd in rounds if rnd > self.completed_rounds or
                       not (self.output / f'round_{rnd:03d}/complete.json').exists()]
            if missing:
                raise ValueError(f'Requested rounds do not have completed snapshots: {missing}')
        previous = self.server.round
        try:
            for rnd in rounds:
                if rnd <= self.completed_rounds:
                    self.server.round = rnd-1
                    self.server.test_all()
        finally:
            self.server.round = previous

    def mark_complete(self):
        atomic_json(self.output / 'completed.json', {'completed_rounds': self.completed_rounds,
                    'evaluation_rounds': evaluation_rounds(self.args), 'identity': self.identity,
                    'eval_limit': self.args.eval_limit, 'diagnostic_only': self.args.eval_limit is not None or
                    self.prepared['report']['diagnostic_budget'], 'framework': FRAMEWORK_PROTOCOL})
