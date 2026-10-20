"""Train/evaluate the pooled zsRE baseline in the completed chat/EOS run."""

import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone
import argparse


ROOT = Path(__file__).resolve().parent
RUN = ROOT / 'exp' / 'zsre8_chat_eos_v2'
STATE = RUN / 'centralized_sequence_state.json'
LOGS = RUN / 'centralized_sequence_logs'
METHODS = ('base', 'local', 'fedit', 'fedrotlora')
OUTPUT_NAME = 'centralized'


def save(state):
    state['updated_at'] = datetime.now(timezone.utc).isoformat()
    temporary = STATE.with_suffix('.tmp')
    temporary.write_text(json.dumps(state, indent=2), encoding='utf-8')
    temporary.replace(STATE)


def read_jsonl(path):
    with path.open(encoding='utf-8') as handle:
        return [json.loads(line) for line in handle if line.strip()]


def preflight():
    if STATE.exists():
        raise RuntimeError(f'Centralized state already exists: {STATE}')
    if (RUN / OUTPUT_NAME).exists():
        raise RuntimeError('Centralized output already exists; refusing overwrite')
    data_root = ROOT / 'dataset' / 'zsre'
    pooled = read_jsonl(data_root / 'train' / 'pooled.jsonl')
    pooled_by_id = {row['sample_id']: row for row in pooled}
    if len(pooled) != 4000 or len(pooled_by_id) != 4000:
        raise ValueError('Expected 4000 distinct pooled train records')
    client_by_id = {}
    for client_id in range(8):
        train = read_jsonl(data_root / 'train' / f'{client_id}.jsonl')
        test = read_jsonl(data_root / 'test' / f'{client_id}.jsonl')
        if len(train) != 500 or len(test) != 500:
            raise ValueError(f'Client {client_id} must have 500 train and test records')
        for row in train:
            if row['sample_id'] in client_by_id:
                raise ValueError(f'Duplicate training sample: {row["sample_id"]}')
            client_by_id[row['sample_id']] = row
    if pooled_by_id != client_by_id:
        raise ValueError('Pooled train data differ from the union of eight clients')
    for method in METHODS:
        path = RUN / method / 'evaluation' / 'zsre' / 'metrics.json'
        metrics = json.loads(path.read_text(encoding='utf-8'))
        if (metrics['generation'].get('protocol') != 'zsre_chat_eos_v2'
                or set(metrics['per_client']) != {str(i) for i in range(8)}
                or any(row['num_samples'] != 500 for row in metrics['per_client'].values())):
            raise ValueError(f'{method} is not a complete comparable evaluation')


def main():
    global STATE, LOGS, OUTPUT_NAME
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--roundwise', action='store_true',
                        help='save every round and evaluate all eight test groups')
    args = parser.parse_args()
    if args.roundwise:
        OUTPUT_NAME = 'centralized_roundwise'
        STATE = RUN / 'centralized_roundwise_sequence_state.json'
        LOGS = RUN / 'centralized_roundwise_sequence_logs'
    preflight()
    LOGS.mkdir(parents=True, exist_ok=False)
    commands = (
        ('train', ['main.py', '--config', 'config/config.zsre_centralized.yaml',
                   '--suffix', f'zsre8_chat_eos_v2/{OUTPUT_NAME}']),
        ('eval', ['eval_zsre.py', '--config', 'config/config.zsre_centralized.yaml',
                  '--method', 'centralized', '--suffix', f'zsre8_chat_eos_v2/{OUTPUT_NAME}']),
        ('summarize', ['summarize_stage1.py', '--run', 'zsre8_chat_eos_v2',
                       '--methods', 'base,local,centralized,fedit,fedrotlora', '--metric', 'em',
                       '--centralized-dir', OUTPUT_NAME,
                       '--output', 'stage1_summary_em_centralized_roundwise.csv'
                       if args.roundwise else 'stage1_summary_em.csv']),
    )
    state = {'pid': os.getpid(), 'python': sys.executable, 'status': 'running',
             'variant': OUTPUT_NAME, 'completed': []}
    with STATE.open('x', encoding='utf-8') as handle:
        json.dump(state, handle, indent=2)
    for step, arguments in commands:
        command = [sys.executable, '-X', 'utf8', '-u', *arguments]
        log = LOGS / f'{step}.log'
        state.update(step=step, command=command, log=str(log), child_pid=None)
        save(state)
        try:
            with log.open('w', encoding='utf-8') as handle:
                child = subprocess.Popen(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT)
                state['child_pid'] = child.pid
                save(state)
                code = child.wait()
            if code:
                raise RuntimeError(f'{step} exited with code {code}; see {log}')
            if step == 'eval':
                metrics = json.loads((RUN / OUTPUT_NAME / 'evaluation' / 'zsre' /
                                      'metrics.json').read_text(encoding='utf-8'))
                if (metrics['generation'].get('protocol') != 'zsre_chat_eos_v2'
                        or set(metrics['per_client']) != {str(i) for i in range(8)}
                        or any(row['num_samples'] != 500 for row in metrics['per_client'].values())):
                    raise ValueError('Centralized evaluation must cover all 8 x 500 samples')
            if step == 'train' and args.roundwise:
                for round_idx in range(10):
                    path = RUN / OUTPUT_NAME / 'round_evaluation' / 'zsre' / f'round_{round_idx:03d}' / 'metrics.json'
                    metrics = json.loads(path.read_text(encoding='utf-8'))
                    if (metrics['generation'].get('protocol') != 'zsre_chat_eos_v2'
                            or set(metrics['per_client']) != {str(i) for i in range(8)}
                            or any(row['num_samples'] != 500 for row in metrics['per_client'].values())):
                        raise ValueError(f'Round {round_idx}: expected all 8 x 500 test samples')
        except Exception as exc:
            state.update(status='failed', error=str(exc), child_pid=None)
            save(state)
            return 1
        state['completed'].append(step)
        state['child_pid'] = None
        save(state)
    state.update(status='complete', child_pid=None)
    save(state)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
