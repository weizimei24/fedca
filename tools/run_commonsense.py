"""Compatibility CLI: translate options and execute the original main.py framework."""
from __future__ import annotations
import argparse
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]


def framework_command(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--method', choices=('fedit', 'centralized', 'localonly', 'fedca', 'ffalora', 'fedsvd', 'fedexlora', 'fedrotlora', 'flexlora', 'fedmomentum'), required=True)
    p.add_argument('--rounds', type=int)
    p.add_argument('--eval-every', type=int)
    p.add_argument('--checkpoint-steps', type=int)
    p.add_argument('--eval-batch-size', type=int)
    p.add_argument('--eval-max-length', type=int)
    p.add_argument('--eval-limit', type=int)
    p.add_argument('--eval-rounds')
    p.add_argument('--eval-only', action='store_true')
    p.add_argument('--check-only', action='store_true')
    p.add_argument('--resume', choices=('auto', 'never'))
    p.add_argument('--bs', type=int)
    p.add_argument('--grad-accum', type=int)
    p.add_argument('--lr', type=float)
    p.add_argument('--seed', type=int)
    p.add_argument('--max-length', type=int)
    p.add_argument('--lora-rank', type=int)
    p.add_argument('--lora-alpha', type=int)
    p.add_argument('--lora-dropout', type=float)
    p.add_argument('--device', help='physical CUDA device number; e.g. 0 or cuda:0')
    p.add_argument('--model', type=Path)
    p.add_argument('--dataset', type=Path)
    p.add_argument('--output', type=Path, help='output directory under this project exp/')
    p.add_argument('--gradient-checkpointing', action=argparse.BooleanOptionalAction, default=None)
    p.add_argument('--dtype', choices=('float16',), help='the original framework uses float16')
    a = p.parse_args(argv)
    command = ['main.py', '--config', str(ROOT / 'config' / 'experiments' / 'commonsense170k.yaml'), '--alg', a.method]
    if a.method == 'centralized':
        command += ['--cn', '1']
    mappings = {'rounds': 'rnd', 'eval_every': 'test_gap', 'checkpoint_steps': 'checkpoint_steps',
                'eval_batch_size': 'eval_batch_size', 'eval_max_length': 'eval_max_length',
                'eval_limit': 'eval_limit', 'eval_rounds': 'eval_rounds', 'resume': 'resume',
                'bs': 'bs', 'grad_accum': 'grad_accum', 'lr': 'lr', 'seed': 'seed',
                'max_length': 'max_length', 'lora_rank': 'lora_rank', 'lora_alpha': 'lora_alpha',
                'lora_dropout': 'lora_dropout', 'model': 'model'}
    for source, target in mappings.items():
        value = getattr(a, source)
        if value is not None:
            command += ['--'+target, str(value)]
    for name in ('eval_only', 'check_only'):
        if getattr(a, name):
            command.append('--'+name)
    if a.device is not None:
        command += ['--device', a.device.removeprefix('cuda:')]
    if a.dataset is not None:
        path = a.dataset.resolve()
        try:
            dataset = str(path.relative_to(ROOT / 'dataset'))
        except ValueError:
            dataset = str(path)
        command += ['--dataset', dataset]
    if a.output is not None:
        try:
            suffix = a.output.resolve().relative_to(ROOT / 'exp')
        except ValueError:
            p.error('--output must be below the project exp/ directory; use main.py --suffix for framework output')
        command += ['--suffix', suffix.as_posix()]
    if a.gradient_checkpointing is not None:
        command.append('--gradient_checkpointing' if a.gradient_checkpointing else '--no-gradient_checkpointing')
    return command


if __name__ == '__main__':
    sys.argv = framework_command()
    print('Using existing FedSim framework: ' + ' '.join(sys.argv), flush=True)
    sys.path.insert(0, str(ROOT))
    runpy.run_path(str(ROOT / 'main.py'), run_name='__main__')
