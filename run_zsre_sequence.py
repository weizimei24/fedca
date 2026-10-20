"""Run the user-authorized overnight zsRE sequence, stopping on first failure."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone


ROOT = Path(__file__).resolve().parent
RUN = ROOT / 'exp' / 'zsre8_chat_eos_v2'
STATE = RUN / 'sequence_state.json'


def save(state):
    state['updated_at'] = datetime.now(timezone.utc).isoformat()
    temporary = STATE.with_suffix('.tmp')
    temporary.write_text(json.dumps(state, indent=2), encoding='utf-8')
    temporary.replace(STATE)


def main():
    global RUN, STATE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', default='zsre8_chat_eos_v2')
    args = parser.parse_args()
    if Path(args.run).name != args.run or args.run in ('.', '..'):
        raise ValueError('--run must be a single directory name')
    RUN = ROOT / 'exp' / args.run
    STATE = RUN / 'sequence_state.json'
    logs = RUN / 'sequence_logs'
    logs.mkdir(parents=True, exist_ok=True)
    commands = [('base_eval', ['eval_zsre.py', '--config', 'config/config.zsre_base.yaml',
                              '--base-only', '--method', 'base', '--suffix', f'{args.run}/base'])]
    for method, config in (
        ('local', 'config/config.zsre_localonly.yaml'),
        ('fedit', 'config/config.zsre_fedit.yaml'),
        ('fedrotlora', 'config/config.zsre_fedrotlora.yaml'),
    ):
        suffix = ['--suffix', f'{args.run}/{method}']
        commands.append((method + '_train', ['main.py', '--config', config, *suffix]))
        commands.append((method + '_eval', ['eval_zsre.py', '--config', config, '--method', method, *suffix]))
    commands.append(('summarize', ['summarize_stage1.py', '--run', args.run, '--metric', 'em']))
    if STATE.exists():
        previous = json.loads(STATE.read_text(encoding='utf-8'))
        raise RuntimeError(f'Sequence already has state ({previous.get("status")}); inspect before restarting')
    state = {'pid': os.getpid(), 'python': sys.executable, 'status': 'running', 'completed': []}
    # Exclusive creation prevents two simultaneously launched runners.
    with STATE.open('x', encoding='utf-8') as output:
        json.dump(state, output, indent=2)
    save(state)
    for name, arguments in commands:
        command = [sys.executable, '-X', 'utf8', '-u', *arguments]
        state.update(step=name, command=command, log=str(logs / (name + '.log')))
        save(state)
        try:
            with (logs / (name + '.log')).open('w', encoding='utf-8') as output:
                child = subprocess.Popen(command, cwd=ROOT, stdout=output, stderr=subprocess.STDOUT)
                state['child_pid'] = child.pid
                save(state)
                code = child.wait()
        except Exception as exc:
            state.update(status='failed', error=str(exc))
            save(state)
            return 1
        if code:
            state.update(status='failed', exit_code=code)
            save(state)
            return code
        if name.endswith('_eval'):
            try:
                method = name[:-5]
                metrics = json.loads((RUN / method / 'evaluation/zsre/metrics.json').read_text(encoding='utf-8'))
                if (metrics['generation']['protocol'] != 'zsre_chat_eos_v2'
                        or set(metrics['per_client']) != {str(i) for i in range(8)}
                        or any(row['num_samples'] != 500 for row in metrics['per_client'].values())):
                    raise ValueError(f'{method}: expected new protocol and 8 x 500 test samples')
            except Exception as exc:
                state.update(status='failed', error=str(exc))
                save(state)
                return 1
        state['completed'].append(name)
        save(state)
    state.update(status='complete', child_pid=None)
    save(state)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
