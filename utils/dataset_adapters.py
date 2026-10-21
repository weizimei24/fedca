"""Datasets supply encoding and task definitions, never continuation logic."""
from dataclasses import dataclass, replace
import json
from pathlib import Path

from utils.runtime_utils import sha256

ROOT = Path(__file__).resolve().parents[1]
TRAIN_PROTOCOL = 'commonsense_chat_eos_v1'
DOMAINS = ('boolq', 'piqa', 'social_i_qa', 'hellaswag', 'winogrande',
           'arc_easy', 'arc_challenge', 'openbookqa')


def read_rows(path):
    # JSONL is delimited by physical newlines. str.splitlines() also splits
    # valid U+0085/U+2028/U+2029 characters inside mathematical JSON strings.
    rows = []
    with Path(path).open(encoding='utf-8') as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f'Invalid JSONL in {path}, line {line_number}') from exc
    return rows


@dataclass
class DatasetAdapter:
    name: str
    encoding: str
    evaluator: str
    primary_metric: str
    evaluator_packages: tuple = ()

    def prompt(self, tokenizer, text):
        return tokenizer.apply_chat_template(
            [{'role': 'user', 'content': text}], tokenize=True,
            add_generation_prompt=True, enable_thinking=False)

    def encode(self, tokenizer, row, max_length):
        prompt = self.prompt(tokenizer, row['input_ids'])
        answer = tokenizer(str(row['label']), add_special_tokens=False)['input_ids']
        if not answer or tokenizer.eos_token_id is None or tokenizer.pad_token_id is None:
            raise ValueError('Nonempty targets and EOS/pad IDs are required')
        target = answer + [tokenizer.eos_token_id]
        if len(prompt) + len(target) > max_length:
            raise ValueError(f"{row.get('sample_id', 'sample')} has {len(prompt)+len(target)} tokens, "
                             f'exceeding max_length={max_length}; no silent truncation')
        return {'input_ids': prompt + target, 'labels': [-100] * len(prompt) + target}

    def tasks(self, path):
        raise NotImplementedError

    def validate_clients(self, files):
        pass

    def validate_tasks(self, tasks, files):
        pass


class ZsreAdapter(DatasetAdapter):
    def prompt(self, tokenizer, text):
        from utils.zsre_utils import encode_zsre_prompt
        return encode_zsre_prompt(tokenizer, text)

    def tasks(self, path):
        return [(f'client_{p.stem}', p, int(p.stem)) for p in sorted((path / 'test').glob('*.jsonl'))
                if p.stem.isdigit()]

    def validate_tasks(self, tasks, files):
        if sorted(cid for _, _, cid in tasks) != list(range(len(files))):
            raise ValueError('zsRE evaluation groups must match the original client IDs')


class CommonsenseAdapter(DatasetAdapter):
    def tasks(self, path):
        generation = [(domain, path / 'test' / f'{cid}.jsonl', cid) for cid, domain in enumerate(DOMAINS)]
        native = [(f'native_{domain}', path / 'benchmarks' / f'{domain}.jsonl', cid)
                  for cid, domain in enumerate(DOMAINS)]
        return generation + native

    def validate_clients(self, files):
        if len(files) != len(DOMAINS):
            raise ValueError('Commonsense170K requires eight source clients')


class MathAdapter(DatasetAdapter):
    def tasks(self, path):
        return [(name, path / 'evaluation' / f'{name}.jsonl', None) for name in ('gsm8k', 'math')]


DATASETS = {
    'zsre': ZsreAdapter('zsre', 'zsre_chat_eos_dynamic_v1', 'generation_zsre_v2', 'em'),
    'commonsense170k': CommonsenseAdapter('commonsense170k', TRAIN_PROTOCOL,
        'commonsense_chat_generation_plus_native_v2', 'generation_accuracy', ('lm_eval',)),
    'metamathqa': MathAdapter('metamathqa', 'math_chat_eos_v1', 'math_chat_greedy_training_prompt_v3',
        'exact_match', ('lm_eval',)),
}


def adapter_for(args):
    name = getattr(args, 'task', '')
    if not name:
        if args.dataset == 'zsre':
            name = 'zsre'
        else:
            path = ROOT / 'dataset' / args.dataset / 'manifest.json'
            if not path.is_file():
                raise ValueError('Specify --task or use a processed dataset with manifest.json')
            name = json.loads(path.read_text(encoding='utf-8'))['experiment']
    if name not in DATASETS:
        raise ValueError(f'No dataset adapter for {name!r}')
    adapter = DATASETS[name]
    if name == 'commonsense170k' and getattr(args, 'commonsense_eval', 'generation') == 'native':
        adapter = replace(adapter, primary_metric='acc,none')
    return adapter


def prepare_data(args, tokenizer):
    adapter = adapter_for(args)
    folder = (ROOT / 'dataset' / args.dataset).resolve()
    manifest_path = folder / 'manifest.json'
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        if manifest['experiment'] != adapter.name:
            raise ValueError('Dataset manifest experiment differs from task')
        for name, expected in manifest['files'].items():
            path = (folder / name).resolve()
            if not path.is_relative_to(folder) or not path.is_file():
                raise ValueError(f'Invalid/missing manifest file: {name}')
            if path.stat().st_size != expected['bytes'] or sha256(path) != expected['sha256']:
                raise ValueError(f'Dataset checksum mismatch: {path}')
    files = sorted(p for p in (folder / 'train').glob('*.jsonl') if p.stem.isdigit())
    if not files or sorted(int(p.stem) for p in files) != list(range(len(files))):
        raise ValueError('Expected contiguous client train files: 0.jsonl, 1.jsonl, ...')
    adapter.validate_clients(files)
    expected = 1 if args.alg == 'centralized' else len(files)
    if args.cn is None:
        args.cn = expected
    if args.cn != expected:
        raise ValueError(f'{args.alg} requires cn={expected} for {adapter.name}')
    encoded, audit = {}, {}
    for path in files:
        cid = int(path.stem)
        raw = read_rows(path)
        if not raw:
            raise ValueError(f'Empty training file: {path}')
        if any('client_id' in row and row['client_id'] != cid for row in raw):
            raise ValueError(f'Wrong client_id in {path}')
        encoded[cid] = [adapter.encode(tokenizer, row, args.max_length) for row in raw]
        audit[str(cid)] = {'rows': len(raw), 'max_tokens': max(len(r['input_ids']) for r in encoded[cid])}
        print(f'{adapter.name} client {cid}: {len(raw)} rows, max {audit[str(cid)]["max_tokens"]} tokens', flush=True)
    tasks = adapter.tasks(folder)
    if not tasks or any(not path.is_file() for _, path, _ in tasks):
        raise ValueError('Missing evaluation tasks/data')
    adapter.validate_tasks(tasks, files)
    if args.alg == 'localonly' and any(cid is None for _, _, cid in tasks):
        # Mathematics has global evaluation sets; each local model evaluates both.
        tasks = [(f'client_{cid}_{name}', path, cid) for cid in range(len(files)) for name, path, _ in tasks]
    data_files = files + [p for _, p, _ in tasks]
    data_files += [p for p in (folder / 'manifest.json', folder / 'client_mapping.json', folder / 'stats.json') if p.exists()]
    return {'adapter': adapter, 'dataset': folder, 'encoded': encoded, 'tasks': tasks,
            'audit': audit, 'data_files': {p.relative_to(folder).as_posix(): sha256(p) for p in set(data_files)}}


def load_client_data(args, idx):
    from datasets import Dataset
    prepared = args._experiment_prepared
    rows = ([row for cid in sorted(prepared['encoded']) for row in prepared['encoded'][cid]]
            if args.alg == 'centralized' else prepared['encoded'][idx])
    return {'train': Dataset.from_list(rows)}
