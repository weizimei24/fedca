"""Evaluate the unmodified local base model with the shared task protocols.

No PEFT adapter is constructed or loaded. --eval_limit is a per-task limit.
This entry point uses the same encoders/scorers as the training runner.
"""
from pathlib import Path
import json
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.dataset_adapters import prepare_data, read_rows
from utils.evaluation_adapters import evaluate_task, validate_evaluator
from utils.model_utils import load_model, local_model_fingerprint
from utils.options import build_parser
from utils.runtime_utils import atomic_json, fingerprint, run_lock, sha256, source_sha256, replace_file
from utils.seed_utils import set_global_seed


def write_predictions(path, rows, predictions):
    if len(rows) != len(predictions):
        raise ValueError('Prediction count differs from selected evaluation rows')
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.jsonl.tmp')
    with temporary.open('w', encoding='utf-8') as stream:
        for row, prediction in zip(rows, predictions):
            record = {'prompt': row['input_ids'], **prediction}
            if record['prompt'] != row['input_ids']:
                record['dataset_prompt'] = row['input_ids']
            if 'choice_labels' in row:
                record['choice_labels'] = row['choice_labels']
            stream.write(json.dumps(record, ensure_ascii=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())
    replace_file(temporary, path)


def check_encoding(prepared, tokenizer):
    count = 0
    for rows in prepared['encoded'].values():
        for row in rows:
            ids, labels = row['input_ids'], row['labels']
            supervised = [index for index, label in enumerate(labels) if label != -100]
            if (len(ids) != len(labels) or not supervised or supervised[0] == 0 or
                    supervised != list(range(supervised[0], len(ids))) or
                    labels[-1] != tokenizer.eos_token_id or
                    any(labels[index] != ids[index] for index in supervised)):
                raise ValueError('Invalid prompt masking, target supervision or EOS encoding')
            count += 1
    return {'training_rows_checked': count, 'prompt_masked': True,
            'answer_and_eos_supervised': True, 'enable_thinking': False}


def cached_result(path, cache_key, output):
    if not path.exists():
        return None
    result = json.loads(path.read_text(encoding='utf-8'))
    checksum = fingerprint({key: value for key, value in result.items() if key != 'cache_sha256'})
    if result.get('cache_key') != cache_key or result.get('cache_sha256') != checksum:
        raise ValueError(f'Base evaluation cache differs: {path}')
    if result.get('predictions_file'):
        predictions = output / result['predictions_file']
        if not predictions.is_file() or sha256(predictions) != result['predictions_sha256']:
            raise ValueError(f'Base prediction file is missing or corrupt: {predictions}')
    return result


def main():
    parser = build_parser()
    args = parser.parse_args()
    if args.runtime != 'shared' or args.mode != 'prototype' or args.task_type != 'CAUSAL_LM':
        parser.error('Base evaluation requires one of config/experiments/*.yaml shared task configurations')
    for name in ('eval_batch_size', 'eval_max_length', 'max_new_tokens', 'commonsense_max_new_tokens'):
        value = getattr(args, name)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            parser.error(f'{name} must be a positive integer')
    if args.eval_limit is not None and (isinstance(args.eval_limit, bool) or args.eval_limit <= 0):
        parser.error('eval_limit must be a positive integer')
    set_global_seed(args.seed, device=args.device, deterministic=args.deterministic)
    args._shared = True
    # Base has one unmodified model for all original client test groups.
    args.alg, args.cn, args.gradient_checkpointing = 'fedit', None, False
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    prepared = prepare_data(args, tokenizer)
    adapter = prepared['adapter']
    tasks = adapter.scheduled_tasks(prepared['tasks'], 0, args)
    validate_evaluator(adapter, tasks)
    encoding = check_encoding(prepared, tokenizer)
    sources = ('tools/eval_base.py', 'utils/dataset_adapters.py', 'utils/evaluation_adapters.py',
               'utils/commonsense_eval.py', 'utils/zsre_utils.py', 'utils/model_utils.py', 'eval_zsre.py')
    settings = {'method': 'base', 'task': adapter.name, 'protocol': adapter.evaluator,
        'model_files': local_model_fingerprint(args.model), 'seed': args.seed,
        'eval_limit': args.eval_limit, 'batch_size': args.eval_batch_size,
        'max_length': args.eval_max_length, 'max_new_tokens': args.max_new_tokens,
        'commonsense_max_new_tokens': args.commonsense_max_new_tokens,
        'commonsense_native_eval': args.commonsense_native_eval,
        'generation': {'do_sample': False, 'use_model_defaults': False},
        'primary_metric': adapter.primary_metric, 'data_files': prepared['data_files'],
        'implementation': {name: source_sha256(ROOT / name) for name in sources}}
    import importlib.metadata
    packages = ['torch', 'transformers', 'tokenizers', 'accelerate']
    if adapter.name == 'metamathqa' or any(name.startswith('native_') for name, _, _ in tasks):
        packages.append('lm_eval')
    settings['packages'] = {name: importlib.metadata.version(name) for name in packages}
    evaluation_id = fingerprint(settings)
    suffix = (Path('base') / adapter.name / Path(args.dataset).name
              if args.suffix == 'default' else Path(args.suffix))
    if suffix.is_absolute() or '..' in suffix.parts:
        parser.error('suffix must be relative to exp/')
    parent = ROOT / 'exp' / suffix
    output = parent / 'base_evaluation' / evaluation_id[:16]
    model, results = None, {}
    with run_lock(output):
        atomic_json(output / 'run_manifest.json', {'settings': settings,
            'evaluation_id': evaluation_id, 'encoding_audit': encoding,
            'no_adapter_loaded': True, 'training_started': False})
        for name, path, _ in tasks:
            key = fingerprint({**settings, 'task_name': name, 'task_sha256': sha256(path)})
            destination = output / f'{name}.json'
            result = cached_result(destination, key, output)
            if result is None:
                if model is None:
                    model = load_model(args)
                    model.eval()
                print(f'Base evaluation: {name}, limit={args.eval_limit}', flush=True)
                scored = evaluate_task(adapter, model, tokenizer, name, path, args)
                predictions = scored.pop('predictions', None)
                result = {'cache_key': key, 'task': name, **scored}
                if predictions is not None:
                    rows = read_rows(path)
                    if args.eval_limit is not None:
                        rows = rows[:args.eval_limit]
                    prediction_file = output / 'predictions' / f'{name}.jsonl'
                    write_predictions(prediction_file, rows, predictions)
                    result.update(predictions_file=prediction_file.relative_to(output).as_posix(),
                                  predictions_sha256=sha256(prediction_file))
                result['cache_sha256'] = fingerprint(result)
                atomic_json(destination, result)
            else:
                print(f'Base cache reused: {name}', flush=True)
            results[name] = result
            print(f'{name}: {result["metrics"]}', flush=True)
        primary = [result['metrics'][adapter.primary_metric] for result in results.values()
                   if adapter.primary_metric in result['metrics']]
        if not primary:
            raise ValueError('No selected Base task provides the primary metric')
        summary = {'method': 'base', 'task': adapter.name, 'evaluation_id': evaluation_id,
            'diagnostic_only': args.eval_limit is not None, 'primary_metric_units': 'fraction',
            'primary_metric': adapter.primary_metric, 'macro_average': sum(primary) / len(primary),
            'eval_limit_per_task': args.eval_limit, 'settings': settings,
            'per_task': {name: result['metrics'] for name, result in results.items()},
            'encoding_audit': encoding, 'output': str(output.relative_to(ROOT))}
        atomic_json(output / 'summary.json', summary)
        pointer = 'latest_full.json' if args.eval_limit is None else 'latest_diagnostic.json'
        atomic_json(parent / pointer, summary)
    print(f'Base evaluation completed: {output / "summary.json"}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
