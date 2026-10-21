"""Task-specific scoring. Loading historical algorithm state belongs to the runner."""
import importlib.util
from pathlib import Path
import re
from fractions import Fraction

import torch

from utils.dataset_adapters import read_rows


def validate_evaluator(adapter, tasks):
    if adapter.name in ('commonsense170k', 'metamathqa'):
        from utils.commonsense_eval import ensure_runtime
        ensure_runtime()
    if adapter.name == 'commonsense170k':
        from utils.commonsense_eval import build_task
        for name, path, _ in tasks:
            if name.startswith('native_'):
                build_task(name.removeprefix('native_'), path)
            else:
                for row in read_rows(path):
                    labels = row.get('choice_labels', [])
                    if not labels or row.get('answer') not in labels or not row.get('input_ids'):
                        raise ValueError(f'Invalid Commonsense generation evaluation row: {path}')
    if adapter.name == 'metamathqa':
        math_utilities()


def math_utilities():
    """Use the pinned harness' MATH normalization, not a new symbolic grader."""
    from utils.commonsense_eval import ensure_runtime
    ensure_runtime()
    import lm_eval
    path = Path(lm_eval.__file__).parent / 'tasks/hendrycks_math/utils.py'
    spec = importlib.util.spec_from_file_location('_pinned_math_grading', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def boxed_answer(text):
    """Extract the last box, retaining nested braces (e.g. LaTeX fractions)."""
    matches = list(re.finditer(r'\\(?:boxed|fbox)\s*\{', text))
    if not matches:
        return None
    start = matches[-1].end()
    depth = 1
    for index in range(start, len(text)):
        depth += (text[index] == '{') - (text[index] == '}')
        if depth == 0:
            return text[start:index]
    return None


def numeric_answer(text):
    text = str(text).strip().strip('$').strip().rstrip('.')
    text = text.replace('−', '-').replace(',', '')
    text = re.sub(r'\\(?:dfrac|tfrac)', r'\\frac', text)
    fraction = re.fullmatch(r'(-?)\\frac\{\s*([+-]?\d+)\s*\}\{\s*([+-]?\d+)\s*\}', text)
    if fraction:
        sign, numerator, denominator = fraction.groups()
        text = f'{sign}{numerator}/{denominator}'
    # Parse the entire explicit answer. Never turn 1/2 into the answer 2.
    if not re.fullmatch(r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?(?:\s*/\s*[+-]?\d+)?', text):
        return None
    text = re.sub(r'\s+', '', text)
    try:
        value = Fraction(text)
    except (ValueError, ZeroDivisionError):
        return None
    return str(value) if '/' in text else text


def final_math_answer(text, task, math_utils=None):
    if task == 'gsm8k':
        box = boxed_answer(text)
        if box is not None:
            return numeric_answer(box)
        if '\\boxed' in text or '\\fbox' in text:
            return None  # Incomplete/truncated box, not an intermediate number.
        if '####' in text:
            return numeric_answer(text.rsplit('####', 1)[-1])
        final = re.search(r'(?:the (?:final )?answer is|final answer)\s*:?\s*(.+)', text, re.I)
        if final:
            return numeric_answer(final.group(1))
        numbers = re.findall(r'[+-]?\d[\d,]*(?:\.\d+)?(?:\s*/\s*[+-]?\d+)?', text)
        return numeric_answer(numbers[-1]) if numbers else None
    utils = math_utils or math_utilities()
    boxed = utils.last_boxed_only_string(text)
    if boxed:
        try:
            return utils.remove_boxed(boxed)
        except (AssertionError, IndexError):
            return None
    # No arbitrary whole-reasoning matching or unverified symbolic equivalence.
    match = re.search(r'(?:the (?:final )?answer is|final answer)\s*:?\s*(.+)', text, re.I)
    return match.group(1).strip().rstrip('.') if match else None


def generate_math(model, tokenizer, rows, task, args):
    from transformers import GenerationConfig
    utils = math_utilities() if task == 'math' else None
    predictions = []
    original_padding = tokenizer.padding_side
    tokenizer.padding_side = 'left'
    try:
        for offset in range(0, len(rows), args.eval_batch_size):
            batch = rows[offset:offset + args.eval_batch_size]
            prompts = [tokenizer.apply_chat_template(
                [{'role': 'user', 'content': row['input_ids']}],
                tokenize=True, add_generation_prompt=True, enable_thinking=False) for row in batch]
            if any(len(p) + args.max_new_tokens > args.eval_max_length for p in prompts):
                raise ValueError('Math evaluation prompt + max_new_tokens exceeds eval_max_length; no truncation')
            inputs = tokenizer.pad([{'input_ids': p} for p in prompts], padding=True, return_tensors='pt')
            device = next(model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            model.eval()
            with torch.inference_mode():
                generated = model.generate(**inputs, generation_config=GenerationConfig(
                    do_sample=False, max_new_tokens=args.max_new_tokens,
                    eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id,
                    bos_token_id=tokenizer.bos_token_id, use_cache=True))
            completions = generated[:, inputs['input_ids'].shape[1]:]
            for row, tokens in zip(batch, completions):
                text = tokenizer.decode(tokens, skip_special_tokens=True)
                answer = final_math_answer(text, task, utils)
                target = row['answer']
                if task == 'gsm8k':
                    try:
                        correct = answer is not None and Fraction(answer) == Fraction(str(target).replace(',', ''))
                    except (ValueError, ZeroDivisionError):
                        correct = False
                else:
                    correct = answer is not None and utils.is_equiv(answer, str(target))
                predictions.append({'sample_id': row.get('sample_id'), 'prediction': text,
                    'extracted_answer': answer, 'target': target, 'correct': bool(correct),
                    'hit_max_new_tokens': not bool((tokens == tokenizer.eos_token_id).any())
                        and len(tokens) >= args.max_new_tokens})
    finally:
        tokenizer.padding_side = original_padding
    return {'metrics': {'exact_match': sum(p['correct'] for p in predictions)/len(predictions)},
            'num_samples': len(predictions), 'predictions': predictions}


def commonsense_answer(text, labels):
    """Extract one whole answer token; ambiguous or missing tokens are wrong.

    Prefer the final explicit answer statement so option words occurring in
    reasoning cannot override it. Never use the gold answer for extraction.
    """
    explicit = list(re.finditer(r'(?:the (?:correct |final )?answer is|final answer|answer)\s*:?\s*([^\n]+)', text, re.I))
    candidate = explicit[-1].group(1) if explicit else text
    found = [label for label in labels if re.search(
        r'(?<!\w)' + re.escape(str(label)) + r'(?!\w)', candidate, re.I)]
    return found[0] if len(found) == 1 else None


def generate_commonsense(model, tokenizer, rows, args):
    from transformers import GenerationConfig
    predictions = []
    original_padding = tokenizer.padding_side
    tokenizer.padding_side = 'left'
    limit = getattr(args, 'commonsense_max_new_tokens', 32)
    try:
        for offset in range(0, len(rows), args.eval_batch_size):
            batch = rows[offset:offset + args.eval_batch_size]
            prompts = [tokenizer.apply_chat_template(
                [{'role': 'user', 'content': row['input_ids']}], tokenize=True,
                add_generation_prompt=True, enable_thinking=False) for row in batch]
            if any(len(prompt) + limit > args.eval_max_length for prompt in prompts):
                raise ValueError('Commonsense evaluation prompt + generation exceeds eval_max_length; no truncation')
            inputs = tokenizer.pad([{'input_ids': prompt} for prompt in prompts], padding=True, return_tensors='pt')
            inputs = {key: value.to(next(model.parameters()).device) for key, value in inputs.items()}
            model.eval()
            with torch.inference_mode():
                tokens = model.generate(**inputs, generation_config=GenerationConfig(
                    do_sample=False, max_new_tokens=limit, eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id, bos_token_id=tokenizer.bos_token_id, use_cache=True))
            for row, completion in zip(batch, tokens[:, inputs['input_ids'].shape[1]:]):
                text = tokenizer.decode(completion, skip_special_tokens=True)
                answer = commonsense_answer(text, row['choice_labels'])
                predictions.append({'sample_id': row.get('sample_id'), 'prediction': text,
                    'extracted_answer': answer, 'target': row['answer'], 'correct': answer == row['answer'],
                    'hit_max_new_tokens': not bool((completion == tokenizer.eos_token_id).any()) and len(completion) >= limit})
    finally:
        tokenizer.padding_side = original_padding
    return {'metrics': {'generation_accuracy': sum(row['correct'] for row in predictions) / len(predictions)},
            'num_samples': len(predictions), 'predictions': predictions,
            'protocol': 'commonsense_chat_generation_v1'}


def evaluate_task(adapter, model, tokenizer, name, path, args):
    if adapter.name == 'commonsense170k' and name.startswith('native_'):
        name = name.removeprefix('native_')
        from utils.commonsense_eval import ensure_runtime, build_task, json_safe
        HFLM = ensure_runtime()
        from lm_eval import evaluator
        from lm_eval.tasks import TaskManager
        model.eval()
        wrapper = HFLM(pretrained=model.get_base_model() if hasattr(model, 'get_base_model') else model,
            tokenizer=tokenizer, backend='causal', batch_size=args.eval_batch_size,
            max_length=args.eval_max_length, device=str(next(model.parameters()).device))
        result = evaluator.simple_evaluate(model=wrapper, tasks=[build_task(name, path)], num_fewshot=0,
            limit=args.eval_limit, bootstrap_iters=0, log_samples=True, apply_chat_template=False,
            task_manager=TaskManager(include_defaults=False), random_seed=args.seed,
            numpy_random_seed=args.seed, torch_random_seed=args.seed, fewshot_random_seed=args.seed)
        return json_safe({'metrics': result['results'][f'cs170k_{name}'], 'harness_result': result})
    rows = read_rows(path)
    if args.eval_limit is not None:
        rows = rows[:args.eval_limit]
    if not rows:
        raise ValueError('Empty evaluation task')
    if adapter.name == 'commonsense170k':
        return generate_commonsense(model, tokenizer, rows, args)
    if adapter.name == 'zsre':
        from eval_zsre import generate_and_score
        metrics, predictions = generate_and_score(model, tokenizer, rows, args.eval_batch_size,
                                                 max_new_tokens=args.max_new_tokens)
        # The shared evaluation format consistently uses fractions.
        for key in ('em', 'f1', 'containment', 'eos_rate', 'max_tokens_rate'):
            metrics[key] /= 100.0
        return {'metrics': metrics, 'num_samples': len(rows), 'predictions': predictions}
    return generate_math(model, tokenizer, rows, name.rsplit('_', 1)[-1], args)
