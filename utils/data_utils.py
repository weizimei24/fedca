import os

from datasets import load_dataset
from utils.model_utils import load_tokenizer
from utils.qa_utils import encode_squad_training_example
from utils.zsre_utils import encode_zsre_training_example


DATASET_CACHE_DIR = os.path.join('dataset', '.hf_cache')

def load_data(args, idx):
    if getattr(args, '_shared', False):
        from utils.dataset_adapters import load_client_data
        return load_client_data(args, idx)
    if getattr(args, 'commonsense', False):
        from utils.commonsense_framework import load_client_data
        return load_client_data(args, idx)
    dataset = args.dataset
    train_filename = getattr(args, 'train_file', '') or f'{idx}.jsonl'
    train_dir = os.path.join('dataset', dataset, 'train', train_filename)
    data_files = {'train': train_dir}
    # Cross-domain QA keeps named server-side test files.  They are
    # deliberately not exposed as client-indexed ``test/<id>.jsonl`` files;
    # the cross-domain evaluator reads and reports every domain separately.
    if not getattr(args, 'global_test', False) and not getattr(args, 'cross_domain_qa', False):
        data_files['test'] = os.path.join('dataset', dataset, f'test/{idx}.jsonl')

    dataset = load_dataset(
        "json", data_files=data_files, cache_dir=DATASET_CACHE_DIR
    )
    tokenizer = load_tokenizer(args)
    format_func = get_format_func(args, tokenizer)
    # Retain only model tensors after formatting.  In particular, SQuAD's
    # ``answers`` is a ragged list of reference strings and PyTorch's default
    # collator cannot batch it.  Raw columns remain available separately to
    # ``eval.py`` for multi-reference EM/F1 scoring.
    dataset['train'] = dataset['train'].map(
        format_func, remove_columns=dataset['train'].column_names
    )
    if 'test' in dataset:
        dataset['test'] = dataset['test'].map(
            format_func, remove_columns=dataset['test'].column_names
        )

    return dataset


def load_global_test_data(args):
    test_path = os.path.join(
        'dataset', args.dataset, 'test', args.global_test_file
    )
    dataset = load_dataset(
        'json', data_files={'test': test_path}, cache_dir=DATASET_CACHE_DIR
    )
    tokenizer = load_tokenizer(args)
    format_func = get_format_func(args, tokenizer)
    dataset['test'] = dataset['test'].map(
        format_func, remove_columns=dataset['test'].column_names
    )
    return dataset

def get_format_func(args, tokenizer):
    if args.task_type == 'SEQ_CLS':
        def _format_classification(example):
            tokenized = tokenizer(
                    example["input_ids"],
                    return_tensors="pt",
                    truncation=True,
                    padding="max_length",
                    max_length=512
                )
            return {
                "input_ids": tokenized.input_ids[0].tolist(),
                "attention_mask": tokenized.attention_mask[0].tolist(),
                "labels": example["label"]
            }
        return _format_classification
    elif args.task_type == 'CAUSAL_LM':
        if args.dataset == 'zsre':
            def _format_zsre(example):
                tokenizer.padding_side = 'right'
                return encode_zsre_training_example(tokenizer, example)
            return _format_zsre

        if args.dataset.startswith('squad_v1') or getattr(args, 'cross_domain_qa', False):
            def _format_squad(example):
                tokenizer.padding_side = 'right'
                return encode_squad_training_example(tokenizer, example)
            return _format_squad

        def _format_QA(example):
            prompt = f"Instruct: {example['input_ids']}\nAnswer:"
            # Build prompt and answer separately so long SQuAD contexts cannot
            # truncate every supervised answer token at the sequence limit.
            tokenizer.padding_side = 'right'
            prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
            answer_ids = tokenizer(str(example["label"]), add_special_tokens=False)["input_ids"]
            if len(prompt_ids) >= 512:
                # Preserve room for at least one target token even for a long prompt.
                prompt_ids = prompt_ids[:511]
            answer_ids = answer_ids[: 512 - len(prompt_ids)]
            input_ids = prompt_ids + answer_ids
            attention_mask = [1] * len(input_ids)
            prompt_len = len(prompt_ids)
            pad_id = tokenizer.pad_token_id
            padding = 512 - len(input_ids)
            input_ids += [pad_id] * padding
            attention_mask += [0] * padding
            labels = [-100] * prompt_len + answer_ids + [-100] * padding
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels
            }
        return _format_QA
