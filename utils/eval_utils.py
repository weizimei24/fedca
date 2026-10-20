import math
import os
import re
import string
from collections import Counter

import torch
import yaml
from torch.amp import autocast
from torch.utils.data import DataLoader
from utils.model_utils import load_tokenizer


def _load_eval_config():
    config_path = os.path.join(os.path.dirname(__file__), 'eval.yaml')
    # Explicit UTF-8 makes evaluation portable on Windows systems whose
    # process default encoding is GBK.
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


_EVAL_CONFIG = _load_eval_config()


def get_dataset_config(dataset_name: str) -> dict:
    """Return the eval config entry for a given dataset name."""
    datasets = _EVAL_CONFIG.get('datasets', {})
    if dataset_name not in datasets:
        raise ValueError(
            f"Dataset '{dataset_name}' not found in eval.yaml. "
            f"Available: {list(datasets.keys())}"
        )
    return datasets[dataset_name]


class Evaluator:
    """
    Lightweight per-round evaluator: loss and perplexity only (CAUSAL_LM),
    or accuracy (SEQ_CLS).  Instantiated once per client and reused across rounds.
    For full-metric evaluation after training, use eval.py.
    """

    def __init__(self, args, dataset):
        self.args = args
        self.dataset_cfg = get_dataset_config(args.dataset)
        self.task_type = self.dataset_cfg['task_type']
        self.tokenizer = load_tokenizer(args)

        # Evaluation always covers every row actually present in the test
        # split. Dataset size belongs to the data, not to eval.yaml.
        self.eval_loader = DataLoader(
            dataset['test'], batch_size=1, shuffle=False
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def evaluate(self, model, round_idx=None, client_id=None) -> dict:
        model.eval()
        if self.task_type == 'SEQ_CLS':
            result = self._eval_seq_cls(model)
        elif self.task_type == 'CAUSAL_LM':
            result = self._eval_causal_lm(model)
        else:
            raise ValueError(f"Unsupported task_type: {self.task_type}")

        self._log(result, round_idx, client_id)
        return result

    # ------------------------------------------------------------------
    # Per-task implementations
    # ------------------------------------------------------------------

    def _eval_seq_cls(self, model) -> dict:
        correct = 0
        total = 0
        for batch in self.eval_loader:
            input_ids = torch.stack(batch['input_ids']).transpose(0, 1).to(model.device)
            attention_mask = torch.stack(batch['attention_mask']).transpose(0, 1).to(model.device)
            labels = torch.tensor(batch['labels']).to(model.device)
            with torch.no_grad(), autocast('cuda'):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                preds = torch.argmax(outputs.logits, dim=-1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        accuracy = correct / total if total > 0 else 0.0
        return {'accuracy': accuracy}

    def _eval_causal_lm(self, model) -> dict:
        total_nll = 0.0
        total_tokens = 0
        for batch in self.eval_loader:
            input_ids = torch.stack(batch['input_ids']).transpose(0, 1).to(model.device)
            attention_mask = torch.stack(batch['attention_mask']).transpose(0, 1).to(model.device)
            labels = torch.stack(batch['labels']).transpose(0, 1).to(model.device)
            with torch.no_grad(), autocast('cuda'):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                valid_tokens = int((labels[:, 1:] != -100).sum().item())
                total_nll += outputs.loss.item() * valid_tokens
                total_tokens += valid_tokens

        avg_loss = total_nll / total_tokens if total_tokens > 0 else 0.0
        perplexity = math.exp(avg_loss) if avg_loss < 20 else float('inf')
        return {'eval_loss': avg_loss, 'perplexity': perplexity}

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _log(self, metrics: dict, round_idx, client_id):
        prefix = ""
        if round_idx is not None:
            prefix += f"Round {round_idx} | "
        if client_id is not None:
            prefix += f"Client {client_id} | "
        elif round_idx is not None:
            prefix += "Global | "
        parts = [f"{k}: {v:.4f}" for k, v in metrics.items()]
        print(prefix + " | ".join(parts))


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def normalize_answer(text: str) -> str:
    """Official SQuAD v1.1 answer normalization."""
    def remove_articles(value: str) -> str:
        return re.sub(r'\b(a|an|the)\b', ' ', value)

    def remove_punctuation(value: str) -> str:
        return ''.join(character for character in value if character not in string.punctuation)

    def white_space_fix(value: str) -> str:
        return ' '.join(value.split())

    return white_space_fix(remove_articles(remove_punctuation(text.lower())))


def exact_match_score(prediction: str, ground_truth: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def f1_score(prediction: str, ground_truth: str) -> float:
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    if not prediction_tokens or not ground_truth_tokens:
        return float(prediction_tokens == ground_truth_tokens)
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(prediction_tokens)
    recall = num_same / len(ground_truth_tokens)
    return 2 * precision * recall / (precision + recall)


def metric_max_over_ground_truths(metric_fn, prediction: str, ground_truths: list[str]) -> float:
    return max((metric_fn(prediction, truth) for truth in ground_truths), default=0.0)


# Compatibility alias used by the other dataset evaluators.
_normalize = normalize_answer
