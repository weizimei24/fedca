"""Building blocks for FedCA (compatibility-aware absorption of client LoRA updates).

Three pieces, matching the method's three steps:

1. Protection signal from the global model.  The global adapter the client
   downloads is the only legitimate carrier of other clients' knowledge.  A
   client's *necessary* change is to raise its own answer token; everything
   else it changes in the model's answer-position distribution is *excess*.
   Excess is measured as not-true KL: KL between the global and the current
   model after the gold token is removed from both distributions.

2. Compatibility-constrained local learning uses that KL as a regulariser
   (``not_true_kl``) with the global model as a frozen teacher
   (``LoraSwap`` swaps the global adapter in for the teacher pass).

3. Compatibility-aware absorption.  The server moves the global adapter along
   the weighted mean factor update by a step ``eta``.  ``probe_measure`` lets
   each client score candidate steps on a fixed probe subset of its *training*
   data (answer CE and excess KL); ``select_eta`` picks the largest useful
   step whose excess stays within the clients' own local excess.

No function here reads test data.
"""

from __future__ import annotations

import math
from contextlib import contextmanager, nullcontext
from typing import Iterable, Mapping, Sequence

import torch
import torch.nn.functional as F

_MASK_VALUE = -1.0e9  # finite, so 0 * (masked - masked) stays 0 instead of NaN


# ---------------------------------------------------------------------------
# LoRA state handling
# ---------------------------------------------------------------------------

def lora_state(model) -> dict[str, torch.Tensor]:
    """Detached copy of every LoRA tensor in the model's state dict."""
    return {k: v.detach().clone() for k, v in model.state_dict().items() if "lora_" in k}


def lora_parameters(model) -> dict[str, torch.nn.Parameter]:
    """Map state-dict key -> live parameter for every LoRA tensor."""
    params = {name: p for name, p in model.named_parameters() if "lora_" in name}
    keys = {k for k in model.state_dict() if "lora_" in k}
    if set(params) != keys:
        missing = sorted(keys ^ set(params))[:3]
        raise KeyError(f"LoRA parameter/state key mismatch, e.g. {missing}")
    return params


@torch.no_grad()
def assign_lora(params: Mapping[str, torch.nn.Parameter], state: Mapping[str, torch.Tensor]) -> None:
    for key, param in params.items():
        param.copy_(state[key].to(device=param.device, dtype=param.dtype))


class LoraSwap:
    """Temporarily replace the trainable LoRA values, then restore them exactly.

    Used for the teacher pass inside local training: the optimizer keeps its
    references to the same Parameter objects; only their data is swapped.
    """

    def __init__(self, model):
        self.params = lora_parameters(model)

    @contextmanager
    def use(self, state: Mapping[str, torch.Tensor]):
        saved = {k: p.detach().clone() for k, p in self.params.items()}
        try:
            assign_lora(self.params, state)
            yield
        finally:
            assign_lora(self.params, saved)


# ---------------------------------------------------------------------------
# Losses on answer positions
# ---------------------------------------------------------------------------

def valid_length(attention_mask: torch.Tensor) -> int:
    """Length after stripping right padding (training data is right padded)."""
    lengths = attention_mask.long().sum(dim=1)
    return max(int(lengths.max()), 2)


def _answer_view(logits: torch.Tensor, labels: torch.Tensor):
    """Logits that predict a supervised token, with the token they predict."""
    shifted_logits = logits[:, :-1, :]
    shifted_labels = labels[:, 1:]
    mask = shifted_labels.ne(-100)
    return shifted_logits[mask].float(), shifted_labels[mask]


def not_true_kl(student_logits: torch.Tensor, teacher_logits: torch.Tensor,
                labels: torch.Tensor, tau: float = 1.0, reduction: str = "mean", compact: bool = False) -> torch.Tensor:
    """KL(teacher || student) over non-gold tokens at every supervised position.

    Both distributions drop the gold token and renormalise, so raising the
    gold answer (the client's necessary change) costs nothing; reshaping the
    rest of the answer distribution (what the global model would otherwise
    say) is penalised.  Scaled by tau^2 as in standard distillation.
    """
    if student_logits.shape != teacher_logits.shape:
        raise ValueError("student/teacher logits differ in shape")
    if compact:
        # Commonsense can have >1000 prompt tokens but only a few supervised
        # answer tokens. Slicing before the vocabulary softmax preserves the
        # same KL and avoids materializing prompt-length float32 distributions.
        # Basic slices + stack keep backward deterministic (no index_put).
        positions = labels[:, 1:].ne(-100).nonzero(as_tuple=False).tolist()
        if not positions:
            return student_logits.sum() * 0 if reduction != 'none' else student_logits.new_empty(0)
        s = torch.stack([student_logits[b, i] for b, i in positions]).float()
        t = torch.stack([teacher_logits[b, i] for b, i in positions]).float().detach()
        target = torch.stack([labels[b, i+1] for b, i in positions])
        gold = torch.arange(s.shape[-1], device=s.device).unsqueeze(0).eq(target.unsqueeze(1))
        log_q = F.log_softmax(s.masked_fill(gold, _MASK_VALUE) / tau, dim=-1)
        log_p = F.log_softmax(t.masked_fill(gold, _MASK_VALUE) / tau, dim=-1)
        per_position = (log_p.exp() * (log_p - log_q)).sum(dim=-1) * (tau * tau)
        return per_position if reduction == 'none' else per_position.sum() if reduction == 'sum' else per_position.mean()
    # Mask arithmetic instead of boolean indexing: no index_put in backward,
    # so the loss stays valid under torch.use_deterministic_algorithms(True).
    s = student_logits[:, :-1, :].float()
    t = teacher_logits[:, :-1, :].float().detach()
    target = labels[:, 1:]
    mask = target.ne(-100)
    vocab = torch.arange(s.shape[-1], device=s.device)
    gold = vocab.view(1, 1, -1).eq(target.clamp(min=0).unsqueeze(-1))
    log_q = F.log_softmax(s.masked_fill(gold, _MASK_VALUE) / tau, dim=-1)
    log_p = F.log_softmax(t.masked_fill(gold, _MASK_VALUE) / tau, dim=-1)
    per_position = (log_p.exp() * (log_p - log_q)).sum(dim=-1) * (tau * tau)
    per_position = per_position * mask
    if reduction == "none":
        return per_position[mask]
    total = per_position.sum()
    if reduction == "sum":
        return total
    return total / mask.sum().clamp(min=1)


def answer_ce(logits: torch.Tensor, labels: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
    s, target = _answer_view(logits, labels)
    return F.cross_entropy(s, target, reduction=reduction)


# ---------------------------------------------------------------------------
# Probe evaluation (client side; training data only)
# ---------------------------------------------------------------------------

def select_probe_indices(num_rows: int, size: int, seed: int) -> list[int]:
    """Fixed, seeded subset of a client's *training* rows."""
    if size <= 0:
        return []
    generator = torch.Generator().manual_seed(int(seed))
    order = torch.randperm(num_rows, generator=generator).tolist()
    return sorted(order[:min(size, num_rows)])


def _autocast(device: torch.device):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def _batches(rows, indices: Sequence[int], batch_size: int, device, dynamic_padding=False, pad_token_id=0):
    for start in range(0, len(indices), batch_size):
        chunk = rows[list(indices[start:start + batch_size])]
        if dynamic_padding:
            from utils.runtime_utils import collate
            records = [{'input_ids': ids, 'labels': labels} for ids, labels in zip(chunk['input_ids'], chunk['labels'])]
            batch = collate(records, pad_id=pad_token_id, device=device)
            yield batch['input_ids'], batch['attention_mask'], batch['labels']
            continue
        input_ids = torch.tensor(chunk["input_ids"], device=device)
        attention_mask = torch.tensor(chunk["attention_mask"], device=device)
        labels = torch.tensor(chunk["labels"], device=device)
        length = valid_length(attention_mask)
        yield input_ids[:, :length], attention_mask[:, :length], labels[:, :length]


@torch.no_grad()
def probe_measure(model, rows, indices: Sequence[int], teacher_state: Mapping[str, torch.Tensor],
                  candidates: Mapping[str, Mapping[str, torch.Tensor]], batch_size: int = 8,
                  tau: float = 1.0, dynamic_padding: bool = False, pad_token_id: int = 0) -> dict:
    """Answer CE and excess (not-true KL to the teacher) for candidate adapters.

    Returns token-averaged values: ``{"teacher_ce": float, "ce": {name: float},
    "excess": {name: float}, "tokens": int}``.  The model's LoRA values and
    train/eval mode are restored before returning.
    """
    params = lora_parameters(model)
    original = {k: p.detach().clone() for k, p in params.items()}
    was_training = model.training
    device = next(model.parameters()).device
    totals = {"teacher_ce": 0.0, "ce": {n: 0.0 for n in candidates}, "excess": {n: 0.0 for n in candidates}}
    tokens = 0
    model.eval()
    try:
        forward_options = {'use_cache': False} if dynamic_padding else {}
        for input_ids, attention_mask, labels in _batches(rows, indices, batch_size, device, dynamic_padding, pad_token_id):
            assign_lora(params, teacher_state)
            with _autocast(device):
                teacher_logits = model(input_ids=input_ids, attention_mask=attention_mask, **forward_options).logits
            count = int(labels[:, 1:].ne(-100).sum())
            tokens += count
            totals["teacher_ce"] += float(answer_ce(teacher_logits, labels, reduction="sum"))
            for name, state in candidates.items():
                assign_lora(params, state)
                with _autocast(device):
                    logits = model(input_ids=input_ids, attention_mask=attention_mask, **forward_options).logits
                totals["ce"][name] += float(answer_ce(logits, labels, reduction="sum"))
                totals["excess"][name] += float(not_true_kl(logits, teacher_logits, labels, tau, reduction="sum", compact=dynamic_padding))
    finally:
        assign_lora(params, original)
        model.train(was_training)
    if tokens == 0:
        raise ValueError("probe subset has no supervised tokens")
    return {
        "teacher_ce": totals["teacher_ce"] / tokens,
        "ce": {n: v / tokens for n, v in totals["ce"].items()},
        "excess": {n: v / tokens for n, v in totals["excess"].items()},
        "tokens": tokens,
    }


# ---------------------------------------------------------------------------
# Server-side absorption
# ---------------------------------------------------------------------------

def weighted_direction(global_state: Mapping[str, torch.Tensor],
                       client_states: Sequence[Mapping[str, torch.Tensor]],
                       weights: Sequence[float]) -> dict[str, torch.Tensor]:
    """sum_i w_i (theta_i - theta_g) in float32, factor by factor."""
    if abs(sum(weights) - 1.0) > 1e-6:
        raise ValueError("absorption weights must sum to 1")
    out = {}
    for key, g in global_state.items():
        acc = torch.zeros_like(g, dtype=torch.float32)
        for w, state in zip(weights, client_states):
            acc += float(w) * (state[key].float() - g.float())
        out[key] = acc
    return out


def absorb(global_state: Mapping[str, torch.Tensor],
           client_states: Sequence[Mapping[str, torch.Tensor]],
           weights: Sequence[float], eta: float) -> dict[str, torch.Tensor]:
    """theta_g + eta * sum_i w_i (theta_i - theta_g), per LoRA factor.

    For eta == 1 this is computed exactly as FedIT computes its factor
    average (same accumulation order), so the eta = 1, lambda = 0 setting
    reproduces FedIT bit for bit.
    """
    out = {}
    for key, g in global_state.items():
        mean = 0
        for w, state in zip(weights, client_states):
            mean = mean + state[key] * w
        if eta == 1.0:
            out[key] = mean
        else:
            out[key] = (g.float() + eta * (mean.float() - g.float())).to(g.dtype)
    return out


def fedexp_eta(global_state, client_states, weights, eps: float = 1e-12) -> float:
    """FedExP server step (Jhunjhunwala et al., ICLR 2023) in LoRA factor space:
    max(1, sum_i w_i ||d_i||^2 / (2 ||sum_i w_i d_i||^2))."""
    individual = 0.0
    mean_sq = 0.0
    for key, g in global_state.items():
        deltas = [state[key].double() - g.double() for state in client_states]
        individual += sum(float(w) * float(d.pow(2).sum()) for w, d in zip(weights, deltas))
        mean = sum(float(w) * d for w, d in zip(weights, deltas))
        mean_sq += float(mean.pow(2).sum())
    return max(1.0, individual / (2.0 * mean_sq + eps))


def compatibility_weights(base_weights: Sequence[float], excess: Sequence[float], beta: float) -> list[float]:
    """Optional per-client tilt: w_i ∝ p_i * (mean_excess / excess_i)^beta.  beta = 0 -> p_i."""
    if beta == 0:
        return [float(w) for w in base_weights]
    mean_excess = sum(excess) / len(excess)
    raw = [p * (mean_excess / max(e, 1e-12)) ** beta for p, e in zip(base_weights, excess)]
    total = sum(raw)
    return [r / total for r in raw]


def select_eta(grid: Sequence[float], gain: Mapping[float, float], excess: Mapping[float, float],
               budget: float) -> tuple[float, bool]:
    """Largest-gain step whose excess fits the budget; ties go to the smaller step.

    Returns (eta, budget_bound).  If no step is feasible the smallest grid step
    is used, so the method never absorbs more than the safest candidate.
    """
    ordered = sorted(float(e) for e in grid)
    feasible = [e for e in ordered if excess[e] <= budget]
    if not feasible:
        return ordered[0], True
    best = max(feasible, key=lambda e: (gain[e], -e))
    bound = any(gain[e] > gain[best] for e in ordered if e not in feasible)
    return best, bound


def parse_grid(value) -> list[float]:
    if isinstance(value, (list, tuple)):
        grid = [float(v) for v in value]
    else:
        grid = [float(v) for v in str(value).replace(" ", "").split(",") if v]
    if not grid or any((not math.isfinite(v)) or v <= 0 for v in grid):
        raise ValueError(f"invalid eta grid: {value!r}")
    return sorted(set(grid))


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values)
