"""Fixed-rank federated functional harmonization (FedFH) utilities.

Client side
    ``collect_input_geometry`` runs one gradient-free pass over the client's
    training rows and, for every LoRA target layer, estimates the uncentred
    second moment of the layer input ``Sigma = E[x x^T]`` over valid (non-pad)
    tokens. Only its top-k eigenpairs plus two scalars are kept ("uploaded").
    Modules that read the same tensor (q/k/v projections of one attention
    block, gate/up projections of one MLP) share a single estimate.

Server side
    ``harmonize_layer`` solves, for one LoRA layer,

        min_{rank(M) <= R}  sum_i p_i || (M - B_i A_i) Sigma_i~^{1/2} ||_F^2,
        Sigma_i~ = Q_i Lam_i Q_i^T + lam * I,

    exactly: whitened truncated SVD, then re-gauged to the FlexLoRA factor
    convention (B = U S, A = V^T). As lam -> infinity it reduces to FlexLoRA.
"""

from __future__ import annotations

import math

import torch

QKV_SUFFIXES = ("q_proj", "k_proj", "v_proj")
GATE_UP_SUFFIXES = ("gate_proj", "up_proj")


# ---------------------------------------------------------------------------
# Naming helpers
# ---------------------------------------------------------------------------

def module_name_from_lora_key(key: str) -> str:
    """'...self_attn.q_proj.lora_A.default.weight' -> '...self_attn.q_proj'."""
    for marker in (".lora_A.", ".lora_B."):
        if marker in key:
            return key.split(marker)[0]
    raise KeyError(f"Not a LoRA factor key: {key}")


def geometry_group(module_name: str) -> str:
    """Modules reading the same input tensor share one geometry estimate."""
    parent, _, leaf = module_name.rpartition(".")
    if leaf in QKV_SUFFIXES:
        return f"{parent}.<qkv_input>"
    if leaf in GATE_UP_SUFFIXES:
        return f"{parent}.<gate_up_input>"
    return module_name


def _lora_module_names(model) -> list[str]:
    names = []
    for name, module in model.named_modules():
        if hasattr(module, "lora_A") and hasattr(module, "base_layer"):
            names.append(name)
    if not names:
        raise ValueError("No PEFT LoRA layers found in the model")
    return names


# ---------------------------------------------------------------------------
# Client: input geometry summary (module 1)
# ---------------------------------------------------------------------------

def _batch_tensors(rows, device):
    input_ids = torch.tensor(rows["input_ids"], dtype=torch.long)
    attention_mask = torch.tensor(rows["attention_mask"], dtype=torch.long)
    # Rows are right-padded; trimming to the longest valid row in the batch
    # leaves every valid token and its causal context unchanged.
    valid_len = int(attention_mask.sum(dim=1).max().item())
    return input_ids[:, :valid_len].to(device), attention_mask[:, :valid_len].to(device)


@torch.no_grad()
def collect_input_geometry(model, train_rows, top_k: int, batch_size: int = 16) -> dict:
    """Return {group: {"Q", "Lam", "trace", "tail", "tokens"}} for one client.

    ``train_rows`` is the formatted HF dataset (columns input_ids,
    attention_mask). The model is used in eval mode (no dropout) and its
    training mode is restored afterwards. No RNG is consumed.
    """
    groups: dict[str, str] = {}
    for name in _lora_module_names(model):
        groups.setdefault(geometry_group(name), name)
    modules = dict(model.named_modules())
    device = next(p for p in model.parameters()).device

    accumulators: dict[str, torch.Tensor] = {}
    state = {"mask": None}

    def make_hook(group):
        def hook(_module, inputs):
            x = inputs[0]
            rows = x[state["mask"]].float()          # (valid tokens, n)
            acc = accumulators.get(group)
            if acc is None:
                acc = torch.zeros(rows.shape[1], rows.shape[1], dtype=torch.float32, device=rows.device)
                accumulators[group] = acc
            acc.addmm_(rows.T, rows)
        return hook

    handles = [modules[name].register_forward_pre_hook(make_hook(group)) for group, name in groups.items()]
    was_training = model.training
    model.eval()
    total_tokens = 0
    try:
        for start in range(0, len(train_rows), batch_size):
            input_ids, attention_mask = _batch_tensors(train_rows[start:start + batch_size], device)
            state["mask"] = attention_mask.bool()
            total_tokens += int(attention_mask.sum().item())
            model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    finally:
        for handle in handles:
            handle.remove()
        model.train(was_training)
    if total_tokens == 0:
        raise RuntimeError("Geometry pass saw no valid tokens")

    geometry = {}
    for group, acc in accumulators.items():
        sigma = acc / total_tokens
        sigma = 0.5 * (sigma + sigma.T)
        # fp32 is ample for the top-k eigenpairs and is fast on consumer GPUs
        # (fp64 eigh there is ~30x slower); the trace is accumulated in fp64.
        evals, evecs = torch.linalg.eigh(sigma)           # ascending
        trace = float(torch.diagonal(sigma).double().sum().item())
        evals = evals.flip(0).clamp_min(0.0)
        evecs = evecs.flip(1)
        k = min(top_k, evals.numel())
        geometry[group] = {
            "Q": evecs[:, :k].float().cpu(),
            "Lam": evals[:k].float().cpu(),
            "trace": trace,
            "tail": float(evals[k].item()) if k < evals.numel() else 0.0,
            "tokens": total_tokens,
        }
        del acc
    accumulators.clear()
    return geometry


# ---------------------------------------------------------------------------
# Server: functional harmonization + fixed-rank reconstruction (modules 2, 3)
# ---------------------------------------------------------------------------

def truncated_product_svd(left: torch.Tensor, right: torch.Tensor, rank: int):
    """Rank-R SVD of left @ right without forming it.

    Returns (B, A, sigma) with B = U_R S_R, A = V_R^T (FlexLoRA convention) and
    sigma the full singular spectrum of left @ right.
    """
    qb, rb = torch.linalg.qr(left, mode="reduced")
    qa, ra = torch.linalg.qr(right.T, mode="reduced")
    u, sigma, vh = torch.linalg.svd(rb @ ra.T, full_matrices=False)
    if rank > sigma.numel():
        raise ValueError(f"rank {rank} exceeds product dimension {sigma.numel()}")
    return (qb @ u[:, :rank]) * sigma[:rank], vh[:rank, :] @ qa.T, sigma


def build_mean_geometry(entries, weights, lam: float):
    """Dense Sigma_bar~ = sum p_i Q_i Lam_i Q_i^T + lam I and its inverse sqrt."""
    n = entries[0]["Q"].shape[0]
    device, dtype = entries[0]["Q"].device, entries[0]["Q"].dtype
    sigma_bar = lam * torch.eye(n, device=device, dtype=dtype)
    for p, entry in zip(weights, entries):
        q, lam_i = entry["Q"], entry["Lam"]
        sigma_bar += p * (q * lam_i) @ q.T
    sigma_bar = 0.5 * (sigma_bar + sigma_bar.T)
    evals, evecs = torch.linalg.eigh(sigma_bar)
    if evals.min().item() <= 0:
        raise FloatingPointError("Sigma_bar~ is not positive definite; check lambda")
    inv_sqrt = (evecs * evals.rsqrt()) @ evecs.T
    return inv_sqrt


def harmonize_layer(b_list, a_list, entries, weights, lam: float, rank: int, inv_sqrt: torch.Tensor):
    """Optimal rank-R global LoRA for one layer under the sketched objective.

    All tensors must share device and float64 dtype. Returns (B_g, A_g, diag).
    """
    b_hat = torch.cat(b_list, dim=1)                                  # m x Nr
    a_rows, own_energy = [], 0.0
    for p, b, a, entry in zip(weights, b_list, a_list, entries):
        aq = a @ entry["Q"]                                           # r x k
        a_sigma = (aq * entry["Lam"]) @ entry["Q"].T + lam * a        # A_i Sigma_i~
        a_rows.append(p * a_sigma)
        # tr(M_i Sigma_i~ M_i^T) = tr((B^T B)(A Sigma~ A^T)), small r x r.
        own_energy += p * float(torch.trace((b.T @ b) @ (a_sigma @ a.T)).item())
    a_hat = torch.cat(a_rows, dim=0)                                  # Nr x n

    # [Y~]_R with Y~ = Z~ Sigma_bar~^{-1/2}, then de-whiten and re-gauge.
    b_f, a_f, sigma = truncated_product_svd(b_hat, a_hat @ inv_sqrt, rank)
    b_g, a_g, _ = truncated_product_svd(b_f, a_f @ inv_sqrt, rank)

    y_energy = float((sigma ** 2).sum().item())
    kept_energy = float((sigma[:rank] ** 2).sum().item())
    intrinsic = max(own_energy - y_energy, 0.0)
    rank_loss = y_energy - kept_energy
    # Fixed-rank gap certificate (Prop. 6.2): sum p_i tail_i ||M_R - M_i||_F^2.
    certificate = 0.0
    for p, b, a, entry in zip(weights, b_list, a_list, entries):
        if entry["tail"] > 0:
            left = torch.cat([b_g, -b], dim=1)
            right = torch.cat([a_g, a], dim=0)
            diff_sq = float(torch.trace((left.T @ left) @ (right @ right.T)).item())
            certificate += p * entry["tail"] * diff_sq
    objective = intrinsic + rank_loss                                 # L~(M_R), Theorem C
    diag = {
        "own_energy": own_energy,
        "intrinsic": intrinsic,
        "rank_loss": rank_loss,
        "objective": objective,
        "certificate": certificate,
    }
    return b_g, a_g, diag


def layer_lambda(entries, weights, gamma: float) -> float:
    n = entries[0]["Q"].shape[0]
    tau = sum(p * entry["trace"] for p, entry in zip(weights, entries)) / n
    lam = gamma * tau
    if not math.isfinite(lam) or lam <= 0:
        raise ValueError(f"Invalid lambda {lam} (gamma={gamma}, tau={tau})")
    return lam
