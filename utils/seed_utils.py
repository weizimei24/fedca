"""Reproducibility controls shared by training and evaluation entry points."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_rngs(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def set_global_seed(seed: int, device: int | None = None, deterministic: bool = True) -> None:
    if device is not None and "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device)
    if deterministic:
        # Required by deterministic CUDA matrix multiplication on CUDA >= 10.2.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    seed_rngs(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True)
        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


def client_round_seed(base_seed: int, client_id: int, round_index: int) -> int:
    """Derive a stable RNG seed for one client in one communication round."""
    return int(base_seed) + int(round_index) * 1_000_003 + int(client_id) * 10_007
