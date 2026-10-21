"""Dataset-independent persistence and model-state primitives.

Optimization remains owned by utils.train_utils.Trainer.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def source_sha256(path):
    """Equivalent Python source survives Windows/Linux checkout line endings."""
    return hashlib.sha256(Path(path).read_text(encoding='utf-8-sig').encode('utf-8')).hexdigest()


def replace_file(source, target):
    # Windows indexers occasionally hold a newly written file open briefly.
    for attempt in range(10):
        try:
            os.replace(source, target)
            return
        except PermissionError:
            if attempt == 9:
                raise
            time.sleep(min(0.1 * 2 ** attempt, 1))


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w", encoding="utf-8") as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    replace_file(temp, path)


def atomic_torch(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("wb") as f:
        torch.save(value, f)
        f.flush()
        os.fsync(f.fileno())
    replace_file(temp, path)


@contextlib.contextmanager
def run_lock(directory):
    """An OS lock releases on process exit; no stale PID-file bypass is needed."""
    path = Path(directory) / "run.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as f:
        f.seek(0)
        if not f.read(1):
            f.write(b"0")
            f.flush()
        f.seek(0)
        if os.name == "nt":
            import msvcrt
            try:
                msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                raise RuntimeError(f"Another process is using {directory}") from exc
            try:
                yield
            finally:
                f.seek(0)
                msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
            try:
                yield
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)


class CheckpointStore:
    """Two rotating slots; pointer updates only after a complete fsynced save.

    A failed new save leaves the previously committed slot intact. Hash failures
    are explicit errors, never a silent restart or rollback to a different round.
    """
    def __init__(self, directory, identity):
        self.directory = Path(directory)
        self.identity = identity
        self.pointer = self.directory / "latest.json"

    def load(self):
        if not self.pointer.exists():
            return None
        meta = json.loads(self.pointer.read_text(encoding="utf-8"))
        if meta["identity"] != self.identity:
            raise ValueError("Checkpoint settings/data/model/code differ; use a new experiment output directory (--suffix in main.py)")
        path = self.directory / meta["file"]
        if meta["file"] not in ("state_0.pt", "state_1.pt") or sha256(path) != meta["sha256"]:
            raise ValueError("Committed checkpoint is missing or corrupt; refusing silent restart")
        state = torch.load(path, map_location="cpu", weights_only=True)
        if state["identity"] != self.identity:
            raise ValueError("Checkpoint identity mismatch")
        # Immutable large tensors may live in a durable initial/round snapshot.
        # Resolve explicit references only after verifying both byte checksum
        # and experiment identity. Older self-contained checkpoints still load.
        for reference in state.pop('_tensor_references', []):
            root = self.directory.parent.resolve()
            relative = Path(reference['file'])
            source = (root / relative).resolve()
            if relative.is_absolute() or '..' in relative.parts or not source.is_relative_to(root):
                raise ValueError('Invalid immutable checkpoint reference path')
            if not source.is_file() or sha256(source) != reference['sha256']:
                raise ValueError('Immutable checkpoint reference is missing or corrupt')
            blob = torch.load(source, map_location='cpu', weights_only=True)
            if blob.get('identity') != self.identity:
                raise ValueError('Immutable checkpoint reference identity differs')
            value = blob
            for key in reference['source_keys']:
                value = value[key]
            destination = state
            for key in reference['destination_keys'][:-1]:
                destination = destination[key]
            destination[reference['destination_keys'][-1]] = value
        return state

    def save(self, state):
        self.directory.mkdir(parents=True, exist_ok=True)
        old = json.loads(self.pointer.read_text(encoding="utf-8")) if self.pointer.exists() else {}
        generation = old.get("generation", 0) + 1
        name = f"state_{generation % 2}.pt"
        payload = {**state, "identity": self.identity}
        atomic_torch(self.directory / name, payload)
        atomic_json(self.pointer, {"identity": self.identity, "generation": generation,
                                  "file": name, "sha256": sha256(self.directory / name),
                                  "completed_rounds": state["completed_rounds"],
                                  "completed_clients": len(state["pending"]),
                                  "active_optimizer_steps": (state.get("active") or {}).get("updates", 0)})


def cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_tree(v) for k, v in value.items()}
    if isinstance(value, list):
        return [cpu_tree(v) for v in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(v) for v in value)
    return value


def adapter_state(model):
    return {name: p.detach().cpu().clone() for name, p in model.named_parameters() if p.requires_grad}


def load_adapter(model, state):
    params = dict(model.named_parameters())
    expected = {name for name, p in params.items() if p.requires_grad}
    if set(state) != expected:
        raise ValueError("Adapter parameter keys do not match the current model")
    with torch.no_grad():
        for name, value in state.items():
            if value.shape != params[name].shape:
                raise ValueError(f"Adapter shape mismatch: {name}")
            params[name].copy_(value)


def factor_average(states, sizes):
    """FedIT: average A/B factors, in client order and original parameter dtype.

    Matches alg.ftbase.FTBaseServer.aggregate (not the average of B@A deltas).
    """
    total = sum(sizes)
    out = {}
    for state, size in zip(states, sizes):
        for name, value in state.items():
            out[name] = out.get(name, 0) + value * (size / total)
    return out


def evaluation_rounds(args):
    """Common interval/final-round scheduling for shared and legacy callers."""
    if args.eval_rounds:
        rounds = sorted(set(int(value) for value in args.eval_rounds.split(',')))
        if not getattr(args, 'eval_only', False) or not rounds or min(rounds) < 1 or max(rounds) > args.rnd:
            raise ValueError('eval_rounds requires eval_only and rounds within 1..rnd')
        return rounds
    return sorted(set(range(args.test_gap, args.rnd + 1, args.test_gap)) | {args.rnd})


def capture_rng():
    ns = np.random.get_state()
    return {"python": random.getstate(), "numpy": (ns[0], ns[1].tolist(), ns[2], ns[3], ns[4]),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state["python"])
    ns = state["numpy"]
    np.random.set_state((ns[0], np.asarray(ns[1], dtype=np.uint32), *ns[2:]))
    torch.set_rng_state(state["torch"])
    if state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])



def collate(rows, pad_id, device):
    length = max(len(r["input_ids"]) for r in rows)
    return {"input_ids": torch.tensor([r["input_ids"] + [pad_id] * (length-len(r["input_ids"])) for r in rows], device=device),
            "labels": torch.tensor([r["labels"] + [-100] * (length-len(r["labels"])) for r in rows], device=device),
            "attention_mask": torch.tensor([[1] * len(r["input_ids"]) + [0] * (length-len(r["input_ids"])) for r in rows], device=device)}


