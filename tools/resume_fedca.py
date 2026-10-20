"""Continue a FedCA-suite run from its last completed global adapter.

Use through run_fedca_suite.py --resume-training so the original experiment's
flags and the ordinary final/saved-round generation evaluations are retained.
Only full-participation prototype runs are supported: local optimizers are
recreated each round, and local RNG/shuffle seeds depend on client and round.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def find_checkpoint(folder: Path, args) -> tuple[int, Path]:
    records = [json.loads(line) for line in
               (folder / "fedca_rounds.jsonl").read_text(encoding="utf-8").splitlines()
               if line.strip()]
    for completed, record in enumerate(records, 1):
        if (record["completed_rounds"] != completed or record["round"] != completed - 1
                or record["mode"] != args.ca_eta_mode
                or record["lambda"] != args.ca_lambda
                or len(record["weights"]) != args.cn):
            raise ValueError("Round records do not match the requested experiment/configuration")
    checkpoints = sorted((int(p.name[6:]), p / "lora_weights.pt")
                         for p in folder.glob("round_*")
                         if p.is_dir() and p.name[6:].isdigit()
                         and (p / "lora_weights.pt").is_file())
    if not checkpoints:
        raise FileNotFoundError(f"No completed-round checkpoint under {folder}")
    completed, checkpoint = checkpoints[-1]
    if completed != len(records):
        raise ValueError("Latest checkpoint and round log disagree; preserve and inspect the run before retrying")
    if not 0 < completed <= args.rnd:
        raise ValueError(f"Checkpoint round {completed} exceeds requested target {args.rnd}")
    config_path = folder / "training_config.json"
    if config_path.is_file():
        saved = json.loads(config_path.read_text(encoding="utf-8"))
        for key in ("alg", "dataset", "model", "seed", "deterministic", "cn", "sr", "rnd",
                    "bs", "grad_accum", "epoch", "step", "lr", "task_type", "train_file",
                    "lora_rank", "lora_alpha", "lora_dropout", "ca_lambda", "ca_tau",
                    "ca_eta_mode", "ca_eta", "ca_eta_grid", "ca_kappa", "ca_beta",
                    "ca_probe_size", "ca_probe_bs", "ca_probe_diag"):
            if saved.get(key) != getattr(args, key):
                raise ValueError(f"Saved configuration differs for {key}")
    return completed, checkpoint


def restore_adapter(model, checkpoint: Path) -> dict:
    import torch

    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    current = {k: v for k, v in model.state_dict().items() if "lora_" in k}
    if set(saved) != set(current):
        raise ValueError("Checkpoint LoRA keys do not match the model")
    for key, tensor in saved.items():
        if (not isinstance(tensor, torch.Tensor) or tensor.shape != current[key].shape
                or not torch.isfinite(tensor).all()):
            raise ValueError(f"Invalid checkpoint tensor: {key}")
    restored = {k: v.to(device=current[k].device, dtype=current[k].dtype) for k, v in saved.items()}
    model.load_state_dict(restored, strict=False)
    return restored


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n",
                         encoding="utf-8")
    os.replace(temporary, path)


def continue_training(args, server_factory=None) -> None:
    if args.alg != "fedca" or args.dataset != "zsre" or args.mode != "prototype" or args.sr != 1.0:
        raise ValueError("Resume supports only full-participation zsRE FedCA prototype runs")
    if args.ca_save_every != 1:
        raise ValueError("Resume requires ca_save_every=1 to keep round records and checkpoints aligned")
    name = f"{args.alg}_{args.dataset}_{args.model}_{args.cn}c_{args.epoch}E_lr{args.lr}"
    folder = Path(args.suffix) / "adapter" / name
    state_path = Path(args.suffix) / "resume_state.json"
    lock_path = Path(args.suffix) / "resume_training.lock"
    # Never train against a directory already owned by another continuation.
    descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    info = {"status": "initializing", "pid": os.getpid()}
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(str(os.getpid()))
        completed, checkpoint = find_checkpoint(folder, args)
        info.update(source_checkpoint=str(checkpoint),
                    source_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                    source_completed_rounds=completed, target_rounds=args.rnd,
                    config=vars(args),
                    note="Original YAML plus suite overrides; wall clock covers the continuation only.")
        atomic_json(state_path, info)
        if server_factory is None:
            from alg.fedca import Client, Server
            from utils.seed_utils import set_global_seed

            set_global_seed(args.seed, device=args.device, deterministic=args.deterministic)
            clients = [Client(index, args) for index in range(args.cn)]
            set_global_seed(args.seed, device=args.device, deterministic=args.deterministic)
            server = Server(args, clients)
        else:
            server = server_factory(args)
        server.global_lora = restore_adapter(server.model, checkpoint)
        print(f"Loaded {checkpoint}: {completed}/{args.rnd} completed rounds", flush=True)
        for round_index in range(completed, args.rnd):
            server.round = round_index
            info.update(status="training", current_round=round_index + 1,
                        learning_rate=args.lr * (0.99 ** round_index))
            atomic_json(state_path, info)
            print(f"=== Resumed round {round_index + 1}/{args.rnd}; lr={info['learning_rate']:.12g} ===", flush=True)
            started = time.time()
            server.run()
            # Match main.py's per-round loss/perplexity evaluation.
            metrics = server.test_all()
            info.update(completed_rounds=round_index + 1, round_test=metrics,
                        round_elapsed_seconds=time.time() - started)
            atomic_json(state_path, info)
            print(f"Round {round_index + 1}: {metrics}", flush=True)
        # Unlike main.py's finally block, a failed/aborted round must never be
        # promoted to a final adapter that the suite would mistake for complete.
        server.save_adapter()
        info.update(status="completed", completed_rounds=args.rnd)
        atomic_json(state_path, info)
    except BaseException as exc:
        info.update(status="failed", error=repr(exc))
        atomic_json(state_path, info)
        raise
    finally:
        lock_path.unlink(missing_ok=True)


def main() -> int:
    from alg.fedca import add_args
    from utils.options import build_parser

    args = add_args(build_parser())
    args.suffix = f"exp/{args.suffix}"
    continue_training(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
