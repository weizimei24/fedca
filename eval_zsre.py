"""Final, generation-based zsRE evaluation for the configured method.

The script always reads every ``dataset/zsre/test/<client>.jsonl`` record and
uses the identical greedy generation configuration for every method. It is
separate from MRQA because zsRE has no context field and local-only needs one
adapter per client.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from peft import get_peft_model
from transformers import GenerationConfig

from utils.eval_utils import exact_match_score, f1_score, normalize_answer
from utils.model_utils import load_lora_config, load_model, load_tokenizer
from utils.options import build_parser
from utils.seed_utils import set_global_seed
from utils.zsre_utils import ZSRE_MAX_NEW_TOKENS, ZSRE_PROTOCOL, ZSRE_INSTRUCTION, encode_zsre_prompt


def read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def containment_score(prediction: str, target: str) -> float:
    prediction, target = normalize_answer(prediction).split(), normalize_answer(target).split()
    return float(bool(target) and any(
        prediction[i:i + len(target)] == target
        for i in range(len(prediction) - len(target) + 1)
    ))


def adapter_path(args) -> Path:
    name = f"{args.alg}_{args.dataset}_{args.model}_{args.cn}c_{args.epoch}E_lr{args.lr}"
    return Path(args.suffix) / "adapter" / name


def load_adapter(model, path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Missing adapter: {path}")
    model.load_state_dict(torch.load(path, map_location="cpu"), strict=False)


def load_merged_base_weights(model, path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Missing merged base weights: {path}")
    merged = torch.load(path, map_location="cpu", weights_only=True)
    model_state = model.state_dict()
    if not merged or any(key not in model_state or model_state[key].shape != value.shape
                         for key, value in merged.items()):
        raise ValueError("Merged base weights do not match the evaluation model")
    model.load_state_dict(merged, strict=False)


def generation_metadata(tokenizer, *, max_new_tokens=ZSRE_MAX_NEW_TOKENS) -> dict:
    return {
        "protocol": ZSRE_PROTOCOL, "do_sample": False,
        "max_new_tokens": max_new_tokens, "chat_template": True,
        "enable_thinking": False, "instruction": ZSRE_INSTRUCTION,
        "eos_token_id": tokenizer.eos_token_id,
        "containment": "whole-token target span in prediction",
    }


def generate_and_score(model, tokenizer, rows: list[dict], batch_size: int, *,
                       max_new_tokens: int = ZSRE_MAX_NEW_TOKENS) -> tuple[dict, list[dict]]:
    if batch_size <= 0 or max_new_tokens <= 0:
        raise ValueError("batch_size and max_new_tokens must be positive")
    original_padding_side = tokenizer.padding_side
    try:
        tokenizer.padding_side = "left"
        generation_config = GenerationConfig(
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            bos_token_id=tokenizer.bos_token_id,
        )
        predictions: list[dict] = []
        model.eval()
        for start in range(0, len(rows), batch_size):
            batch = rows[start:start + batch_size]
            prompts = [encode_zsre_prompt(tokenizer, row['input_ids']) for row in batch]
            encoded = tokenizer.pad({"input_ids": prompts, "attention_mask": [[1] * len(p) for p in prompts]},
                                    padding=True, return_tensors="pt").to(model.device)
            with torch.no_grad():
                generated = model.generate(**encoded, generation_config=generation_config, use_model_defaults=False)
            prompt_length = encoded["input_ids"].shape[1]
            decoded = tokenizer.batch_decode(generated[:, prompt_length:], skip_special_tokens=True)
            for row, prediction, tokens in zip(batch, decoded, generated[:, prompt_length:].tolist()):
                eos_reached = tokenizer.eos_token_id in tokens
                token_count = tokens.index(tokenizer.eos_token_id) + 1 if eos_reached else len(tokens)
                prediction = prediction.strip()
                target = str(row["label"])
                predictions.append({
                    "sample_id": row.get("sample_id"),
                    "client_id": row.get("client_id"),
                    "prompt": row["input_ids"],
                    "target": target,
                    "prediction": prediction,
                    "em": exact_match_score(prediction, target),
                    "f1": f1_score(prediction, target),
                    "containment": containment_score(prediction, target),
                    "generated_tokens": token_count,
                    "eos_reached": eos_reached,
                    "hit_max_new_tokens": not eos_reached and token_count >= max_new_tokens,
                })
        if not predictions:
            raise ValueError("zsRE test file is empty")
        n = len(predictions)
        return {
            "em": 100.0 * sum(row["em"] for row in predictions) / n,
            "f1": 100.0 * sum(row["f1"] for row in predictions) / n,
            "containment": 100.0 * sum(row["containment"] for row in predictions) / n,
            "num_samples": n,
            "average_generated_tokens": sum(row["generated_tokens"] for row in predictions) / n,
            "eos_rate": 100.0 * sum(row["eos_reached"] for row in predictions) / n,
            "max_tokens_rate": 100.0 * sum(row["hit_max_new_tokens"] for row in predictions) / n,
        }, predictions
    finally:
        tokenizer.padding_side = original_padding_side


def main() -> int:
    parser = build_parser(max_new_tokens_default=ZSRE_MAX_NEW_TOKENS)
    parser.add_argument("--base-only", action="store_true", help="evaluate the unmodified pretrained model")
    parser.add_argument("--method", choices=("base", "local", "centralized", "fedit", "fedrotlora", "rolora", "flora", "fedexlora", "ffalora", "flexlora", "ilora", "fedfh", "fedca", "fedsvd", "fedmomentum"), default=None)
    parser.add_argument("--data-dir", default="dataset/zsre")
    parser.add_argument("--num-clients", type=int, default=8)
    parser.add_argument(
        "--zsre-eval-batch-size", type=int, default=None,
        help="override the configured evaluation batch size for zsRE generation",
    )
    args = parser.parse_args()
    if args.dataset != "zsre":
        raise ValueError("eval_zsre.py requires dataset: zsre")
    if args.base_only and args.method not in (None, "base"):
        raise ValueError("--base-only can only be used with --method base")
    method = args.method or ("base" if args.base_only else ("local" if args.alg == "localonly" else args.alg))
    if method == "base" and not args.base_only:
        raise ValueError("Base evaluation requires --base-only")

    set_global_seed(args.seed, device=args.device, deterministic=args.deterministic)
    args.suffix = f"exp/{args.suffix}"
    tokenizer = load_tokenizer(args)
    model = load_model(args)
    if not args.base_only:
        model = get_peft_model(model, load_lora_config(args))

    root = Path(args.data_dir)
    output_dir = Path(args.suffix) / "evaluation" / "zsre"
    predictions_dir = output_dir / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    per_client = {}
    root_adapter = adapter_path(args)
    if args.alg in ("flora", "fedexlora", "ilora", "fedmomentum") and not args.base_only:
        load_merged_base_weights(model, root_adapter / "merged_weights.pt")
    for client_id in range(args.num_clients):
        rows = read_jsonl(root / "test" / f"{client_id}.jsonl")
        if args.alg == "localonly" and not args.base_only:
            load_adapter(model, root_adapter / f"client_{client_id}.pt")
        elif not args.base_only and client_id == 0:
            load_adapter(model, root_adapter / "lora_weights.pt")
        batch_size = args.zsre_eval_batch_size or args.eval_batch_size
        metrics, predictions = generate_and_score(model, tokenizer, rows, batch_size,
                                                 max_new_tokens=args.max_new_tokens)
        per_client[str(client_id)] = metrics
        with (predictions_dir / f"client_{client_id}.jsonl").open("w", encoding="utf-8") as handle:
            for row in predictions:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"{method} client {client_id}: EM={metrics['em']:.2f} F1={metrics['f1']:.2f} containment={metrics['containment']:.2f} n={metrics['num_samples']}")
    keys = ("em", "f1", "containment", "num_samples", "average_generated_tokens", "eos_rate", "max_tokens_rate")
    macro = {key: sum(per_client[str(i)][key] for i in range(args.num_clients)) / args.num_clients for key in keys}
    payload = {
        "method": method,
        "generation": generation_metadata(tokenizer, max_new_tokens=args.max_new_tokens),
        "per_client": per_client,
        "macro_average": macro,
    }
    (output_dir / "metrics.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"{method} macro: EM={macro['em']:.2f} F1={macro['f1']:.2f} containment={macro['containment']:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
