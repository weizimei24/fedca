"""Generation EM/F1 for saved per-round global adapters (FedCA and its ablations).

Reads <exp>/<suffix>/adapter/<name>/round_XXX/lora_weights.pt ("XXX completed
rounds"), evaluates each requested round on all 8 x 500 zsRE test questions
with the exact protocol of eval_zsre.py, and writes

  <exp>/<suffix>/evaluation/zsre/rounds/round_XXX/metrics.json (+ predictions)
  <exp>/<suffix>/evaluation/zsre/rounds/summary.json

Test data is only read here, after training; nothing feeds back into training.

  python -X utf8 -u tools/eval_zsre_rounds.py --config config.zsre_fedca.yaml \
      --suffix zsre8_chat_eos_v2/fedca_suite/fedca --rounds 2,4,6,8,10
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from peft import get_peft_model

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval_zsre import adapter_path, generate_and_score, generation_metadata, read_jsonl  # noqa: E402
from utils.model_utils import load_lora_config, load_model, load_tokenizer  # noqa: E402
from utils.options import build_parser  # noqa: E402
from utils.seed_utils import set_global_seed  # noqa: E402

KEYS = ("em", "f1", "containment", "num_samples", "average_generated_tokens", "eos_rate", "max_tokens_rate")


def main() -> int:
    parser = build_parser()
    parser.add_argument("--rounds", default="all", help="comma-separated completed rounds, or 'all'")
    parser.add_argument("--data-dir", default="dataset/zsre")
    parser.add_argument("--num-clients", type=int, default=8)
    parser.add_argument("--skip-existing", action="store_true")
    args, _ = parser.parse_known_args()
    if args.dataset != "zsre":
        raise ValueError("requires dataset: zsre")
    set_global_seed(args.seed, device=args.device, deterministic=args.deterministic)
    args.suffix = f"exp/{args.suffix}"
    root_adapter = adapter_path(args)
    available = sorted(int(p.name.split("_")[1]) for p in root_adapter.glob("round_*")
                       if (p / "lora_weights.pt").is_file())
    if not available:
        raise FileNotFoundError(f"no round_XXX/lora_weights.pt under {root_adapter}")
    rounds = available if args.rounds == "all" else [int(r) for r in args.rounds.split(",") if r]
    missing = sorted(set(rounds) - set(available))
    if missing:
        raise FileNotFoundError(f"rounds {missing} were not saved under {root_adapter}")

    tokenizer = load_tokenizer(args)
    model = get_peft_model(load_model(args), load_lora_config(args))
    out_root = Path(args.suffix) / "evaluation" / "zsre" / "rounds"
    summary_path = out_root / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.is_file() else {}
    test_rows = {c: read_jsonl(Path(args.data_dir) / "test" / f"{c}.jsonl") for c in range(args.num_clients)}
    for completed in rounds:
        out = out_root / f"round_{completed:03d}"
        if args.skip_existing and (out / "metrics.json").is_file():
            continue
        state = torch.load(root_adapter / f"round_{completed:03d}" / "lora_weights.pt", map_location="cpu")
        model.load_state_dict(state, strict=False)
        (out / "predictions").mkdir(parents=True, exist_ok=True)
        per_client = {}
        for client_id, rows in test_rows.items():
            metrics, predictions = generate_and_score(model, tokenizer, rows, args.eval_batch_size)
            per_client[str(client_id)] = metrics
            with (out / "predictions" / f"client_{client_id}.jsonl").open("w", encoding="utf-8") as handle:
                for row in predictions:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        macro = {k: sum(per_client[str(c)][k] for c in test_rows) / len(test_rows) for k in KEYS}
        payload = {"method": args.alg, "completed_rounds": completed, "generation": generation_metadata(tokenizer),
                   "per_client": per_client, "macro_average": macro}
        (out / "metrics.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        summary[str(completed)] = macro
        summary_path.write_text(json.dumps(dict(sorted(summary.items(), key=lambda kv: int(kv[0]))), indent=2) + "\n",
                                encoding="utf-8")
        print(f"round {completed}: EM={macro['em']:.2f} F1={macro['f1']:.2f} EOS={macro['eos_rate']:.1f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
