"""Offline canonical tasks evaluated by the real lm-evaluation-harness engine.

Prompts/metrics follow the pinned upstream task files stored in data/lm_eval_tasks.
SocialIQA is supplied locally (Q: context question / A:, 3 choices, raw accuracy).
No answer-letter generation, chat wrapping, or training rows enter evaluation.
"""
from __future__ import annotations

import importlib.metadata
import json
import math
from pathlib import Path
import re
import sys

from utils.commonsense_training import DOMAINS, atomic_json, fingerprint, load_adapter, sha256

ROOT = Path(__file__).resolve().parents[1]
HARNESS_VERSION = "0.4.13"
EVAL_PROTOCOL = "commonsense_native_mc_0shot_v1"


def ensure_runtime():
    runtime = ROOT / ".runtime/lm_eval"
    if runtime.exists() and str(runtime) not in sys.path:
        sys.path.insert(0, str(runtime))
    try:
        version = importlib.metadata.version("lm_eval")
    except importlib.metadata.PackageNotFoundError as exc:
        raise RuntimeError("Run python tools/install_commonsense_eval.py first") from exc
    if version != HARNESS_VERSION:
        raise RuntimeError(f"Expected lm_eval {HARNESS_VERSION}, found {version}; install the pinned runtime")
    from lm_eval.models.huggingface import HFLM
    return HFLM


def clean_hellaswag(text):
    return re.sub(r"\[.*?\]", "", text.strip().replace(" [title]", ". ")).replace("  ", " ")


def canonical_doc(domain, row):
    if domain == "boolq":
        query = f"{row['passage']}\nQuestion: {row['question']}?\nAnswer:"
        choices, target = ["no", "yes"], int(row["label"])
    elif domain == "piqa":
        query = f"Question: {row['goal']}\nAnswer:"
        choices, target = [row["sol1"], row["sol2"]], int(row["label"])
    elif domain == "social_i_qa":
        query = f"Q: {row['context']} {row['question']}\nA:"
        choices = [row["answerA"], row["answerB"], row["answerC"]]
        target = int(row["label"]) - 1
    elif domain == "hellaswag":
        query = clean_hellaswag(row["activity_label"] + ": " + row["ctx_a"] + " " + row["ctx_b"].capitalize())
        choices = [clean_hellaswag(x) for x in row["endings"]]
        target = int(row["label"])
    elif domain == "winogrande":
        # Upstream uses TWO CONTEXTS and scores the same suffix, not two option
        # continuations. An integer doc_to_text activates multiple_input mode.
        idx = row["sentence"].index("_")
        query = int(row["answer"]) - 1
        choices = [row["sentence"][:idx] + row[k] for k in ("option1", "option2")]
        target = row["sentence"][idx+1:].strip()
    elif domain in ("arc_easy", "arc_challenge", "openbookqa"):
        query = row["question_stem"] if domain == "openbookqa" else f"Question: {row['question']}\nAnswer:"
        choices = row["choices"]["text"]
        target = row["choices"]["label"].index(row["answerKey"].lstrip())
    else:
        raise ValueError(domain)
    if not choices or any(not isinstance(x, str) or not x for x in choices):
        raise ValueError(f"Empty benchmark choice: {domain}")
    if domain != "winogrande" and not 0 <= target < len(choices):
        raise ValueError(f"Invalid benchmark answer: {domain}")
    return {"query": query, "choices": choices, "target": target}


def doc_text(doc):
    return doc["query"]


def doc_target(doc):
    return doc["target"]


def doc_choices(doc):
    return doc["choices"]


def build_task(domain, path):
    ensure_runtime()
    from datasets import Dataset, DatasetDict
    from lm_eval.api.task import ConfigurableTask

    docs = [canonical_doc(domain, json.loads(line))
            for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]

    class LocalCanonicalTask(ConfigurableTask):
        def download(self, *args, **kwargs):
            self.dataset = DatasetDict({"test": Dataset.from_list(docs)})

    metrics = ["acc"] if domain in ("boolq", "social_i_qa", "winogrande") else ["acc", "acc_norm"]
    task = LocalCanonicalTask(config={
        "task": f"cs170k_{domain}", "dataset_path": "local_canonical_jsonl",
        "test_split": "test", "output_type": "multiple_choice", "num_fewshot": 0,
        "doc_to_text": doc_text, "doc_to_target": doc_target, "doc_to_choice": doc_choices,
        "target_delimiter": " ", "fewshot_delimiter": "\n\n",
        "metric_list": [{"metric": m, "aggregation": "mean", "higher_is_better": True} for m in metrics],
        "metadata": {"version": 1.0, "protocol": EVAL_PROTOCOL,
                     "canonical_jsonl_sha256": sha256(path)}})
    task.EVAL_HARNESS_NAME = f"cs170k_{domain}"
    return task


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_safe(v) for v in value]
    if hasattr(value, "item"):
        return json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def evaluate_round(model, tokenizer, dataset, snapshot_path, snapshot_sha, output,
                   method, round_number, identity, *, batch_size=1, max_length=4096, limit=None, seed=42):
    HFLM = ensure_runtime()
    from lm_eval import evaluator
    from lm_eval.tasks import TaskManager
    import torch

    # Snapshot is reloaded even after training: evaluating a local client cannot
    # accidentally use the last client left in the shared model container.
    if sha256(snapshot_path) != snapshot_sha:
        raise ValueError("Round adapter snapshot failed its checksum")
    snapshot = torch.load(snapshot_path, map_location="cpu", weights_only=True)
    if snapshot["identity"] != identity or snapshot["round"] != round_number:
        raise ValueError("Round adapter identity mismatch")
    settings = {"protocol": EVAL_PROTOCOL, "harness": HARNESS_VERSION,
                "implementation_sha256": sha256(Path(__file__)), "method": method,
                "round": round_number, "identity": identity, "snapshot_sha256": snapshot_sha,
                "num_fewshot": 0, "apply_chat_template": False, "limit": limit,
                "batch_size": batch_size, "max_length": max_length, "seed": seed}
    # A diagnostic --eval-limit can never satisfy a later full evaluation.
    evaluation_id = fingerprint(settings)
    directory = Path(output) / "evaluation" / f"round_{round_number:03d}" / evaluation_id[:16]
    directory.mkdir(parents=True, exist_ok=True)
    wrapper = None
    results = {}
    try:
        for cid, domain in enumerate(DOMAINS):
            path = Path(dataset) / "benchmarks" / f"{domain}.jsonl"
            key = fingerprint({**settings, "benchmark_sha256": sha256(path), "task": domain})
            dest = directory / f"{domain}.json"
            if dest.exists():
                cached = json.loads(dest.read_text(encoding="utf-8"))
                if cached["cache_key"] != key:
                    raise ValueError(f"Evaluation cache mismatch: {dest}")
                results[domain] = cached
                print(f"Round {round_number} {domain}: cached", flush=True)
                continue
            load_adapter(model, snapshot["adapters"][str(cid) if method == "localonly" else "global"])
            model.eval()
            if wrapper is None:
                # get_base_model still contains ACTIVE LoRA layers. No merge,
                # weight copy or second base-model allocation is performed.
                hf_model = model.get_base_model() if hasattr(model, "get_base_model") else model
                wrapper = HFLM(pretrained=hf_model, tokenizer=tokenizer, backend="causal",
                               batch_size=batch_size, max_length=max_length,
                               device=str(next(model.parameters()).device))
            task = build_task(domain, path)
            print(f"Evaluating round {round_number}: {domain}, {'FULL' if limit is None else 'DIAGNOSTIC '+str(limit)}", flush=True)
            raw = evaluator.simple_evaluate(
                model=wrapper, tasks=[task], num_fewshot=0, limit=limit,
                bootstrap_iters=0, log_samples=True, apply_chat_template=False,
                task_manager=TaskManager(include_defaults=False),
                random_seed=seed, numpy_random_seed=seed, torch_random_seed=seed,
                fewshot_random_seed=seed)
            task_name = f"cs170k_{domain}"
            result = {"cache_key": key, "settings": settings, "task": domain,
                      "client_id": cid if method == "localonly" else None,
                      "benchmark_sha256": sha256(path),
                      "metrics": raw["results"][task_name], "n_samples": raw.get("n-samples", {}),
                      "harness_result": raw}
            atomic_json(dest, json_safe(result))
            results[domain] = json_safe(result)
            # The wrapper's partial-logit cache is scoped to each loglikelihood
            # call; do not enable persistent SQLite caching across adapters.
        summary = {"settings": settings, "evaluation_id": evaluation_id,
                   "diagnostic_only": limit is not None,
                   "macro_acc": sum(r["metrics"]["acc,none"] for r in results.values()) / len(DOMAINS),
                   "per_task": {d: r["metrics"] for d, r in results.items()},
                   "localonly_protocol": "each independent client on its own source" if method == "localonly" else None}
        atomic_json(directory / "summary.json", summary)
        atomic_json(Path(output) / "evaluation" / f"round_{round_number:03d}" /
                    ("latest_full.json" if limit is None else "latest_diagnostic.json"),
                    {"summary": str((directory / "summary.json").relative_to(output)), **summary})
        print(f"Round {round_number}: macro_acc={summary['macro_acc']:.4f}", flush=True)
        return summary
    finally:
        del wrapper
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
