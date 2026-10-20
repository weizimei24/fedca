"""Prepare full and per-client capped, seed-42 federated reasoning corpora.

Raw downloads remain under data/. Only new dataset/ directories are written.
No training, tokenizer truncation, model download, or trainer/config change.
Run after download_reasoning_data.py, using an environment with pyarrow.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import shutil
import sys
import tempfile
import time
from collections import Counter, defaultdict
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from download_reasoning_data import BENCHMARKS, DOMAINS, PINS, digest, valid

MATH_CLIENTS = (
    "GSM_AnsAug", "GSM_Rephrased", "GSM_FOBAR", "GSM_SV",
    "MATH_AnsAug", "MATH_Rephrased", "MATH_FOBAR", "MATH_SV",
)
POLICY = {
    "normalization_for_checks": "commonsense: casefold plus collapsed whitespace; math: collapsed whitespace, case preserved; stored text unchanged",
    "duplicates": "remove repeated normalized prompt+target, globally for commonsense and within augmentation type for math",
    "conflicts": "quarantine all commonsense rows sharing a prompt but conflicting answer labels",
    "heldout": "exclude exact normalized commonsense evaluation prompts; exclude math query/original_question matching GSM8K or MATH test problems",
    "multiple_math_solutions": "retain distinct response augmentations; do not deduplicate merely by original_question",
    "sampling": "without replacement; min(4000, available) per client; never top up small sources",
    "evaluation": "retain complete labelled benchmark holdouts, never sampled into training",
    "tokenization": "none: preserve complete prompts, reasoning and mathematical expressions; no length filtering/truncation",
}


def norm(text: str) -> str:
    return " ".join(text.casefold().split())


def fingerprint(*texts: str) -> str:
    return hashlib.sha256(json.dumps([norm(x) for x in texts], ensure_ascii=False).encode()).hexdigest()


def math_fingerprint(*texts: str) -> str:
    # Case is semantic in TeX: \Pi is different from \pi, as are X and x.
    values = [" ".join(text.split()) for text in texts]
    return hashlib.sha256(json.dumps(values, ensure_ascii=False).encode()).hexdigest()


def iter_array(path: Path, chunk_size: int = 1024 * 1024):
    """Iterate a JSON array without loading the 395 MB MetaMath file at once."""
    decoder = json.JSONDecoder()
    with path.open(encoding="utf-8-sig") as handle:
        buffer, position, eof = "", 0, False

        def fill():
            nonlocal buffer, position, eof
            buffer = buffer[position:] + handle.read(chunk_size)
            position = 0
            eof = len(buffer) == 0 or handle.tell() == path.stat().st_size

        def whitespace():
            nonlocal position
            while True:
                while position < len(buffer) and buffer[position].isspace():
                    position += 1
                if position < len(buffer) or eof:
                    return
                fill()

        fill()
        whitespace()
        if position >= len(buffer) or buffer[position] != "[":
            raise ValueError(f"Expected JSON array: {path}")
        position += 1
        first = True
        while True:
            whitespace()
            if position < len(buffer) and buffer[position] == "]":
                position += 1
                break
            if not first:
                if position >= len(buffer) or buffer[position] != ",":
                    raise ValueError(f"Missing array separator in {path}")
                position += 1
                whitespace()
                if position < len(buffer) and buffer[position] == "]":
                    raise ValueError(f"Trailing comma in {path}")
            while True:
                try:
                    row, end = decoder.raw_decode(buffer, position)
                    break
                except json.JSONDecodeError:
                    if eof:
                        raise
                    fill()
            if not isinstance(row, dict):
                raise ValueError(f"Expected object row: {path}")
            yield row
            position = end
            first = False
        whitespace()
        if position < len(buffer) or handle.read().strip():
            raise ValueError(f"Trailing data after JSON array: {path}")


def dump_json(path: Path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def promote_directory(stage: Path, output: Path):
    # File indexing/antivirus may briefly hold recently written files on Windows.
    for attempt in range(12):
        if output.exists():
            raise FileExistsError(f"Refusing to replace an existing dataset: {output}")
        try:
            stage.rename(output)
            return
        except PermissionError:
            if attempt == 11:
                raise
            time.sleep(min(0.1 * 2 ** attempt, 1.0))


def write_row(handle, row):
    handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            write_row(handle, row)
            count += 1
    return count


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def choice_data(row: dict) -> tuple[str, list[str], list[str], int]:
    """Preserve source labels (true/false, solution1, answer5, etc.)."""
    instruction, answer = row["instruction"], row["answer"]
    body, label_text = instruction.rsplit("Answer format:", 1)
    labels = [label.strip() for label in label_text.strip().split("/")]
    if len(set(labels)) != len(labels) or answer not in labels:
        raise ValueError("Answer does not match the source answer format")
    if set(labels) == {"true", "false"}:
        return body.strip(), labels, labels, labels.index(answer)
    positions = []
    for label in labels:
        matches = list(re.finditer(rf"\b{re.escape(label)}\s*:\s*", body, flags=re.IGNORECASE))
        if len(matches) != 1:
            raise ValueError(f"Expected one {label} option marker")
        positions.append(matches[0])
    if [p.start() for p in positions] != sorted(p.start() for p in positions):
        raise ValueError("Option order disagrees with labels")
    choices = [body[p.end():positions[i + 1].start() if i + 1 < len(positions) else len(body)].strip()
               for i, p in enumerate(positions)]
    if any(not choice for choice in choices):
        raise ValueError("Empty option")
    return body[:positions[0].start()].strip(), choices, labels, labels.index(answer)


def boxed_answer(solution: str) -> str | None:
    """Extract the last balanced boxed/fbox expression, including nested TeX."""
    matches = list(re.finditer(r"\\(?:boxed|fbox)(?=\s|\{)", solution))
    if not matches:
        return None
    position = matches[-1].end()
    while position < len(solution) and solution[position].isspace():
        position += 1
    if position >= len(solution):
        return None
    if solution[position] != "{":
        # Two official algebra training solutions use TeX's unbraced numeric
        # form, `\boxed 2` / `\boxed 9`.
        match = re.match(r"([+-]?\d+(?:\.\d+)?)(?=[\s$.,;}]|$)", solution[position:])
        return match[1] if match else None
    start = position + 1
    depth = 1
    for index in range(start, len(solution)):
        char = solution[index]
        if index and solution[index - 1] == "\\":
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return solution[start:index]
    return None


def sample_indices(size: int, cap: int, seed: int, client_id: int) -> list[int]:
    return sorted(random.Random(seed + client_id * 10_007).sample(range(size), min(size, cap)))


def parquet_rows(path: Path):
    import pyarrow.parquet as pq

    for batch in pq.ParquetFile(path).iter_batches(batch_size=1024):
        yield from batch.to_pylist()


def canonical_common_eval(root: Path):
    result = {}
    for task, (_, _, config, split) in BENCHMARKS.items():
        base = root / "data/reasoning_benchmarks/commonsense" / task / config
        paths = sorted(base.glob(f"{split}*.parquet")) or sorted((base / split).glob("*.parquet"))
        if not paths:
            raise FileNotFoundError(f"Missing canonical {task}/{split} parquet")
        records = []
        for path in paths:
            for index, row in enumerate(parquet_rows(path)):
                # Keep benchmark fields/labels untouched for future harness use.
                records.append({**row, "_provenance": {"benchmark": task, "split": split,
                                "source_file": path.relative_to(root).as_posix(), "source_row": index,
                                "evaluation_only": True}})
        result[task] = records
    return result


def math_references(root: Path):
    evaluation, original_train = {}, {}
    for split in ("train", "test"):
        source = root / "data/reasoning_benchmarks/gsm8k/grade_school_math/data" / f"{split}.jsonl"
        gsm = []
        for index, row in enumerate(read_jsonl(source)):
            if "####" not in row["answer"]:
                raise ValueError(f"Missing GSM8K gold answer: {index}")
            record = {"sample_id": f"gsm8k:{split}:{index:05d}", "source_dataset": "gsm8k",
                      "source_split": split, "source_file": source.relative_to(root).as_posix(), "source_row": index,
                      "input_ids": row["question"], "label": row["answer"],
                      "answer": row["answer"].rsplit("####", 1)[1].strip(), "evaluation_only": split == "test"}
            gsm.append(record)
        math = []
        for path in sorted((root / "data/reasoning_benchmarks/math").glob(f"*/{split}-*.parquet")):
            for index, row in enumerate(parquet_rows(path)):
                answer = boxed_answer(row["solution"])
                if answer is None:
                    raise ValueError(f"Cannot extract official MATH answer: {path}:{index}")
                math.append({"sample_id": f"math:{path.parent.name}:{split}:{index:05d}", "source_dataset": "math",
                             "source_split": split, "source_file": path.relative_to(root).as_posix(), "source_row": index,
                             "input_ids": row["problem"], "label": row["solution"], "answer": answer,
                             "level": row["level"], "subject": row["type"], "evaluation_only": split == "test"})
        if (len(gsm), len(math)) != ((7473, 7500) if split == "train" else (1319, 5000)):
            raise ValueError(f"Unexpected {split} benchmark sizes: {len(gsm)}, {len(math)}")
        target = evaluation if split == "test" else original_train
        target.update(gsm8k=gsm, math=math)
    return evaluation, original_train


def common_rows(root: Path):
    """Prove the published aggregate equals the eight source multisets."""
    raw = root / "data/commonsense170k"
    sources = {}
    expected = Counter()
    heldout = set()
    answers_by_prompt = defaultdict(set)
    for task, name in DOMAINS.items():
        path = raw / "dataset" / name / "train.json"
        rows = list(iter_array(path))
        sources[task] = rows
        expected.update(json.dumps(row, sort_keys=True, ensure_ascii=False) for row in rows)
        for row in rows:
            answers_by_prompt[fingerprint(row["instruction"], row["input"])].add(norm(row["output"]))
        for row in iter_array(raw / "dataset" / name / "test.json"):
            heldout.add(fingerprint(row["instruction"], row["input"]))
    actual = Counter(json.dumps(row, sort_keys=True, ensure_ascii=False)
                     for row in iter_array(raw / "ft-training_set/commonsense_170k.json"))
    if actual != expected or sum(actual.values()) != 170420:
        raise ValueError("Commonsense170K aggregate does not exactly match source datasets")
    return sources, heldout, {key for key, labels in answers_by_prompt.items() if len(labels) > 1}


def add_file_manifest(folder: Path, manifest: dict):
    manifest["files"] = {}
    for path in sorted(folder.rglob("*")):
        if path.is_file() and path.name != "manifest.json":
            manifest["files"][path.relative_to(folder).as_posix()] = {"bytes": path.stat().st_size, "sha256": digest(path)}
    dump_json(folder / "manifest.json", manifest)


def ready_existing(folder: Path, seed: int, policy=None) -> bool:
    if not folder.exists():
        return False
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    if manifest["seed"] != seed or manifest["policy"] != (POLICY if policy is None else policy):
        raise ValueError(f"Existing preparation has different settings: {folder}")
    for name, expected in manifest["files"].items():
        path = folder / name
        if not path.is_file() or digest(path) != expected["sha256"]:
            raise ValueError(f"Existing prepared file changed: {path}")
    print(f"Reusing verified prepared dataset: {folder}", flush=True)
    return True


def dataset_readme(name: str, manifest: dict) -> str:
    lines = [f"# {name}", "", "Data preparation only; no training or evaluation has been run.", "",
             "Training text uses the project's `input_ids` (string) and `label` (string) fields.",
             "`train/0.jsonl` etc. have fixed client IDs in `client_mapping.json`.",
             "Text, complete reasoning, math notation and original answer labels are preserved.",
             "No tokenizer, padding, truncation, model, optimizer, learning-rate or round-count changes are applied.", "",
             "| Client | Partition | Available clean training rows | Selected rows |", "|---|---|---:|---:|"]
    for client in manifest["clients"]:
        lines.append(f"| {client['client_id']} | {client['partition']} | {client['available']} | {client['selected']} |")
    lines += ["", f"Total selected training rows: {manifest['train_samples']:,}.", "",
              "The capped version samples without replacement with seed 42 plus a fixed client offset.",
              f"Sources with fewer than {manifest.get('requested_cap', 4000):,} clean rows keep every row; no duplicate top-ups.", "",
              "The full version means all usable rows after the recorded duplicate/conflict/holdout checks.",
              "Unmodified upstream files remain under `data/`. Exclusions are auditable in the full version.", ""]
    if manifest["experiment"] == "commonsense170k":
        lines += ["`test/<client>.jsonl` preserves the LLM-Adapters labelled benchmark instructions.",
                  "`benchmarks/<task>.jsonl` separately preserves canonical benchmark fields for future",
                  "lm-evaluation-harness use, including BoolQ passages. The two formats are not the same protocol.",
                  "BoolQ, PIQA, SocialIQA, HellaSwag and WinoGrande use labelled validation splits;",
                  "ARC-Easy, ARC-Challenge and OpenBookQA use labelled test splits.",
                  "Benchmark holdouts are kept complete in both full and capped versions.", ""]
    else:
        lines += ["The eight clients follow the published `type`: GSM/MATH x AnsAug/Rephrased/FOBAR/SV.",
                  "Different augmentation clients can share a training original_question by design.",
                  "Distinct solutions are retained; `original_question` and lineage verification are recorded.",
                  "`evaluation/gsm8k.jsonl` contains all 1,319 official test items;",
                  "`evaluation/math.jsonl` contains all 5,000 MATH test items across seven subjects.",
                  "These benchmark files are global evaluation data, not client training files.",
                  "Gold final answers are extracted as data; an actual mathematical equivalence evaluator is not implemented here.", ""]
    lines += ["See `manifest.json`, `quality_report.json`, and (for capped datasets) `selection.json` for provenance.",
              "Training integration is a later step: existing zsRE-only evaluators and suites cannot be used unchanged.",
              "In particular, the current generic encoder hard-codes 512 tokens and does not supervise EOS;",
              "do not silently truncate mathematical solutions. The existing step=0 behavior also is not unlimited training.", ""]
    return "\n".join(lines)


def make_full(root: Path, experiment: str, seed: int, eval_math, original_train, common_eval):
    output = root / "dataset" / f"{experiment}_full_seed{seed}"
    if ready_existing(output, seed):
        return output
    output.parent.mkdir(parents=True, exist_ok=True)
    partitions = list(DOMAINS) if experiment == "commonsense170k" else list(MATH_CLIENTS)
    stats = {key: Counter() for key in partitions}
    counts = Counter()
    seen = {}
    source_counts = Counter()
    lineage = Counter()
    lengths = {key: {"max_prompt_chars": 0, "max_target_chars": 0} for key in partitions}
    with tempfile.TemporaryDirectory(prefix=f".{output.name}.preparing-", dir=output.parent) as tmp:
        stage = Path(tmp) / "dataset"
        (stage / "train").mkdir(parents=True)
        with ExitStack() as stack:
            handles = {key: stack.enter_context((stage / "train" / f"{i}.jsonl").open("w", encoding="utf-8", newline="\n"))
                       for i, key in enumerate(partitions)}
            rejected = stack.enter_context((stage / "excluded.jsonl").open("w", encoding="utf-8", newline="\n"))

            def reject(key, row_id, source_index, reason, **details):
                stats[key][reason] += 1
                write_row(rejected, {"sample_id": row_id, "partition": key, "source_row": source_index,
                                     "reason": reason, **details})

            def accept(key, record, check_key):
                if check_key in seen:
                    reject(key, record["sample_id"], record["source_row"], "duplicate_prompt_target", duplicate_of=seen[check_key])
                    return
                seen[check_key] = record["sample_id"]
                write_row(handles[key], record)
                counts[key] += 1
                lengths[key]["max_prompt_chars"] = max(lengths[key]["max_prompt_chars"], len(record["input_ids"]))
                lengths[key]["max_target_chars"] = max(lengths[key]["max_target_chars"], len(record["label"]))

            if experiment == "commonsense170k":
                sources, heldout, conflicts = common_rows(root)
                for client, (key, rows) in enumerate(sources.items()):
                    source = f"data/commonsense170k/dataset/{DOMAINS[key]}/train.json"
                    for index, row in enumerate(rows):
                        source_counts[key] += 1
                        row_id = f"cs170k:{key}:{index:06d}"
                        prompt_hash = fingerprint(row["instruction"], row["input"])
                        if prompt_hash in heldout:
                            reject(key, row_id, index, "evaluation_prompt_overlap")
                            continue
                        if prompt_hash in conflicts:
                            reject(key, row_id, index, "conflicting_answer_labels")
                            continue
                        try:
                            question, choices, labels, answer_index = choice_data(row)
                        except (ValueError, KeyError) as exc:
                            reject(key, row_id, index, "invalid_choice_record", detail=str(exc))
                            continue
                        if sum(norm(c) == norm(choices[answer_index]) for c in choices) > 1:
                            reject(key, row_id, index, "ambiguous_correct_option")
                            continue
                        record = {"sample_id": row_id, "client_id": client, "source_dataset": key,
                                  "source_split": "train", "source_file": source, "source_row": index,
                                  "input_ids": row["instruction"] + ("\n" + row["input"] if row["input"] else ""),
                                  "label": row["output"], "answer": row["answer"], "choices": choices,
                                  "choice_labels": labels, "answer_index": answer_index}
                        accept(key, record, fingerprint(record["input_ids"], record["label"]))
                    print(f"{key}: raw={len(rows):,} kept={counts[key]:,} excluded={dict(stats[key])}", flush=True)
                for client, key in enumerate(partitions):
                    path = root / "data/commonsense170k/dataset" / DOMAINS[key] / "test.json"
                    rows = []
                    for index, row in enumerate(iter_array(path)):
                        _, choices, labels, answer_index = choice_data(row)
                        rows.append({"sample_id": f"cs170k_eval:{key}:{index:05d}", "client_id": client,
                                     "source_dataset": key, "source_split": BENCHMARKS[key][3],
                                     "source_file": path.relative_to(root).as_posix(), "source_row": index,
                                     "input_ids": row["instruction"], "label": row["output"], "answer": row["answer"],
                                     "choices": choices, "choice_labels": labels, "answer_index": answer_index,
                                     "evaluation_only": True})
                    if len(rows) != len(common_eval[key]):
                        raise ValueError(f"Canonical and LLM-Adapters heldout counts disagree: {key}")
                    write_jsonl(stage / "test" / f"{client}.jsonl", rows)
                    write_jsonl(stage / "benchmarks" / f"{key}.jsonl", common_eval[key])
            else:
                test_questions = {math_fingerprint(r["input_ids"]) for rows in eval_math.values() for r in rows}
                originals = {math_fingerprint(r["input_ids"]): r["sample_id"] for rows in original_train.values() for r in rows}
                source = root / "data/metamathqa/MetaMathQA-395K.json"
                for index, row in enumerate(iter_array(source)):
                    key = row["type"]
                    if key not in partitions:
                        raise ValueError(f"Unrecognized MetaMath type: {key}")
                    source_counts[key] += 1
                    row_id = f"metamathqa:{index:06d}"
                    if not all(isinstance(row.get(field), str) and row[field].strip()
                               for field in ("query", "response", "original_question")):
                        reject(key, row_id, index, "empty_text")
                        continue
                    original_hash = math_fingerprint(row["original_question"])
                    if original_hash in test_questions or math_fingerprint(row["query"]) in test_questions:
                        reject(key, row_id, index, "evaluation_problem_overlap")
                        continue
                    verified = original_hash in originals
                    lineage["verified_original_train" if verified else "unmatched_original_train"] += 1
                    record = {"sample_id": row_id, "client_id": partitions.index(key), "source_dataset": "metamathqa",
                              "source_family": key.split("_", 1)[0], "augmentation_type": key, "source_split": "train",
                              "source_file": source.relative_to(root).as_posix(), "source_row": index,
                              "input_ids": row["query"], "label": row["response"], "original_question": row["original_question"],
                              "original_question_hash": original_hash, "lineage_verified": verified,
                              "source_problem_id": originals.get(original_hash)}
                    accept(key, record, key + ":" + math_fingerprint(row["query"], row["response"]))
                    if (index + 1) % 100000 == 0:
                        print(f"MetaMath processed {index + 1:,} rows", flush=True)
                if sum(source_counts.values()) != 395000:
                    raise ValueError("Unexpected MetaMathQA source count")
                for key, rows in eval_math.items():
                    write_jsonl(stage / "evaluation" / f"{key}.jsonl", rows)
                for key in partitions:
                    print(f"{key}: raw={source_counts[key]:,} kept={counts[key]:,} excluded={dict(stats[key])}", flush=True)
        clients = [{"client_id": i, "partition": key, "raw_source_rows": source_counts[key],
                    "available": counts[key], "selected": counts[key]} for i, key in enumerate(partitions)]
        assert all(c["raw_source_rows"] == c["selected"] + sum(stats[c["partition"]].values()) for c in clients)
        dump_json(stage / "client_mapping.json", clients)
        quality = {"source_counts": source_counts, "excluded_by_partition": stats, "lineage": lineage,
                   "character_lengths_not_tokens": lengths, "duplicate_and_overlap_checks_are_exact_not_semantic": True}
        dump_json(stage / "quality_report.json", quality)
        manifest = {"experiment": experiment, "variant": "full", "seed": seed, "policy": POLICY,
                    "source_pins": PINS, "download_receipt_sha256": digest(root / "data/reasoning_downloads.json"),
                    "clients": clients, "train_samples": sum(counts.values()),
                    "raw_train_samples": sum(source_counts.values()), "created_at": datetime.now(timezone.utc).isoformat(),
                    "training_code_changed": False, "training_hyperparameters_changed": False,
                    "training_started": False, "integration_pending": True}
        (stage / "README.md").write_text(dataset_readme(output.name, manifest), encoding="utf-8")
        add_file_manifest(stage, manifest)
        promote_directory(stage, output)
    return output


def make_capped(root: Path, full: Path, seed: int, cap: int):
    parent = json.loads((full / "manifest.json").read_text(encoding="utf-8"))
    output = root / "dataset" / f"{parent['experiment']}_client{cap}_seed{seed}"
    # Preserve the policy of already prepared 4000-row datasets, while recording
    # the actual cap in new variants instead of claiming all variants use 4000.
    policy = {**POLICY, "sampling": f"without replacement; min({cap}, available) per client; never top up small sources"}
    if ready_existing(output, seed, policy):
        return output
    with tempfile.TemporaryDirectory(prefix=f".{output.name}.preparing-", dir=output.parent) as tmp:
        stage = Path(tmp) / "dataset"
        stage.mkdir()
        clients, selected = [], {}
        for client in parent["clients"]:
            cid = client["client_id"]
            indices = set(sample_indices(client["available"], cap, seed, cid))
            rows = [row for index, row in enumerate(read_jsonl(full / "train" / f"{cid}.jsonl")) if index in indices]
            assert len(rows) == len(indices)
            write_jsonl(stage / "train" / f"{cid}.jsonl", rows)
            selected[str(cid)] = [row["sample_id"] for row in rows]
            clients.append({**client, "selected": len(rows)})
        for name in ("test", "benchmarks", "evaluation"):
            if (full / name).exists():
                shutil.copytree(full / name, stage / name)
        shutil.copyfile(full / "quality_report.json", stage / "quality_report.json")
        dump_json(stage / "selection.json", selected)
        dump_json(stage / "client_mapping.json", clients)
        manifest = {**{k: v for k, v in parent.items() if k != "files"}, "variant": "capped_per_client",
                    "requested_cap": cap, "policy": policy, "parent_dataset": full.relative_to(root).as_posix(),
                    "parent_manifest_sha256": digest(full / "manifest.json"), "clients": clients,
                    "train_samples": sum(c["selected"] for c in clients), "created_at": datetime.now(timezone.utc).isoformat()}
        (stage / "README.md").write_text(dataset_readme(output.name, manifest), encoding="utf-8")
        add_file_manifest(stage, manifest)
        promote_directory(stage, output)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cap", type=int, default=4000)
    parser.add_argument("--experiment", choices=("both", "commonsense170k", "metamathqa"), default="both")
    args = parser.parse_args()
    if args.cap <= 0:
        raise ValueError("cap must be positive")
    receipt = json.loads((args.root / "data/reasoning_downloads.json").read_text(encoding="utf-8"))
    if not receipt["complete"]:
        raise ValueError("Download receipt is incomplete; finish downloads first")
    for spec in receipt["files"]:
        if not valid(args.root / spec["path"], spec):
            raise ValueError(f"Raw source changed: {spec['path']}")
    print("All raw files re-verified", flush=True)
    experiments = ("commonsense170k", "metamathqa") if args.experiment == "both" else (args.experiment,)
    eval_math, original_train = math_references(args.root) if "metamathqa" in experiments else ({}, {})
    common_eval = canonical_common_eval(args.root) if "commonsense170k" in experiments else {}
    for experiment in experiments:
        full = make_full(args.root, experiment, args.seed, eval_math, original_train, common_eval)
        capped = make_capped(args.root, full, args.seed, args.cap)
        print(f"Prepared {full}\nPrepared {capped}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
