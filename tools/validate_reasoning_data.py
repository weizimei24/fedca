"""Verify prepared reasoning corpora, source fidelity and sampled subsets offline."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from prepare_reasoning_data import (DOMAINS, MATH_CLIENTS, POLICY, digest, fingerprint,
                                    iter_array, math_fingerprint, read_jsonl, sample_indices)


def exact_hash(*values):
    return hashlib.sha256(json.dumps(values, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def benchmark_label_valid(task, row):
    if task == "boolq":
        return row["label"] in (0, 1) and bool(row["passage"].strip())
    if task == "piqa":
        return row["label"] in (0, 1) and all(row[k].strip() for k in ("goal", "sol1", "sol2"))
    if task == "social_i_qa":
        return str(row["label"]) in ("1", "2", "3")
    if task == "hellaswag":
        return str(row["label"]) in ("0", "1", "2", "3") and len(row["endings"]) == 4
    if task == "winogrande":
        return row["answer"] in ("1", "2") and "_" in row["sentence"]
    return row["answerKey"] in row["choices"]["label"]


def validate_full(root: Path, experiment: str, seed: int):
    folder = root / "dataset" / f"{experiment}_full_seed{seed}"
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    if manifest["policy"] != POLICY:
        raise ValueError("Policy mismatch")
    row_hashes = {}
    source_hashes = {}
    exclusions = list(read_jsonl(folder / "excluded.jsonl"))
    excluded_ids = {row["sample_id"] for row in exclusions}
    if len(excluded_ids) != len(exclusions):
        raise ValueError("Duplicate exclusion IDs")
    if experiment == "commonsense170k":
        heldout = {fingerprint(row["input_ids"], "") for path in (folder / "test").glob("*.jsonl") for row in read_jsonl(path)}
    else:
        heldout = {math_fingerprint(row["input_ids"]) for path in (folder / "evaluation").glob("*.jsonl") for row in read_jsonl(path)}
    duplicate_keys = set()
    lineage = {"verified": 0, "unmatched": 0}
    for client in manifest["clients"]:
        cid, partition = client["client_id"], client["partition"]
        source_rows = None
        if experiment == "commonsense170k":
            source_rows = list(iter_array(root / "data/commonsense170k/dataset" / DOMAINS[partition] / "train.json"))
        count = 0
        for row in read_jsonl(folder / "train" / f"{cid}.jsonl"):
            uid, index = row["sample_id"], row["source_row"]
            assert uid not in row_hashes and uid not in excluded_ids, uid
            assert row["client_id"] == cid and row["source_split"] == "train", uid
            assert isinstance(row["input_ids"], str) and row["input_ids"].strip(), uid
            assert isinstance(row["label"], str) and row["label"].strip(), uid
            row_hashes[uid] = exact_hash(row)
            if experiment == "commonsense170k":
                original = source_rows[index]
                prompt = original["instruction"] + ("\n" + original["input"] if original["input"] else "")
                assert (row["input_ids"], row["label"], row["answer"]) == (prompt, original["output"], original["answer"]), uid
                assert row["choice_labels"][row["answer_index"]] == row["answer"], uid
                assert fingerprint(prompt, "") not in heldout, uid
                key = fingerprint(prompt, row["label"])
            else:
                assert row["augmentation_type"] == partition == MATH_CLIENTS[cid], uid
                assert row["source_family"] == partition.split("_")[0], uid
                assert row["original_question_hash"] == math_fingerprint(row["original_question"]), uid
                assert math_fingerprint(row["input_ids"]) not in heldout and row["original_question_hash"] not in heldout, uid
                source_hashes[index] = exact_hash(row["input_ids"], row["label"], row["original_question"], partition)
                lineage["verified" if row["lineage_verified"] else "unmatched"] += 1
                key = partition + ":" + math_fingerprint(row["input_ids"], row["label"])
            assert key not in duplicate_keys, uid
            duplicate_keys.add(key)
            count += 1
        assert count == client["selected"] == client["available"], partition
    assert len(row_hashes) == manifest["train_samples"]
    assert len(row_hashes) + len(exclusions) == manifest["raw_train_samples"]
    if experiment == "metamathqa":
        total = 0
        for index, row in enumerate(iter_array(root / "data/metamathqa/MetaMathQA-395K.json")):
            if index in source_hashes:
                assert source_hashes[index] == exact_hash(row["query"], row["response"], row["original_question"], row["type"]), index
            else:
                assert f"metamathqa:{index:06d}" in excluded_ids, index
            total += 1
        assert total == 395000
    eval_counts = {}
    for path in sorted((folder / ("benchmarks" if experiment == "commonsense170k" else "evaluation")).glob("*.jsonl")):
        rows = list(read_jsonl(path))
        if experiment == "commonsense170k":
            assert all(benchmark_label_valid(path.stem, row) for row in rows), path
        else:
            assert all(row["evaluation_only"] and row["source_split"] == "test" and row["answer"].strip() for row in rows), path
        eval_counts[path.stem] = len(rows)
    expected = ({"boolq": 3270, "piqa": 1838, "social_i_qa": 1954, "hellaswag": 10042,
                 "winogrande": 1267, "arc_easy": 2376, "arc_challenge": 1172, "openbookqa": 500}
                if experiment == "commonsense170k" else {"gsm8k": 1319, "math": 5000})
    assert eval_counts == expected, eval_counts
    return folder, manifest, row_hashes, {"train_rows": len(row_hashes), "exclusions": len(exclusions),
                                         "evaluation_rows": eval_counts, "lineage_of_kept_rows": lineage}


def validate_capped(root: Path, full: Path, parent: dict, hashes: dict, seed: int, cap: int):
    folder = root / "dataset" / f"{parent['experiment']}_client{cap}_seed{seed}"
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["parent_manifest_sha256"] == digest(full / "manifest.json")
    selections = json.loads((folder / "selection.json").read_text(encoding="utf-8"))
    count = 0
    for client in manifest["clients"]:
        cid = client["client_id"]
        rows = list(read_jsonl(folder / "train" / f"{cid}.jsonl"))
        expected_indices = set(sample_indices(client["available"], cap, seed, cid))
        expected_ids = [row["sample_id"] for index, row in enumerate(read_jsonl(full / "train" / f"{cid}.jsonl")) if index in expected_indices]
        ids = [row["sample_id"] for row in rows]
        assert ids == expected_ids == selections[str(cid)], cid
        assert len(ids) == len(set(ids)) == client["selected"] == min(client["available"], cap), cid
        assert all(exact_hash(row) == hashes[row["sample_id"]] for row in rows), cid
        count += len(rows)
    for name in ("test", "benchmarks", "evaluation"):
        if (full / name).exists():
            for path in (full / name).glob("*.jsonl"):
                assert digest(path) == digest(folder / name / path.name), path
    assert count == manifest["train_samples"]
    return folder, manifest, count


def verify_files(folder, manifest):
    for name, expected in manifest["files"].items():
        path = folder / name
        assert path.stat().st_size == expected["bytes"] and digest(path) == expected["sha256"], path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cap", type=int, default=4000)
    parser.add_argument("--experiment", choices=("both", "commonsense170k", "metamathqa"), default="both")
    args = parser.parse_args()
    report = {"passed": True, "seed": args.seed, "cap": args.cap, "datasets": {}}
    experiments = ("commonsense170k", "metamathqa") if args.experiment == "both" else (args.experiment,)
    for experiment in experiments:
        full, manifest, hashes, info = validate_full(args.root, experiment, args.seed)
        verify_files(full, manifest)
        sample, sampled_manifest, count = validate_capped(args.root, full, manifest, hashes, args.seed, args.cap)
        verify_files(sample, sampled_manifest)
        report["datasets"][experiment] = {**info, "capped_train_rows": count,
                                          "full_path": full.relative_to(args.root).as_posix(),
                                          "capped_path": sample.relative_to(args.root).as_posix()}
        print(f"Verified {experiment}: {info['train_rows']:,} full + {count:,} capped rows", flush=True)
    filename = "reasoning_validation.json" if args.experiment == "both" and args.cap == 4000 else f"{args.experiment}_client{args.cap}_validation.json"
    path = args.root / "dataset" / filename
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Validation passed: {path}", flush=True)


if __name__ == "__main__":
    main()
