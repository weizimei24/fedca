"""Validate zsRE client files before a Base/Local/Centralized/FedIT run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def fail(message: str) -> None:
    raise ValueError("zsRE validation failed: " + message)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", default="dataset/zsre")
    parser.add_argument("--num_clients", type=int, default=8)
    parser.add_argument("--samples_per_client", type=int, default=None,
                        help="require this exact count (omit to use stats.json's selected count)")
    args = parser.parse_args()
    root = Path(args.data_dir)
    stats_path = root / "stats.json"
    stats = json.loads(stats_path.read_text(encoding="utf-8")) if stats_path.is_file() else {}
    expected = args.samples_per_client if args.samples_per_client is not None else stats.get("selected_samples_per_client")
    mapped = json.loads((root / "client_mapping.json").read_text(encoding="utf-8"))
    mapping = {entry["sample_id"]: entry["client_id"] for entry in mapped}
    if len(mapping) != len(mapped):
        fail("client_mapping.json has duplicated sample IDs")

    all_ids: set[str] = set()
    per_client = {}
    for client_id in range(args.num_clients):
        train = read_jsonl(root / "train" / f"{client_id}.jsonl")
        test = read_jsonl(root / "test" / f"{client_id}.jsonl")
        metadata = read_jsonl(root / "metadata" / f"{client_id}.jsonl")
        if not (len(train) == len(test) == len(metadata)):
            fail(f"client {client_id}: train/test/metadata counts differ ({len(train)}/{len(test)}/{len(metadata)})")
        if expected is not None and len(train) != expected:
            fail(f"client {client_id}: expected {expected} samples, found {len(train)}")
        train_by_id = {row.get("sample_id"): row for row in train}
        test_by_id = {row.get("sample_id"): row for row in test}
        meta_by_id = {row.get("sample_id"): row for row in metadata}
        if len(train_by_id) != len(train) or set(train_by_id) != set(test_by_id) or set(train_by_id) != set(meta_by_id):
            fail(f"client {client_id}: sample IDs are not one-to-one across train/test/metadata")
        overlap = all_ids & set(train_by_id)
        if overlap:
            fail(f"client {client_id}: {len(overlap)} IDs overlap another client")
        all_ids.update(train_by_id)
        for sample_id, train_row in train_by_id.items():
            test_row, metadata_row = test_by_id[sample_id], meta_by_id[sample_id]
            if mapping.get(sample_id) != client_id:
                fail(f"client {client_id}/{sample_id}: wrong or missing mapping")
            if any(row.get("client_id") != client_id for row in (train_row, test_row, metadata_row)):
                fail(f"client {client_id}/{sample_id}: client_id metadata mismatch")
            if train_row.get("input_ids") != metadata_row.get("src"):
                fail(f"client {client_id}/{sample_id}: train prompt is not src")
            if test_row.get("input_ids") != metadata_row.get("rephrase"):
                fail(f"client {client_id}/{sample_id}: test prompt is not rephrase")
            if not metadata_row.get("src") or not metadata_row.get("rephrase") or not metadata_row.get("alt"):
                fail(f"client {client_id}/{sample_id}: empty factual field")
            if " ".join(metadata_row["src"].casefold().split()) == " ".join(metadata_row["rephrase"].casefold().split()):
                fail(f"client {client_id}/{sample_id}: src equals rephrase")
            if train_row.get("label") != metadata_row["alt"] or test_row.get("label") != metadata_row["alt"]:
                fail(f"client {client_id}/{sample_id}: labels are not alt")
        per_client[str(client_id)] = len(train)
    if set(mapping) != all_ids:
        fail("client_mapping.json does not exactly cover the selected client files")
    if len(set(per_client.values())) != 1:
        fail(f"clients are not equal sized: {per_client}")
    print(f"zsRE validation passed: {args.num_clients} clients, {next(iter(per_client.values()), 0)} aligned train/test facts each, {len(all_ids)} total.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
