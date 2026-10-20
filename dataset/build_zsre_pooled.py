"""Build an auditable shuffled centralized zsRE pool from client train data."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", default="dataset/zsre")
    parser.add_argument("--num_clients", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="pooled.jsonl")
    args = parser.parse_args()
    root = Path(args.data_dir)
    rows, ids, counts = [], set(), {}
    for client_id in range(args.num_clients):
        client_rows = read_jsonl(root / "train" / f"{client_id}.jsonl")
        client_ids = {row.get("sample_id") for row in client_rows}
        if len(client_ids) != len(client_rows) or None in client_ids or ids & client_ids:
            raise ValueError(f"invalid or overlapping sample IDs for client {client_id}")
        ids.update(client_ids)
        rows.extend(client_rows)
        counts[str(client_id)] = len(client_rows)
    if len(set(counts.values())) != 1:
        raise ValueError(f"centralized pool requires equal source client counts: {counts}")
    random.Random(args.seed).shuffle(rows)
    path = root / "train" / args.output
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {len(rows)} zsRE facts to {path}; client counts: {counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
