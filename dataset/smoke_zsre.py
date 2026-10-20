"""Offline 8-client x 10-fact smoke test for zsRE processing and validation.

This does not download or claim to reproduce the real zsRE benchmark. It only
creates disposable schema-compatible fixture records, uses the explicit TF-IDF
test backend, and proves that client construction plus invariant validation
work end to end.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    with tempfile.TemporaryDirectory(prefix="zsre-smoke-") as temporary:
        temporary_path = Path(temporary)
        raw_dir, output_dir = temporary_path / "raw", temporary_path / "zsre"
        raw_dir.mkdir()
        rows = []
        topic_terms = (
            "astronomy galaxy telescope orbit",
            "culinary recipe kitchen flavor",
            "music melody orchestra rhythm",
            "botany forest flower ecology",
            "geology mineral volcano tectonic",
            "computing algorithm network software",
            "history empire archive dynasty",
            "sports athlete stadium tournament",
        )
        for topic in range(8):
            for index in range(10):
                # Identical in-topic subject text makes this a deliberately
                # clear semantic-clustering fixture; uniqueness remains in
                # src/rephrase/alt, so no record is a content duplicate.
                subject = f"{topic_terms[topic]} itemvariation{chr(97 + index)}"
                rows.append({
                    "case_id": f"smoke-{topic}-{index}",
                    "subject": subject,
                    "src": f"What is revised fact number {index} for {subject}?",
                    "rephrase": f"Name new fact {index} of {subject}.",
                    "alt": f"new-value-{topic}-{index}",
                    "answers": [f"old-value-{topic}-{index}"],
                    "loc": "unrelated locality prompt",
                    "loc_ans": "unrelated answer",
                })
        (raw_dir / "zsre_mend_train.json").write_text(json.dumps(rows), encoding="utf-8")
        process = [
            sys.executable, str(root / "dataset" / "process_zsre.py"),
            "--raw_dir", str(raw_dir), "--output_dir", str(output_dir),
            "--num_clients", "8", "--samples_per_client", "10", "--seed", "42",
            "--embedding-backend", "tfidf",
        ]
        validate = [
            sys.executable, str(root / "dataset" / "validate_zsre.py"),
            "--data_dir", str(output_dir), "--num_clients", "8", "--samples_per_client", "10",
        ]
        subprocess.run(process, cwd=root, check=True)
        subprocess.run(validate, cwd=root, check=True)
    print("zsRE smoke test passed: processed and validated 8 clients x 10 facts.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
