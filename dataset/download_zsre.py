"""Download the public ROME/MEND zsRE files with an auditable manifest.

The training file is deliberately kept raw: ``process_zsre.py`` is the only
place that turns it into client data and records the selected records.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen


# These are the public files published by the ROME project and used by MEND.
ROME_DATASETS = {
    "zsre_mend_train.json": "https://rome.baulab.info/data/dsets/zsre_mend_train.json",
    "zsre_mend_eval.json": "https://rome.baulab.info/data/dsets/zsre_mend_eval.json",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download(url: str, destination: Path) -> None:
    if destination.is_file() and destination.stat().st_size > 0:
        print(f"Using cached {destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    request = Request(url, headers={"User-Agent": "FedLLM-Factory-zsRE/1.0"})
    try:
        with urlopen(request, timeout=120) as response, temporary.open("wb") as handle:
            shutil.copyfileobj(response, handle)
        # Fail early for accidental HTML error pages or partial responses.
        with temporary.open("r", encoding="utf-8") as handle:
            parsed = json.load(handle)
        if not isinstance(parsed, list):
            raise ValueError("expected a JSON list")
        temporary.replace(destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", default="dataset/zsre/raw")
    parser.add_argument(
        "--include-eval", action=argparse.BooleanOptionalAction, default=True,
        help="also cache zsre_mend_eval.json (not used for the client construction)",
    )
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    names = ["zsre_mend_train.json"]
    if args.include_eval:
        names.append("zsre_mend_eval.json")

    files = {}
    for name in names:
        url = ROME_DATASETS[name]
        path = output_dir / name
        print(f"Downloading/checking {name} from {url}")
        download(url, path)
        files[name] = {"url": url, "sha256": sha256(path), "bytes": path.stat().st_size}

    manifest = {
        "dataset": "zsRE MEND/ROME public raw files",
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "files": files,
    }
    (output_dir / "SOURCES.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Saved raw zsRE files and manifest under {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
