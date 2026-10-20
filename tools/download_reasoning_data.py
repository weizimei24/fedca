"""Download pinned public reasoning datasets into data/; never starts training.

Run: python -X utf8 -u tools/download_reasoning_data.py
Only stdlib is needed. Existing verified files are reused; partial transfers
resume when the server supports HTTP Range. Upstream bytes stay unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
PINS = {
    "llm_adapters": "816657208af4db747803f87ba40a4c71383fed7a",
    "metamath": "aa4f34d3d2d3231299b5b03d9b3e5a20da45aa18",
    "math": "21a5633873b6a120296cce3e2df9d5550074f4a3",
    "gsm8k": "3101c7d5072418e28b9008a6636bde82a006892c",
    "harness": "d6de81643928d653435c431bae19945d41d32520",
}
DOMAINS = {
    "boolq": "boolq", "piqa": "piqa", "social_i_qa": "social_i_qa",
    "hellaswag": "hellaswag", "winogrande": "winogrande",
    "arc_easy": "ARC-Easy", "arc_challenge": "ARC-Challenge", "openbookqa": "openbookqa",
}
# Canonical benchmark fields for later harness use. PIQA and SocialIQA use
# the official dataset repos' pinned parquet-conversion revisions, not custom
# dataset loading scripts. No unlabelled public-test split is used as evaluation.
BENCHMARKS = {
    "boolq": ("aps/super_glue", "3de24cf8022e94f4ee4b9d55a6f539891524d646", "boolq", "validation"),
    "piqa": ("ybisk/piqa", "142c51238b3ca2bc61e9a075913871b8b600e8e1", "plain_text", "validation"),
    "social_i_qa": ("allenai/social_i_qa", "537a2ec8ec565adc0b70b70752893e59e024df26", "default", "validation"),
    "hellaswag": ("Rowan/hellaswag", "218ec52e09a7e7462a5400043bb9a69a41d06b76", "data", "validation"),
    "winogrande": ("allenai/winogrande", "01e74176c63542e6b0bcb004dcdea22d94fb67b5", "winogrande_xl", "validation"),
    "arc_easy": ("allenai/ai2_arc", "210d026faf9955653af8916fad021475a3f00453", "ARC-Easy", "test"),
    "arc_challenge": ("allenai/ai2_arc", "210d026faf9955653af8916fad021475a3f00453", "ARC-Challenge", "test"),
    "openbookqa": ("allenai/openbookqa", "388097ea7776314e93a529163e0fea805b8a6454", "main", "test"),
}
HEADERS = {"User-Agent": "FedLLM-Factory-reasoning-data/1.0"}


def get_json(url: str):
    print(f"Reading source metadata: {url}", flush=True)
    for attempt in range(5):
        try:
            with urlopen(Request(url, headers=HEADERS), timeout=60) as response:
                return json.load(response)
        except Exception as exc:
            if attempt == 4:
                raise
            print(f"Retrying metadata ({type(exc).__name__})", flush=True)
            time.sleep(min(2 ** attempt, 8))


def digest(path: Path, git_blob: bool = False) -> str:
    value = hashlib.sha1() if git_blob else hashlib.sha256()
    if git_blob:
        value.update(f"blob {path.stat().st_size}\0".encode())
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def valid(path: Path, spec: dict) -> bool:
    return (path.is_file() and path.stat().st_size == spec["size"]
            and digest(path, spec["checksum_type"] == "git_blob_sha1") == spec["checksum"])


def save_receipt(path: Path, info: dict) -> None:
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    # Windows readers/antivirus can briefly deny replacing an open JSON file.
    for attempt in range(8):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == 7:
                raise
            time.sleep(min(0.1 * 2 ** attempt, 0.8))


def fetch(spec: dict, root: Path) -> dict:
    path = root / spec["path"]
    path.parent.mkdir(parents=True, exist_ok=True)
    if not valid(path, spec):
        part = path.with_suffix(path.suffix + ".part")
        for attempt in range(5):
            try:
                offset = part.stat().st_size if part.exists() else 0
                if offset >= spec["size"]:
                    if valid(part, spec):
                        part.replace(path)
                        break
                    # A checksum mismatch is not reusable as a byte prefix.
                    with part.open("wb"):
                        pass
                    offset = 0
                headers = dict(HEADERS)
                if offset:
                    headers["Range"] = f"bytes={offset}-"
                with urlopen(Request(spec["url"], headers=headers), timeout=90) as response:
                    resumed = response.status == 206
                    if resumed and not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
                        raise ValueError("Unexpected HTTP byte range")
                    with part.open("ab" if resumed else "wb") as handle:
                        total = offset if resumed else 0
                        next_notice = total + 32 * 1024 * 1024
                        while chunk := response.read(1024 * 1024):
                            handle.write(chunk)
                            total += len(chunk)
                            if total >= next_notice:
                                print(f"Downloading {spec['path']}: {total:,}/{spec['size']:,} bytes", flush=True)
                                next_notice = total + 32 * 1024 * 1024
                if not valid(part, spec):
                    raise ValueError(f"Incomplete or checksum-mismatched download: {spec['path']}")
                part.replace(path)
                break
            except Exception as exc:
                if attempt == 4:
                    raise
                print(f"Retry {attempt + 1}: {spec['path']} ({type(exc).__name__})", flush=True)
                time.sleep(min(2 ** attempt, 8))
    if not valid(path, spec):
        raise ValueError(f"Verification failed: {path}")
    print(f"Verified {spec['path']} ({path.stat().st_size:,} bytes)", flush=True)
    return {**spec, "sha256": digest(path)}


def catalog() -> list[dict]:
    specs = []

    def github(repo: str, pin: str, wanted, prefix: str):
        tree = get_json(f"https://api.github.com/repos/{repo}/git/trees/{pin}?recursive=1")
        assert tree["sha"] == pin
        for entry in tree["tree"]:
            if entry["type"] == "blob" and wanted(entry["path"]):
                specs.append({"path": f"{prefix}/{entry['path']}",
                              "url": f"https://raw.githubusercontent.com/{repo}/{pin}/{entry['path']}",
                              "size": entry["size"], "checksum_type": "git_blob_sha1",
                              "checksum": entry["sha"], "revision": pin})

    wanted = {"ft-training_set/commonsense_170k.json", "README.md", "DATA_LICENSE"}
    wanted.update(f"dataset/{domain}/{split}.json" for domain in DOMAINS.values() for split in ["train", "test"])
    github("AGI-Edgerunners/LLM-Adapters", PINS["llm_adapters"],
           lambda name: name in wanted, "data/commonsense170k")
    github("openai/grade-school-math", PINS["gsm8k"],
           lambda name: name in {"README.md", "LICENSE", "grade_school_math/data/train.jsonl", "grade_school_math/data/test.jsonl"},
           "data/reasoning_benchmarks/gsm8k")
    # Task definitions are stored for provenance and later harness integration;
    # no harness package is installed or executed here.
    task_dirs = ("arc/", "piqa/", "hellaswag/", "winogrande/", "openbookqa/", "gsm8k/", "hendrycks_math/",
                 "super_glue/boolq/", "social_iqa/", "social_i_qa/")
    github("EleutherAI/lm-evaluation-harness", PINS["harness"],
           lambda name: (name.startswith(tuple("lm_eval/tasks/" + p for p in task_dirs))
                         and name.endswith((".yaml", ".py"))) or name == "LICENSE.md",
           "data/lm_eval_tasks")
    for repo, pin, prefix in [
        ("meta-math/MetaMathQA", PINS["metamath"], "data/metamathqa"),
        ("EleutherAI/hendrycks_math", PINS["math"], "data/reasoning_benchmarks/math"),
    ]:
        entries = get_json(f"https://huggingface.co/api/datasets/{repo}/tree/{pin}?recursive=true&expand=true")
        for entry in entries:
            if entry["type"] != "file" or entry["path"].startswith("."):
                continue
            lfs = entry.get("lfs")
            specs.append({"path": f"{prefix}/{entry['path']}",
                          "url": f"https://huggingface.co/datasets/{repo}/resolve/{pin}/{entry['path']}?download=true",
                          "size": entry["size"], "checksum_type": "sha256" if lfs else "git_blob_sha1",
                          "checksum": lfs["oid"] if lfs else entry["oid"], "revision": pin})
    benchmark_trees = {}
    for task, (repo, pin, config, heldout) in BENCHMARKS.items():
        if (repo, pin) not in benchmark_trees:
            benchmark_trees[repo, pin] = get_json(
                f"https://huggingface.co/api/datasets/{repo}/tree/{pin}?recursive=true&expand=true")
        for entry in benchmark_trees[repo, pin]:
            name = entry["path"]
            if entry["type"] != "file":
                continue
            if not (name == "README.md" or (name.startswith(config + "/") and name.endswith(".parquet")
                    and any(name.startswith(config + "/" + split) for split in ("train", heldout)))):
                continue
            lfs = entry.get("lfs")
            specs.append({"path": f"data/reasoning_benchmarks/commonsense/{task}/{name}",
                          "url": f"https://huggingface.co/datasets/{repo}/resolve/{pin}/{name}?download=true",
                          "size": entry["size"], "checksum_type": "sha256" if lfs else "git_blob_sha1",
                          "checksum": lfs["oid"] if lfs else entry["oid"], "revision": pin,
                          "benchmark": task, "heldout_split": heldout})
    return sorted(specs, key=lambda spec: spec["path"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    receipt = args.root / "data/reasoning_downloads.json"
    receipt.parent.mkdir(parents=True, exist_ok=True)
    specs = catalog()
    info = {"pins": PINS, "benchmark_pins": BENCHMARKS, "files": [], "complete": False,
            "started_at": datetime.now(timezone.utc).isoformat()}
    save_receipt(receipt, info)
    print(f"Downloading {len(specs)} files, {sum(s['size'] for s in specs):,} bytes", flush=True)
    failures = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(fetch, spec, args.root): spec for spec in specs}
        for future in as_completed(futures):
            try:
                info["files"].append(future.result())
            except Exception as exc:
                failures.append({"path": futures[future]["path"], "error": str(exc)})
            info["files"].sort(key=lambda item: item["path"])
            info["failures"] = failures
    info.update(complete=not failures, finished_at=datetime.now(timezone.utc).isoformat())
    save_receipt(receipt, info)
    if failures:
        raise SystemExit(json.dumps(failures, indent=2))
    print(f"All downloads verified; receipt: {receipt}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
