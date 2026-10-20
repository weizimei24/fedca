"""Create equal-sized, semantically clustered zsRE clients.

Each retained knowledge item has two prompts: source ``src`` is supervised in
train, while the semantically equivalent ``rephrase`` is used only in test.
The target in both files is the counterfactual/new fact ``alt`` -- never the
original answer from ``answers``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


RAW_CANDIDATES = ("zsre_mend_train_10000.json", "zsre_mend_train.json")


def _text(value: Any) -> str:
    if isinstance(value, list):
        value = value[0] if value else ""
    return str(value or "").strip()


def _answer(record: dict) -> str:
    for key in ("answers", "original_answer", "ground_truth", "target_true"):
        if key in record:
            return _text(record[key])
    return ""


def _norm(value: str) -> str:
    return " ".join(value.casefold().split())


def _record_id(record: dict, raw_index: int) -> str:
    for key in ("sample_id", "case_id", "id", "uuid"):
        value = _text(record.get(key))
        if value:
            return value
    return f"zsre-{raw_index:07d}"


def canonicalise(record: dict, raw_index: int) -> tuple[dict | None, str | None]:
    src, rephrase, alt = (_text(record.get(key)) for key in ("src", "rephrase", "alt"))
    if not src:
        return None, "empty_src"
    if not rephrase:
        return None, "empty_rephrase"
    if not alt:
        return None, "empty_alt"
    if _norm(src) == _norm(rephrase):
        return None, "identical_src_rephrase"
    subject = _text(record.get("subject"))
    return {
        "sample_id": _record_id(record, raw_index),
        "raw_index": raw_index,
        "subject": subject,
        "src": src,
        "rephrase": rephrase,
        "alt": alt,
        "original_answer": _answer(record),
        "loc": _text(record.get("loc")),
        "loc_ans": _text(record.get("loc_ans")),
    }, None


def load_records(raw_path: Path) -> tuple[list[dict], dict]:
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"{raw_path} must contain a JSON list, found {type(raw).__name__}")
    valid: list[dict] = []
    invalid = Counter()
    duplicate_content = 0
    duplicate_ids = 0
    seen_content: set[tuple[str, str, str]] = set()
    seen_ids: set[str] = set()
    for index, original in enumerate(raw):
        if not isinstance(original, dict):
            invalid["not_object"] += 1
            continue
        record, reason = canonicalise(original, index)
        if reason:
            invalid[reason] += 1
            continue
        assert record is not None
        content_key = (_norm(record["src"]), _norm(record["rephrase"]), _norm(record["alt"]))
        if content_key in seen_content:
            duplicate_content += 1
            continue
        if record["sample_id"] in seen_ids:
            duplicate_ids += 1
            continue
        seen_content.add(content_key)
        seen_ids.add(record["sample_id"])
        valid.append(record)
    return valid, {
        "raw_examples": len(raw),
        "valid_examples": len(valid),
        "dropped_invalid_examples": sum(invalid.values()),
        "invalid_by_reason": dict(sorted(invalid.items())),
        "duplicate_content_examples": duplicate_content,
        "duplicate_sample_id_examples": duplicate_ids,
    }


def embed(records: list[dict], backend: str, model_name: str) -> tuple[Any, str]:
    texts = [row["subject"] or row["src"] for row in records]
    if backend == "sentence-transformers":
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers is required for the default semantic embedding. "
                "Install requirements-zsre.txt, or use --embedding-backend tfidf only for an offline smoke test."
            ) from exc
        model = SentenceTransformer(model_name)
        return model.encode(texts, batch_size=64, show_progress_bar=True, normalize_embeddings=True), model_name
    if backend == "tfidf":
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
        except ImportError as exc:
            raise RuntimeError("scikit-learn is required for --embedding-backend tfidf") from exc
        # Explicitly opt-in fallback for CI/offline tests; production defaults to
        # a public sentence encoder above.
        matrix = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True).fit_transform(texts)
        # Keep the high-dimensional sparse matrix sparse. SpectralClustering's
        # nearest-neighbour affinity accepts it directly; densifying full zsRE
        # would consume impractical amounts of memory.
        return matrix, "sklearn.TfidfVectorizer(1,2)-offline-smoke"
    raise ValueError(f"Unknown embedding backend: {backend}")


def cluster(embeddings: Any, num_clients: int, seed: int, method: str) -> list[int]:
    try:
        from sklearn.cluster import MiniBatchKMeans, SpectralClustering
    except ImportError as exc:
        raise RuntimeError("scikit-learn is required for spectral clustering; install requirements-zsre.txt") from exc
    # NumPy sentence embeddings and scipy's sparse TF-IDF matrix both expose
    # rows through ``shape``; ``len(sparse_matrix)`` is intentionally invalid.
    sample_count = int(embeddings.shape[0])
    if sample_count < num_clients:
        raise ValueError(f"Need at least {num_clients} valid examples, found {sample_count}")
    if method == "spectral":
        n_neighbors = min(10, sample_count - 1)
        estimator = SpectralClustering(
            n_clusters=num_clients,
            affinity="nearest_neighbors",
            n_neighbors=n_neighbors,
            assign_labels="kmeans",
            n_init=10,
            random_state=seed,
        )
    elif method == "minibatch-kmeans":
        # On high-cardinality entity names, a sparse nearest-neighbour graph
        # can fragment and make spectral clustering collapse into a giant
        # component. This is the scalable semantic-clustering alternative:
        # the same public sentence embeddings are grouped directly, with no
        # label balancing or task-aware post-processing.
        estimator = MiniBatchKMeans(
            n_clusters=num_clients,
            batch_size=min(1024, sample_count),
            n_init=20,
            max_iter=300,
            random_state=seed,
        )
    else:
        raise ValueError(f"Unknown clustering method: {method}")
    return [int(value) for value in estimator.fit_predict(embeddings)]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def summarize_lengths(records: list[dict], field: str) -> dict:
    values = [len(row[field].split()) for row in records]
    return {"average_words": sum(values) / len(values), "median_words": statistics.median(values)}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_raw_path(raw_dir: Path, requested: str | None) -> Path:
    if requested:
        path = raw_dir / requested
        if not path.is_file():
            raise FileNotFoundError(path)
        return path
    for name in RAW_CANDIDATES:
        path = raw_dir / name
        if path.is_file():
            return path
    raise FileNotFoundError(f"No zsRE train file found in {raw_dir}; expected one of {RAW_CANDIDATES}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw_dir", default="dataset/zsre/raw")
    parser.add_argument("--output_dir", default="dataset/zsre")
    parser.add_argument("--raw_file", default=None, help="override raw train filename within --raw_dir")
    parser.add_argument("--num_clients", type=int, default=8)
    parser.add_argument("--samples_per_client", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--embedding-backend", choices=("sentence-transformers", "tfidf"), default="sentence-transformers")
    parser.add_argument(
        "--clustering-method", choices=("minibatch-kmeans", "spectral"), default="minibatch-kmeans",
        help="semantic clustering method; spectral is retained as an explicit reproducible alternative",
    )
    parser.add_argument(
        "--max-clustering-candidates", type=int, default=10_000,
        help=(
            "deterministic valid-record candidate cap before embedding/clustering; "
            "0 uses every valid raw record. The default matches the common zsRE 10k protocol "
            "and keeps spectral clustering practical."
        ),
    )
    args = parser.parse_args()
    if args.num_clients < 2 or args.samples_per_client < 1:
        raise ValueError("--num_clients must be >= 2 and --samples_per_client must be positive")

    raw_path = resolve_raw_path(Path(args.raw_dir), args.raw_file)
    records, cleaning = load_records(raw_path)
    if args.max_clustering_candidates < 0:
        raise ValueError("--max-clustering-candidates must be non-negative")
    candidate_records = records
    if args.max_clustering_candidates:
        if args.max_clustering_candidates < args.num_clients:
            raise ValueError("--max-clustering-candidates must be 0 or at least --num_clients")
        if len(records) > args.max_clustering_candidates:
            chosen = random.Random(args.seed).sample(range(len(records)), args.max_clustering_candidates)
            candidate_records = sorted((records[index] for index in chosen), key=lambda row: row["raw_index"])
            print(
                f"Using explicit deterministic candidate pool of {len(candidate_records)} / {len(records)} valid records "
                "before semantic clustering."
            )
    embeddings, embedding_description = embed(candidate_records, args.embedding_backend, args.embedding_model)
    labels = cluster(embeddings, args.num_clients, args.seed, args.clustering_method)
    clusters: dict[int, list[dict]] = {client_id: [] for client_id in range(args.num_clients)}
    for record, label in zip(candidate_records, labels):
        record["client_id"] = label
        clusters[label].append(record)
    cluster_sizes = {str(client_id): len(clusters[client_id]) for client_id in range(args.num_clients)}
    print("Cluster sizes before balancing:", cluster_sizes)
    minimum = min(cluster_sizes.values())
    if minimum == 0:
        raise RuntimeError("Spectral clustering produced an empty client; re-run after changing the embedding model")
    selected_per_client = min(args.samples_per_client, minimum)
    if selected_per_client != args.samples_per_client:
        print(
            f"Requested {args.samples_per_client}/client but smallest cluster has {minimum}; "
            f"using equal-size {selected_per_client}/client without oversampling."
        )

    output_dir = Path(args.output_dir)
    selected: dict[int, list[dict]] = {}
    for client_id, rows in clusters.items():
        indexes = list(range(len(rows)))
        # A separate deterministic stream makes an individual client stable if
        # another client's raw order changes, while never sampling with replacement.
        chosen = random.Random(args.seed + client_id).sample(indexes, selected_per_client)
        selected[client_id] = sorted((rows[index] for index in chosen), key=lambda row: row["raw_index"])

    mapping = []
    client_examples = {}
    selected_records = [row for client_rows in selected.values() for row in client_rows]
    for client_id, rows in selected.items():
        train_rows, test_rows, metadata_rows = [], [], []
        examples = []
        for row in rows:
            common = {"sample_id": row["sample_id"], "client_id": client_id}
            train_rows.append({**common, "input_ids": row["src"], "label": row["alt"]})
            test_rows.append({**common, "input_ids": row["rephrase"], "label": row["alt"]})
            metadata = {**row, "client_id": client_id}
            metadata_rows.append(metadata)
            mapping.append({"sample_id": row["sample_id"], "client_id": client_id, "raw_index": row["raw_index"]})
            if len(examples) < 3:
                examples.append({"sample_id": row["sample_id"], "subject": row["subject"], "src": row["src"]})
        write_jsonl(output_dir / "train" / f"{client_id}.jsonl", train_rows)
        write_jsonl(output_dir / "test" / f"{client_id}.jsonl", test_rows)
        write_jsonl(output_dir / "metadata" / f"{client_id}.jsonl", metadata_rows)
        client_examples[str(client_id)] = examples
    mapping.sort(key=lambda item: item["sample_id"])
    (output_dir / "client_mapping.json").write_text(
        json.dumps(mapping, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    stats = {
        "dataset": "zsRE counterfactual factual-knowledge client construction",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "raw_file": str(raw_path),
        "raw_sha256": sha256(raw_path),
        "seed": args.seed,
        "num_clients": args.num_clients,
        "requested_samples_per_client": args.samples_per_client,
        "selected_samples_per_client": selected_per_client,
        "total_selected": selected_per_client * args.num_clients,
        "clustering_candidate_pool": {
            "valid_records_available": len(records),
            "records_clustered": len(candidate_records),
            "max_clustering_candidates": args.max_clustering_candidates,
            "selection": "all valid records" if len(candidate_records) == len(records) else "deterministic random sample without replacement",
        },
        "clustering_method": (
            "SpectralClustering(affinity=nearest_neighbors, assign_labels=kmeans)"
            if args.clustering_method == "spectral"
            else "MiniBatchKMeans(n_init=20, max_iter=300) on semantic sentence embeddings"
        ),
        "embedding_backend": args.embedding_backend,
        "embedding_model": embedding_description,
        "representation": "subject when available, otherwise src/question",
        "cluster_sizes_before_balancing": cluster_sizes,
        "cluster_examples": client_examples,
        "cleaning": cleaning,
        "duplicate_statistics": {
            "content_duplicates_removed": cleaning["duplicate_content_examples"],
            "sample_id_duplicates_removed": cleaning["duplicate_sample_id_examples"],
        },
        "original_answer_differs_from_alt": {
            "valid_pool_count_with_original_answer": sum(bool(row["original_answer"]) for row in records),
            "valid_pool_count_different": sum(bool(row["original_answer"]) and _norm(row["original_answer"]) != _norm(row["alt"]) for row in records),
            "selected_count_with_original_answer": sum(bool(row["original_answer"]) for row in selected_records),
            "selected_count_different": sum(bool(row["original_answer"]) and _norm(row["original_answer"]) != _norm(row["alt"]) for row in selected_records),
        },
        "question_length": {"src": summarize_lengths(records, "src"), "rephrase": summarize_lengths(records, "rephrase")},
        "answer_length": summarize_lengths(records, "alt"),
        "schema": {
            "train": "input_ids=src, label=alt",
            "test": "input_ids=rephrase, label=alt",
            "metadata": "sample_id, subject, src, rephrase, alt, original_answer, client_id, loc, loc_ans",
        },
    }
    (output_dir / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {selected_per_client} train/test knowledge items for each of {args.num_clients} clients to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
