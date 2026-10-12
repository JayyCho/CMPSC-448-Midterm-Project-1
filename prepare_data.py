"""Prepare a balanced, grouped-split scientific-ideation attribution dataset.

Only generated idea text becomes a model feature. API metadata, reasoning traces,
quality judgments, and provider-specific JSON formatting are deliberately excluded.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import random
import re
import unicodedata

DATASET_ID = "tax-free/scientific-ideation-diversity"
DATASET_REVISION = "a6f10e87dceae4e3ec1cfcf23d002dfb7d447d76"
PROMPT_REVISION = "bb5a2ab40c2e7f0795453c0851bc5b3d821c68b1"
LABEL_NAMES = ["claude-sonnet-4-6", "gpt-5-4", "gemini-3-1-pro"]
FILENAMES = {name: f"ideas/effort-axis/{name}.jsonl.gz" for name in LABEL_NAMES}


def normalize_text(text: str) -> str:
    """Conservative duplicate key: Unicode normalization, case, whitespace."""
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text).casefold()).strip()


def canonical_model(name: str) -> str:
    name = name.lower()
    if "claude-sonnet-4-6" in name:
        return LABEL_NAMES[0]
    if name == "gpt-5.4" or name.startswith("gpt-5.4-2026-"):
        return LABEL_NAMES[1]
    if name in {"gemini-3.1-pro", "gemini-3.1-pro-preview"}:
        return LABEL_NAMES[2]
    raise ValueError(f"Unexpected generator model: {name!r}")


def load_records(path: Path, effort: str) -> list[dict]:
    """Stream JSONL and retain fields needed for curation."""
    records = []
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
            if raw.get("effort") != effort or raw.get("prompt_style") != "structured_facet":
                continue
            records.append({key: raw.get(key) for key in (
                "idea_model", "keyword", "category", "sample_index", "idea",
                "effort", "prompt_style", "pydantic_validation_status",
            )})
    return records


def split_records(records: list[dict], seed: int = 42,
                  validation_fraction: float = 0.15,
                  test_fraction: float = 0.15) -> dict[str, list[dict]]:
    """Group keywords linked by normalized duplicate outputs.

    Ratios apply to components, not independent rows. All classes remain balanced
    because each keyword contributes the same number of responses per model.
    """
    if not 0 < validation_fraction < 1 or not 0 < test_fraction < 1:
        raise ValueError("Validation and test fractions must be between zero and one.")
    if validation_fraction + test_fraction >= 1:
        raise ValueError("Validation and test fractions must sum to less than one.")
    keywords = sorted({normalize_text(row["keyword"]) for row in records})
    parent = {key: key for key in keywords}

    def find(key):
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    def union(a, b):
        a, b = find(a), find(b)
        if a != b:
            parent[b] = a

    output_owner = {}
    for row in records:
        key = normalize_text(row["keyword"])
        text = normalize_text(row["output"])
        if text in output_owner:
            union(key, output_owner[text])
        else:
            output_owner[text] = key
    groups = defaultdict(list)
    for row in records:
        groups[find(normalize_text(row["keyword"]))].append(row)
    components = [groups[key] for key in sorted(groups)]
    rng = random.Random(seed)
    rng.shuffle(components)
    n = len(components)
    n_validation = max(1, round(n * validation_fraction))
    n_test = max(1, round(n * test_fraction))
    n_train = n - n_validation - n_test
    if n_train < 1:
        raise ValueError("Need at least three independent keyword/duplicate groups for splitting.")
    pieces = {
        "train": components[:n_train],
        "validation": components[n_train:n_train + n_validation],
        "test": components[n_train + n_validation:],
    }
    splits = {}
    for split, selected in pieces.items():
        rows = [row for component in selected for row in component]
        rng.shuffle(rows)
        splits[split] = rows
    assert_no_leakage(splits)
    return splits


def assert_no_leakage(splits: dict[str, list[dict]]) -> None:
    for field in ("keyword", "input", "output"):
        seen = {}
        for split, rows in splits.items():
            for row in rows:
                value = normalize_text(row[field])
                if value in seen and seen[value] != split:
                    raise ValueError(f"Cross-split duplicate {field}: {seen[value]} and {split}")
                seen[value] = split


def prepare_records(records_by_model: dict[str, list[dict]], prompt_template: str,
                    num_keywords: int = 300, samples_per_keyword: int = 10,
                    seed: int = 42, effort: str = "low",
                    validation_fraction: float = 0.15,
                    test_fraction: float = 0.15) -> tuple[dict, dict]:
    """Curate balanced records and split by keyword and duplicate text."""
    if num_keywords < 3 or samples_per_keyword < 1:
        raise ValueError("Use at least 3 keywords and at least 1 sample per keyword.")
    if "{{keywords}}" not in prompt_template:
        raise ValueError("Prompt template must contain the original {{keywords}} placeholder.")
    pools, cleaning = {}, {}
    for model in LABEL_NAMES:
        by_keyword = defaultdict(list)
        seen_indices, seen_text = set(), set()
        counts = Counter()
        for raw in records_by_model[model]:
            counts["rows_read"] += 1
            if raw.get("effort") != effort or raw.get("prompt_style") != "structured_facet":
                counts["wrong_setting"] += 1
                continue
            if canonical_model(str(raw.get("idea_model", ""))) != model:
                raise ValueError(f"Record generator does not match file for {model}.")
            keyword, text = raw.get("keyword"), raw.get("idea")
            if not isinstance(keyword, str) or not keyword.strip():
                counts["empty_keyword"] += 1
                continue
            if not isinstance(text, str) or not text.strip():
                counts["empty_output"] += 1
                continue
            status = raw.get("pydantic_validation_status")
            if status not in {None, "passed"}:
                counts["invalid_structured_output"] += 1
                continue
            index = raw.get("sample_index")
            if not isinstance(index, int) or isinstance(index, bool) or index < 0:
                counts["invalid_sample_index"] += 1
                continue
            key, duplicate_key = (keyword, index), (keyword, normalize_text(text))
            if key in seen_indices or duplicate_key in seen_text:
                counts["duplicate_within_keyword_model"] += 1
                continue
            seen_indices.add(key)
            seen_text.add(duplicate_key)
            by_keyword[keyword].append(raw)
            counts["accepted"] += 1
        pools[model] = by_keyword
        cleaning[model] = dict(counts)
    eligible = set.intersection(*[
        {keyword for keyword, rows in pools[model].items()
         if len(rows) >= samples_per_keyword} for model in LABEL_NAMES
    ])
    if len(eligible) < num_keywords:
        raise ValueError(
            f"Only {len(eligible)} shared keywords have >= {samples_per_keyword} unique "
            f"valid responses from every model; requested {num_keywords}. "
            "Reduce --num-keywords or --samples-per-keyword."
        )
    rng = random.Random(seed)
    selected_keywords = sorted(rng.sample(sorted(eligible), num_keywords))
    records = []
    for keyword in selected_keywords:
        for label, model in enumerate(LABEL_NAMES):
            candidates = sorted(pools[model][keyword], key=lambda row: row["sample_index"])
            for raw in rng.sample(candidates, samples_per_keyword):
                source_key = f"{model}|{keyword}|structured_facet|{effort}|{raw['sample_index']}"
                records.append({
                    "id": hashlib.sha256(source_key.encode()).hexdigest()[:24],
                    "label": label, "model": model, "keyword": keyword,
                    "category": raw.get("category") or "unknown",
                    "input": prompt_template.replace("{{keywords}}", keyword),
                    "output": raw["idea"].strip(),
                    "sample_index": raw["sample_index"],
                })
    splits = split_records(records, seed, validation_fraction, test_fraction)
    split_summary = {}
    for name, rows in splits.items():
        class_counts = Counter(row["model"] for row in rows)
        if len(class_counts) != 3 or len(set(class_counts.values())) != 1:
            raise ValueError(f"Class balance invariant failed in {name}.")
        split_summary[name] = {
            "rows": len(rows), "class_counts": dict(class_counts),
            "keywords": sorted({row["keyword"] for row in rows}),
            "category_counts": dict(Counter(row["category"] for row in rows)),
        }
    metadata = {
        "label_names": LABEL_NAMES, "seed": seed,
        "effort": effort, "prompt_style": "structured_facet",
        "num_keywords": num_keywords, "samples_per_keyword_per_model": samples_per_keyword,
        "eligible_shared_keywords": len(eligible), "total_rows": len(records),
        "cleaning": cleaning, "splits": split_summary,
        "split_method": "Connected components of keyword and normalized exact duplicate outputs",
        "requested_component_fractions": {
            "train": 1 - validation_fraction - test_fraction,
            "validation": validation_fraction, "test": test_fraction,
        },
        "input_only_accuracy_baseline": 1 / 3,
        "input_only_note": "Each identical prompt has equal counts for all three labels in every split.",
        "limitations": [
            "Keyword grouping and normalized exact deduplication do not detect all semantic paraphrases.",
            "This predicts three specific versions on scientific ideas, not all versions or domains.",
        ],
    }
    return splits, metadata


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/prepared"))
    parser.add_argument("--num-keywords", type=int, default=300)
    parser.add_argument("--samples-per-keyword", type=int, default=10)
    parser.add_argument("--effort", choices=["low", "medium", "high"], default="low")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--dataset-revision", default=DATASET_REVISION)
    parser.add_argument("--offline", action="store_true", help="Use local .jsonl.gz files only.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    targets = [args.output_dir / f"{split}.jsonl" for split in ("train", "validation", "test")]
    targets.append(args.output_dir / "metadata.json")
    if not args.overwrite and any(path.exists() for path in targets):
        parser.error("Prepared files already exist. Choose another output directory or use --overwrite.")
    asset = Path(__file__).resolve().parent / "assets/facet_idea_prompt.json"
    template = json.loads(asset.read_text())["idea_prompt"]["description"]
    records, provenance = {}, {}
    print("Reading only the three idea files (~1.12 GB compressed); embeddings are not needed.", flush=True)
    for model, relative_path in FILENAMES.items():
        candidates = [args.raw_dir / relative_path, args.raw_dir / Path(relative_path).name]
        path = next((candidate for candidate in candidates if candidate.is_file()), None)
        source = "existing_local_file"
        if path is None:
            if args.offline:
                parser.error(f"Missing {relative_path} under {args.raw_dir}")
            from huggingface_hub import hf_hub_download
            print(f"Downloading {relative_path} ...", flush=True)
            path = Path(hf_hub_download(
                DATASET_ID, relative_path, repo_type="dataset",
                revision=args.dataset_revision, local_dir=args.raw_dir,
            ))
            source = "huggingface_download_at_requested_revision"
        print(f"Reading {model} ...", flush=True)
        records[model] = load_records(path, args.effort)
        provenance[model] = {
            "file": relative_path, "source": source,
            "sha256": file_sha256(path), "compressed_bytes": path.stat().st_size,
        }
    splits, metadata = prepare_records(
        records, template, args.num_keywords, args.samples_per_keyword,
        args.seed, args.effort, args.validation_fraction, args.test_fraction,
    )
    metadata.update({
        "dataset": DATASET_ID, "requested_dataset_revision": args.dataset_revision,
        "files": provenance, "prompt_code_revision": PROMPT_REVISION,
        "prompt_template_sha256": file_sha256(asset),
        "data_license": "CC-BY-4.0", "prompt_code_license": "MIT",
    })
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in splits.items():
        path = args.output_dir / f"{name}.jsonl"
        temporary = path.with_suffix(".jsonl.tmp")
        with temporary.open("w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        temporary.replace(path)
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps({name: metadata["splits"][name]["class_counts"] for name in splits}, indent=2))
    print(f"Saved {metadata['total_rows']:,} examples to {args.output_dir}.")


if __name__ == "__main__":
    main()
