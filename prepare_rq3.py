"""Prepare keyword-defined subject-domain transfer experiments from fixed splits.

Run from the package directory:
    python prepare_rq3.py --data-dir data/prepared --output-dir data/rq3

Each source-domain directory contains train, validation, and in-domain test
JSONL files, plus target_test.jsonl for the other domain's held-out examples.
Only the source train split may be used to build a vocabulary or fit a model.
The source validation split selects the checkpoint. Both tests are evaluation
only; no target-domain examples are used for fitting or model selection.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

from prepare_data import assert_no_leakage, file_sha256, normalize_text


SPLITS = ("train", "validation", "test")
DOMAINS = ("life_sciences", "physical_sciences")


def read_fixed_splits(data_dir: Path) -> tuple[dict, dict]:
    metadata = json.loads((data_dir / "metadata.json").read_text(encoding="utf-8"))
    labels = metadata["label_names"]
    splits = {}
    ids = set()
    for split in SPLITS:
        rows = [json.loads(line) for line in (data_dir / f"{split}.jsonl").read_text(
            encoding="utf-8").splitlines() if line.strip()]
        if not rows:
            raise ValueError(f"Empty source split: {split}")
        for row in rows:
            if row["id"] in ids:
                raise ValueError(f"Repeated example ID: {row['id']}")
            ids.add(row["id"])
            if type(row["label"]) is not int or not 0 <= row["label"] < len(labels):
                raise ValueError("Invalid label")
            if row["model"] != labels[row["label"]]:
                raise ValueError("Model and label do not match")
        splits[split] = rows
    assert_no_leakage(splits)
    return splits, metadata


def summary(rows: list[dict], labels: list[str]) -> dict:
    counts = Counter(row["model"] for row in rows)
    if set(counts) != set(labels) or len(set(counts.values())) != 1:
        raise ValueError(f"Unbalanced or missing model classes: {counts}")
    keyword_counts = defaultdict(Counter)
    for row in rows:
        keyword_counts[row["keyword"]][row["model"]] += 1
    for keyword, values in keyword_counts.items():
        if set(values) != set(labels) or len(set(values.values())) != 1:
            raise ValueError(f"Unbalanced keyword: {keyword}")
    return {
        "rows": len(rows),
        "class_counts": {name: counts[name] for name in labels},
        "keyword_count": len(keyword_counts),
        "keywords": sorted(keyword_counts),
        "balanced_within_each_keyword": True,
    }


def prepare_domains(splits: dict, source_metadata: dict, mapping: dict) -> tuple[dict, dict]:
    labels = source_metadata["label_names"]
    all_keywords = {row["keyword"] for rows in splits.values() for row in rows}
    assigned = {}
    seen_normalized = set()
    for domain in DOMAINS:
        keywords = mapping["domains"][domain]["keywords"]
        for keyword in keywords:
            normalized = normalize_text(keyword)
            if normalized in seen_normalized:
                raise ValueError(f"Repeated or overlapping mapped keyword: {keyword}")
            if keyword not in all_keywords:
                raise ValueError(f"Mapped keyword missing from prepared dataset: {keyword}")
            seen_normalized.add(normalized)
            assigned[keyword] = domain

    for example in mapping.get("metadata_category_audit", []):
        categories = {row["category"] for rows in splits.values() for row in rows
                      if row["keyword"] == example["keyword"]}
        if categories != {example["provided_category"]}:
            raise ValueError(f"Category audit no longer matches: {example['keyword']}")

    filtered = {
        domain: {split: [row for row in rows if assigned.get(row["keyword"]) == domain]
                 for split, rows in splits.items()}
        for domain in DOMAINS
    }
    data, experiment_metadata = {}, {}
    source_fingerprints = {split: {row["id"]: row for row in rows}
                           for split, rows in splits.items()}
    for domain in DOMAINS:
        target_domain = next(other for other in DOMAINS if other != domain)
        prepared = {**filtered[domain], "target_test": filtered[target_domain]["test"]}
        assert_no_leakage(prepared)
        summaries = {split: summary(rows, labels) for split, rows in prepared.items()}
        for split, rows in prepared.items():
            original_split = "test" if split == "target_test" else split
            for row in rows:
                if source_fingerprints[original_split].get(row["id"]) != row:
                    raise ValueError("A row changed or moved between original splits")
        data[domain] = prepared
        experiment_metadata[domain] = {
            "experiment": "RQ3 subject-domain transfer within scientific ideation",
            "label_names": labels,
            "source_domain": domain,
            "target_domain": target_domain,
            "dataset": source_metadata.get("dataset"),
            "requested_dataset_revision": source_metadata.get("requested_dataset_revision"),
            "prompt_code_revision": source_metadata.get("prompt_code_revision"),
            "data_license": source_metadata.get("data_license"),
            "effort": source_metadata.get("effort"),
            "prompt_style": source_metadata.get("prompt_style"),
            "splits": summaries,
            "original_split_assignments_preserved": True,
            "vocabulary_and_training": "source-domain train only",
            "checkpoint_selection": "source-domain validation only",
            "evaluation": "test is source-domain; target_test is the other domain's original test",
            "leakage_checks": {
                "original_ids_unique": True,
                "original_rows_unchanged": True,
                "pairwise_disjoint_keywords": True,
                "pairwise_disjoint_normalized_inputs": True,
                "pairwise_disjoint_normalized_outputs": True,
                "all_splits_class_balanced": True,
            },
            "limitations": [
                mapping["limitation"],
                "Domain assignment follows the requested keyword; an idea may discuss other fields.",
                "Only exact normalized text duplicates are excluded; related concepts can remain.",
                "Source sample sizes differ by direction; compare each source and target test within its fitted model.",
            ],
        }
    manifest = {
        "domain_mapping": mapping,
        "original_rows": sum(len(rows) for rows in splits.values()),
        "original_keywords": len(all_keywords),
        "selected_keywords": len(assigned),
        "excluded_keywords": sorted(all_keywords - set(assigned)),
        "experiments": experiment_metadata,
    }
    return data, manifest


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/prepared"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/rq3"))
    parser.add_argument("--mapping", type=Path,
                        default=Path(__file__).resolve().parent / "assets/rq3_domains.json")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    targets = [args.output_dir / "manifest.json"]
    for domain in DOMAINS:
        targets.extend(args.output_dir / domain / f"{split}.jsonl"
                       for split in (*SPLITS, "target_test"))
        targets.append(args.output_dir / domain / "metadata.json")
    if not args.overwrite and any(path.exists() for path in targets):
        parser.error("RQ3 files already exist; choose another output directory or use --overwrite.")
    splits, source_metadata = read_fixed_splits(args.data_dir)
    mapping = json.loads(args.mapping.read_text(encoding="utf-8"))
    data, manifest = prepare_domains(splits, source_metadata, mapping)
    provenance = {
        "prepared_file_sha256": {f"{split}.jsonl": file_sha256(args.data_dir / f"{split}.jsonl")
                                 for split in SPLITS},
        "prepared_metadata_sha256": file_sha256(args.data_dir / "metadata.json"),
        "domain_mapping_sha256": file_sha256(args.mapping),
    }
    manifest["provenance"] = provenance
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "manifest.json", manifest)
    for domain, prepared in data.items():
        directory = args.output_dir / domain
        directory.mkdir(parents=True, exist_ok=True)
        for split, rows in prepared.items():
            with (directory / f"{split}.jsonl").open("w", encoding="utf-8") as stream:
                for row in rows:
                    stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        metadata = {**manifest["experiments"][domain], "provenance": provenance}
        write_json(directory / "metadata.json", metadata)
        print(domain + ": " + ", ".join(f"{s}={len(r):,}" for s, r in prepared.items()))
    print(f"Selected {manifest['selected_keywords']} keywords; excluded {len(manifest['excluded_keywords'])}.")
    print("All original rows and split assignments preserved; leakage and class-balance checks passed.")


if __name__ == "__main__":
    main()
