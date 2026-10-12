"""Fit four source-only classifiers and evaluate subject-domain transfer.

Run from the package directory:
    python rq3_experiments.py --data-dir data/rq3 --output-dir runs/rq3

This is an exploratory extension because the original project test results were
already inspected. The RQ3 keyword map and training protocol are fixed before
these new fits. This tests subject domains within one scientific-ideation task.
"""
from __future__ import annotations

import argparse
from copy import copy
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import torch

from prepare_data import assert_no_leakage, file_sha256
from prepare_rq3 import DOMAINS, summary
import train


def read_domain(data_dir: Path):
    splits, names, metadata = train.read_data(data_dir)
    target_rows = [json.loads(line) for line in (data_dir / "target_test.jsonl").read_text(
        encoding="utf-8").splitlines() if line.strip()]
    summary(target_rows, names)
    ids = {row["id"] for rows in splits.values() for row in rows}
    for row in target_rows:
        if row["id"] in ids:
            raise ValueError("Target ID overlaps source data or repeats in target test")
        ids.add(row["id"])
        if type(row["label"]) is not int or not 0 <= row["label"] < len(names):
            raise ValueError("Target label is invalid")
        if row["model"] != names[row["label"]]:
            raise ValueError("Target model and label disagree")
    assert_no_leakage({**splits, "target_test": target_rows})
    return splits, target_rows, names, metadata


def evaluate_target(checkpoint_path: Path, rows: list[dict], device: str = "cpu") -> tuple:
    """Apply a saved source checkpoint with its original vocabulary unchanged."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    config, vocab, names = checkpoint["config"], checkpoint["vocab"], checkpoint["label_names"]
    if config["mode"] != "output":
        raise ValueError("RQ3 requires a response-only source checkpoint")
    if config["model"] not in ("cnn", "lstm"):
        raise ValueError("Unsupported source architecture")
    args = argparse.Namespace(**config)
    args.device = device
    architecture = train.TextCNN if config["model"] == "cnn" else train.BiLSTM
    model = architecture(len(vocab), config["embedding_dim"], config["hidden_dim"],
                         len(names), config["dropout"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    dataset = train.EncodedDataset(rows, vocab, "output", config["max_input"], config["max_output"])
    metrics, predictions, probabilities, indices = train.evaluate(
        model, train.make_loader(dataset, args), device, names)
    records = []
    for prediction, probability, index in zip(predictions, probabilities, indices):
        row = rows[index]
        records.append({
            "id": row["id"], "keyword": row["keyword"], "label": row["label"],
            "model": names[row["label"]], "prediction": prediction,
            "predicted_model": names[prediction], "probabilities": probability,
        })
    return metrics, records


def write_results(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_training_args(data_dir: Path, output_dir: Path, cpu_threads: int):
    # Reuse the original command's defaults so this is the same model recipe.
    return train.parse_args([
        "--data-dir", str(data_dir), "--output-dir", str(output_dir),
        "--models", "cnn", "lstm", "--modes", "output", "--device", "cpu",
        "--cpu-threads", str(cpu_threads),
    ])


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/rq3"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/rq3"))
    parser.add_argument("--cpu-threads", type=int, default=4)
    args = parser.parse_args(argv)
    if args.cpu_threads < 1:
        parser.error("--cpu-threads must be positive")
    if (args.output_dir / "protocol.json").exists():
        parser.error("Output protocol already exists; use a fresh output directory.")
    for domain in DOMAINS:
        for architecture in ("cnn", "lstm"):
            if (args.output_dir / domain / f"{architecture}_output" / "best.pt").exists():
                parser.error("A requested checkpoint already exists; use a fresh output directory.")
    torch.set_num_threads(args.cpu_threads)
    prepared = {domain: read_domain(args.data_dir / domain) for domain in DOMAINS}
    base_args = make_training_args(args.data_dir, args.output_dir, args.cpu_threads)
    protocol = {
        "experiment": "RQ3 keyword-defined subject-domain transfer",
        "status": "Frozen before the four RQ3 fits and evaluations",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "exploratory_extension": True,
        "exploratory_reason": "Original project test results were previously inspected; these are new predefined extension experiments, not a pristine confirmatory test.",
        "source_domains": list(DOMAINS),
        "architectures": ["cnn", "lstm"],
        "mode": "output",
        "fit_and_vocabulary": "source-domain original train only",
        "selection": "source-domain original validation macro-F1; validation loss breaks ties",
        "evaluation": "same selected checkpoint on source-domain test and other-domain target_test",
        "target_adaptation": "none",
        "interpretation": "Subject-domain transfer within the same scientific-ideation task, prompt, effort, and generator versions; not cross-task generalization.",
        "training_arguments": {k: str(v) if isinstance(v, Path) else v
                               for k, v in vars(base_args).items()},
        "files_sha256": {
            "train.py": file_sha256(Path(train.__file__)),
            "rq3_experiments.py": file_sha256(Path(__file__)),
            "manifest.json": file_sha256(args.data_dir / "manifest.json"),
            **{f"{domain}/{filename}": file_sha256(args.data_dir / domain / filename)
               for domain in DOMAINS
               for filename in ("train.jsonl", "validation.jsonl", "test.jsonl", "target_test.jsonl", "metadata.json")},
        },
        "split_sizes": {domain: {**{s: len(rs) for s, rs in prepared[domain][0].items()},
                                  "target_test": len(prepared[domain][1])}
                        for domain in DOMAINS},
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train.write_json(args.output_dir / "protocol.json", protocol)
    print("Frozen protocol saved before training; exploratory extension, seed 42.", flush=True)
    results = []
    for source_domain in DOMAINS:
        splits, target_rows, names, metadata = prepared[source_domain]
        target_domain = metadata["target_domain"]
        training_args = copy(base_args)
        training_args.data_dir = args.data_dir / source_domain
        training_args.output_dir = args.output_dir / source_domain
        training_args.output_dir.mkdir(parents=True, exist_ok=True)
        train.write_json(training_args.output_dir / "tokenization_stats.json",
                         train.tokenization_statistics({**splits, "target_test": target_rows},
                                                       training_args.max_input, training_args.max_output))
        vocab = train.build_vocab(splits["train"], "output", training_args.max_vocab, training_args.min_freq)
        datasets = {split: train.EncodedDataset(rows, vocab, "output", training_args.max_input,
                                                training_args.max_output)
                    for split, rows in splits.items()}
        for architecture in ("cnn", "lstm"):
            print(f"\nSOURCE={source_domain} TARGET={target_domain} MODEL={architecture}", flush=True)
            source = train.run_experiment(architecture, "output", datasets, vocab, splits,
                                          names, metadata, training_args)
            run_dir = training_args.output_dir / f"{architecture}_output"
            checkpoint = run_dir / "best.pt"
            before_hash = file_sha256(checkpoint)
            started = time.monotonic()
            target, predictions = evaluate_target(checkpoint, target_rows)
            target_seconds = time.monotonic() - started
            if file_sha256(checkpoint) != before_hash:
                raise RuntimeError("Source checkpoint changed during target evaluation")
            train.write_json(run_dir / "cross_domain_metrics.json", {
                "source_domain": source_domain, "target_domain": target_domain,
                "model": architecture, "mode": "output", "seed": 42,
                "best_epoch": source["best_epoch"], "checkpoint_sha256": before_hash,
                "target_test": target, "elapsed_seconds": target_seconds,
            })
            with (run_dir / "cross_domain_predictions.jsonl").open("w", encoding="utf-8") as stream:
                for row in predictions:
                    stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            train.save_confusion_plot(target["confusion_matrix"], names,
                                      run_dir / "cross_domain_confusion_matrix.png")
            row = {
                "source_domain": source_domain, "target_domain": target_domain,
                "model": architecture, "mode": "output", "seed": 42,
                "best_epoch": source["best_epoch"], "epochs_run": source["epochs_run"],
                "train_n": len(splits["train"]), "validation_n": len(splits["validation"]),
                "in_domain_n": len(splits["test"]), "target_n": len(target_rows),
                "vocab_size": len(vocab), "validation_macro_f1": source["validation_macro_f1"],
                "in_domain_accuracy": source["test_accuracy"],
                "in_domain_macro_f1": source["test_macro_f1"],
                "target_accuracy": target["accuracy"], "target_macro_f1": target["macro_f1"],
                "delta_accuracy_pp": 100 * (target["accuracy"] - source["test_accuracy"]),
                "delta_macro_f1_pp": 100 * (target["macro_f1"] - source["test_macro_f1"]),
                "fit_and_source_eval_seconds": source["elapsed_seconds"],
                "target_eval_seconds": round(target_seconds, 3),
            }
            results.append(row)
            write_results(args.output_dir / "results.csv", results)
            print(f"TARGET accuracy={target['accuracy']:.4f} macro_f1={target['macro_f1']:.4f} "
                  f"delta_f1_pp={row['delta_macro_f1_pp']:+.2f}", flush=True)
    print(f"\nCompleted {len(results)} fits; saved {args.output_dir / 'results.csv'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
