#!/usr/bin/env python3
"""Train six CNN/BiLSTM authorship experiments on prepared JSONL splits.

Python 3.10+. Example:
    python train.py --data-dir data/prepared --output-dir runs

Each split row must contain id, label, model, keyword, category, input, output.
The preparation script supplies metadata.json with ordered label_names.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
import os
from pathlib import Path
import random
import re
import time
import unicodedata

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence
from torch.utils.data import DataLoader, Dataset


PAD, UNK, SEP = "<pad>", "<unk>", "<sep>"
MODES = ("input", "output", "input_output")


def tokenize(text: str) -> list[str]:
    """Tokenize lowercase text into Unicode words and punctuation."""
    return re.findall(r"\w+|[^\w\s]", text.lower(), flags=re.UNICODE)


def record_tokens(record: dict, mode: str) -> list[str]:
    if mode == "input":
        return tokenize(record["input"])
    if mode == "output":
        return tokenize(record["output"])
    if mode == "input_output":
        return tokenize(record["input"]) + tokenize(record["output"])
    raise ValueError(f"Unknown input mode: {mode}")


def build_vocab(records: list[dict], mode: str, max_vocab: int = 30000,
                min_freq: int = 2) -> dict[str, int]:
    """Build from training records; max_vocab includes three special tokens."""
    if max_vocab < 4 or min_freq < 1:
        raise ValueError("max_vocab must be >= 4 and min_freq >= 1")
    counts: Counter = Counter()
    for record in records:
        counts.update(record_tokens(record, mode))
    vocab = {PAD: 0, UNK: 1, SEP: 2}
    candidates = sorted(counts.items(), key=lambda pair: (-pair[1], pair[0]))
    for token, count in candidates:
        if count < min_freq or len(vocab) >= max_vocab:
            break
        if token not in vocab:
            vocab[token] = len(vocab)
    return vocab


def encode_record(record: dict, vocab: dict[str, int], mode: str,
                  max_input: int = 128, max_output: int = 384) -> list[int]:
    if max_input < 1 or max_output < 1:
        raise ValueError("Input and output token budgets must be positive")
    if mode == "input":
        tokens = tokenize(record["input"])[:max_input]
    elif mode == "output":
        tokens = tokenize(record["output"])[:max_output]
    elif mode == "input_output":
        # Separate budgets ensure a long prompt cannot remove the response.
        tokens = tokenize(record["input"])[:max_input] + [SEP]
        tokens += tokenize(record["output"])[:max_output]
    else:
        raise ValueError(f"Unknown input mode: {mode}")
    return [vocab.get(token, vocab[UNK]) for token in tokens] or [vocab[UNK]]


class EncodedDataset(Dataset):
    def __init__(self, records: list[dict], vocab: dict[str, int], mode: str,
                 max_input: int, max_output: int):
        self.items = [(encode_record(r, vocab, mode, max_input, max_output),
                       r["label"], i) for i, r in enumerate(records)]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


def collate_batch(batch):
    encoded, labels, indices = zip(*batch)
    lengths = torch.tensor([len(x) for x in encoded], dtype=torch.long)
    tokens = pad_sequence([torch.tensor(x, dtype=torch.long) for x in encoded],
                          batch_first=True, padding_value=0)
    return tokens, lengths, torch.tensor(labels), torch.tensor(indices)


class TextCNN(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int = 128,
                 hidden_dim: int = 128, num_classes: int = 3, dropout: float = 0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.kernels = (3, 4, 5)
        self.convs = nn.ModuleList([
            nn.Conv1d(embedding_dim, hidden_dim, k) for k in self.kernels
        ])
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim * len(self.kernels), num_classes)

    def forward(self, tokens, lengths):
        if tokens.size(1) < max(self.kernels):
            tokens = F.pad(tokens, (0, max(self.kernels) - tokens.size(1)))
        embedded = self.embedding(tokens).transpose(1, 2)
        pooled = []
        for kernel, conv in zip(self.kernels, self.convs):
            features = F.relu(conv(embedded))
            # Full windows only; very short examples get one padded window.
            valid_windows = (lengths.to(tokens.device) - kernel + 1).clamp(min=1)
            positions = torch.arange(features.size(2), device=tokens.device)
            mask = positions.unsqueeze(0) < valid_windows.unsqueeze(1)
            features = features.masked_fill(~mask.unsqueeze(1), float("-inf"))
            pooled.append(features.amax(dim=2))
        return self.classifier(self.dropout(torch.cat(pooled, dim=1)))


class BiLSTM(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int = 128,
                 hidden_dim: int = 128, num_classes: int = 3, dropout: float = 0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True,
                            bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(2 * hidden_dim, num_classes)

    def forward(self, tokens, lengths):
        packed = pack_padded_sequence(self.embedding(tokens), lengths.cpu(),
                                      batch_first=True, enforce_sorted=False)
        _, (hidden, _) = self.lstm(packed)
        features = torch.cat((hidden[-2], hidden[-1]), dim=1)
        return self.classifier(self.dropout(features))


def set_seed(seed: int):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def seed_worker(_worker_id):
    worker_seed = torch.initial_seed() % (2 ** 32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def read_data(data_dir: Path):
    metadata = json.loads((data_dir / "metadata.json").read_text(encoding="utf-8"))
    names = metadata["label_names"]
    if not isinstance(names, list) or len(names) < 2 or len(set(names)) != len(names):
        raise ValueError("metadata.label_names must list at least two unique classes")
    splits = {}
    all_ids = set()
    for split in ("train", "validation", "test"):
        rows = []
        with (data_dir / f"{split}.jsonl").open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                for field in ("id", "model", "keyword", "category", "input", "output"):
                    if not isinstance(row.get(field), str):
                        raise ValueError(f"{split}:{line_number}: {field} must be a string")
                label = row.get("label")
                if type(label) is not int or not 0 <= label < len(names):
                    raise ValueError(f"{split}:{line_number}: invalid label {label!r}")
                if row["model"] != names[label]:
                    raise ValueError(f"{split}:{line_number}: model does not match its label")
                if not row["input"].strip() or not row["output"].strip():
                    raise ValueError(f"{split}:{line_number}: input/output cannot be empty")
                if row["id"] in all_ids:
                    raise ValueError(f"Repeated id across dataset: {row['id']}")
                all_ids.add(row["id"])
                rows.append(row)
        if not rows:
            raise ValueError(f"Empty {split} split")
        if {r["label"] for r in rows} != set(range(len(names))):
            raise ValueError(f"{split} split must contain every class")
        splits[split] = rows
    # Match preparation's normalization without relying on its CLI/module import.
    for field in ("keyword", "input", "output"):
        seen = {}
        for split, rows in splits.items():
            for row in rows:
                value = " ".join(unicodedata.normalize("NFKC", row[field]).casefold().split())
                if value in seen and seen[value] != split:
                    raise ValueError(f"Cross-split duplicate {field}: {seen[value]} and {split}")
                seen[value] = split
    return splits, names, metadata


def write_json(path: Path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def tokenization_statistics(splits, max_input, max_output):
    """Describe truncation without fitting any statistics used by the models."""
    stats = {"max_input": max_input, "max_output": max_output,
             "max_combined_tokens": max_input + 1 + max_output, "splits": {}}
    for split, rows in splits.items():
        summary = {"examples": len(rows)}
        for field, limit in (("input", max_input), ("output", max_output)):
            lengths = np.asarray([len(tokenize(row[field])) for row in rows])
            truncated = int((lengths > limit).sum())
            summary[field] = {"mean_tokens": float(lengths.mean()),
                              "median_tokens": float(np.median(lengths)),
                              "p95_tokens": float(np.quantile(lengths, 0.95)),
                              "max_tokens": int(lengths.max()),
                              "truncated_examples": truncated,
                              "truncation_fraction": truncated / len(rows)}
        stats["splits"][split] = summary
    return stats


def score_predictions(labels, predictions, names):
    classes = list(range(len(names)))
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, labels=classes,
                                   average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(labels, predictions, labels=classes,
                                      average="weighted", zero_division=0)),
        "classification_report": classification_report(
            labels, predictions, labels=classes, target_names=names,
            output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(labels, predictions, labels=classes).tolist(),
    }


@torch.no_grad()
def evaluate(model, loader, device, names):
    model.eval()
    truth, predicted, probabilities, indices = [], [], [], []
    total_loss = 0.0
    for tokens, lengths, labels, row_indices in loader:
        logits = model(tokens.to(device), lengths)
        total_loss += F.cross_entropy(logits, labels.to(device), reduction="sum").item()
        truth.extend(labels.tolist())
        predicted.extend(logits.argmax(dim=1).cpu().tolist())
        probabilities.extend(logits.softmax(dim=1).cpu().tolist())
        indices.extend(row_indices.tolist())
    metrics = score_predictions(truth, predicted, names)
    metrics["loss"] = total_loss / len(truth)
    return metrics, predicted, probabilities, indices


def save_confusion_plot(matrix, names, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matrix = np.asarray(matrix)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(im, ax=ax)
    ax.set(xticks=range(len(names)), yticks=range(len(names)),
           xticklabels=names, yticklabels=names, xlabel="Predicted model",
           ylabel="Actual model", title="Test confusion matrix")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center",
                    color="white" if matrix[i, j] > matrix.max() / 2 else "black")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def make_loader(dataset, args, shuffle=False):
    generator = torch.Generator().manual_seed(args.seed)
    return DataLoader(dataset, batch_size=args.batch_size, shuffle=shuffle,
                      num_workers=args.num_workers, collate_fn=collate_batch,
                      worker_init_fn=seed_worker, generator=generator,
                      pin_memory=args.device == "cuda")


def run_experiment(model_name, mode, datasets, vocab, splits, names, metadata, args):
    set_seed(args.seed)
    run_dir = args.output_dir / f"{model_name}_{mode}"
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = run_dir / "best.pt"
    if checkpoint_path.exists() and not args.overwrite:
        raise FileExistsError(f"{checkpoint_path} exists; use --overwrite or a new output directory")
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update(model=model_name, mode=mode, vocab_size=len(vocab),
                  num_classes=len(names), label_names=names,
                  split_sizes={s: len(rows) for s, rows in splits.items()},
                  pytorch_version=str(torch.__version__))
    write_json(run_dir / "config.json", config)
    write_json(run_dir / "vocab.json", vocab)
    write_json(run_dir / "data_metadata.json", metadata)
    model_class = TextCNN if model_name == "cnn" else BiLSTM
    model = model_class(len(vocab), args.embedding_dim, args.hidden_dim,
                        len(names), args.dropout).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate,
                                   weight_decay=args.weight_decay)
    train_loader = make_loader(datasets["train"], args, shuffle=True)
    validation_loader = make_loader(datasets["validation"], args)
    history, best_key, bad_epochs = [], (-1.0, float("-inf")), 0
    started = time.monotonic()
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum, correct, count = 0.0, 0, 0
        for tokens, lengths, labels, _ in train_loader:
            labels = labels.to(args.device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(tokens.to(args.device), lengths)
            loss = F.cross_entropy(logits, labels)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite training loss")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            loss_sum += loss.item() * len(labels)
            correct += (logits.argmax(1) == labels).sum().item()
            count += len(labels)
        validation, _, _, _ = evaluate(model, validation_loader, args.device, names)
        history.append({"epoch": epoch, "train_loss": loss_sum / count,
                        "train_accuracy": correct / count,
                        "validation_loss": validation["loss"],
                        "validation_accuracy": validation["accuracy"],
                        "validation_macro_f1": validation["macro_f1"]})
        print(f"{model_name}/{mode} epoch={epoch} loss={loss_sum/count:.4f} "
              f"val_macro_f1={validation['macro_f1']:.4f}", flush=True)
        candidate = (validation["macro_f1"], -validation["loss"])
        if candidate > best_key:
            best_key, bad_epochs = candidate, 0
            torch.save({"format_version": 1, "model_state": model.state_dict(),
                        "config": config, "vocab": vocab, "label_names": names,
                        "best_epoch": epoch, "validation_metrics": validation}, checkpoint_path)
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                break
    # The test split is evaluated once, after validation selects the checkpoint.
    checkpoint = torch.load(checkpoint_path, map_location=args.device, weights_only=True)
    model.load_state_dict(checkpoint["model_state"])
    test, predictions, probabilities, indices = evaluate(
        model, make_loader(datasets["test"], args), args.device, names)
    elapsed = time.monotonic() - started
    metrics = {"model": model_name, "mode": mode, "seed": args.seed,
               "best_epoch": checkpoint["best_epoch"], "epochs_run": len(history),
               "validation": checkpoint["validation_metrics"], "test": test,
               "elapsed_seconds": elapsed}
    write_json(run_dir / "metrics.json", metrics)
    with (run_dir / "history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    with (run_dir / "test_predictions.jsonl").open("w", encoding="utf-8") as handle:
        for prediction, probability, index in zip(predictions, probabilities, indices):
            row = splits["test"][index]
            output = {"id": row["id"], "keyword": row["keyword"],
                      "label": row["label"], "model": names[row["label"]],
                      "prediction": prediction, "predicted_model": names[prediction],
                      "probabilities": probability}
            handle.write(json.dumps(output, ensure_ascii=False) + "\n")
    save_confusion_plot(test["confusion_matrix"], names, run_dir / "confusion_matrix.png")
    print(f"{model_name}/{mode} TEST accuracy={test['accuracy']:.4f} "
          f"macro_f1={test['macro_f1']:.4f}", flush=True)
    return {"model": model_name, "mode": mode, "seed": args.seed,
            "best_epoch": checkpoint["best_epoch"], "epochs_run": len(history),
            "validation_macro_f1": checkpoint["validation_metrics"]["macro_f1"],
            "test_accuracy": test["accuracy"], "test_macro_f1": test["macro_f1"],
            "test_weighted_f1": test["weighted_f1"], "elapsed_seconds": round(elapsed, 2)}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/prepared"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs"))
    parser.add_argument("--models", nargs="+", choices=("cnn", "lstm"), default=["cnn", "lstm"])
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--max-vocab", type=int, default=30000)
    parser.add_argument("--min-freq", type=int, default=2)
    parser.add_argument("--max-input", type=int, default=128)
    parser.add_argument("--max-output", type=int, default=384)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    for key in ("epochs", "batch_size", "embedding_dim", "hidden_dim", "patience",
                "min_freq", "max_input", "max_output", "cpu_threads"):
        if getattr(args, key) < 1:
            parser.error(f"--{key.replace('_', '-')} must be positive")
    if not 0 <= args.dropout < 1 or args.max_vocab < 4 or args.num_workers < 0:
        parser.error("Require 0 <= dropout < 1, max-vocab >= 4, num-workers >= 0")
    if args.learning_rate <= 0 or args.weight_decay < 0 or args.grad_clip <= 0:
        parser.error("Require positive learning-rate/grad-clip and nonnegative weight-decay")
    if not 0 <= args.seed < 2 ** 32:
        parser.error("seed must lie in [0, 2**32)")
    args.models = list(dict.fromkeys(args.models))
    args.modes = list(dict.fromkeys(args.modes))
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable; use --device cpu")
    return args


def main(argv=None):
    args = parse_args(argv)
    torch.set_num_threads(args.cpu_threads)
    splits, names, metadata = read_data(args.data_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    # Check before training so a later experiment cannot unexpectedly collide.
    for model in args.models:
        for mode in args.modes:
            checkpoint = args.output_dir / f"{model}_{mode}" / "best.pt"
            if checkpoint.exists() and not args.overwrite:
                raise FileExistsError(f"{checkpoint} exists; choose a new output directory or --overwrite")
    write_json(args.output_dir / "tokenization_stats.json",
               tokenization_statistics(splits, args.max_input, args.max_output))
    results = []
    print(f"Device: {args.device}; classes: {names}; selection: validation macro-F1 "
          "(validation loss breaks ties)", flush=True)
    for mode in args.modes:
        vocab = build_vocab(splits["train"], mode, args.max_vocab, args.min_freq)
        datasets = {s: EncodedDataset(rows, vocab, mode, args.max_input, args.max_output)
                    for s, rows in splits.items()}
        for model in args.models:
            results.append(run_experiment(model, mode, datasets, vocab, splits,
                                           names, metadata, args))
            with (args.output_dir / "results.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(results[0]))
                writer.writeheader()
                writer.writerows(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
