#!/usr/bin/env python3
"""Reproduce descriptive report tables and figures without training any model.

Run from the project directory:
  python analyze_results.py
  python analyze_results.py --data-dir data/prepared --runs-dir runs_new --output-dir runs_new/report_figures

Response-style measurements read only the selected data directory's train.jsonl. Saved validation
histories and test metrics/predictions are read solely to summarize evaluations.
Original data, code, and run artifacts are never modified.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import sys
import tempfile

HERE = Path(__file__).resolve().parent
PROJECT = HERE
# Importing the analyzer must not create files inside a saved run directory.
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "llm_authorship_matplotlib"))
sys.dont_write_bytecode = True

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, PercentFormatter
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

spec = importlib.util.spec_from_file_location("saved_train", PROJECT / "train.py")
train = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(train)

SOURCES: set[Path] = {PROJECT / "train.py"}


def read_json(path: Path):
    SOURCES.add(path)
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path):
    SOURCES.add(path)
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def save_figure(fig, name: str, destination: Path):
    fig.savefig(destination / f"{name}.png", dpi=200, facecolor="white", bbox_inches="tight")
    fig.savefig(destination / f"{name}.svg", facecolor="white", bbox_inches="tight")
    plt.close(fig)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=PROJECT / "data/prepared")
    parser.add_argument("--runs-dir", type=Path, default=PROJECT / "runs")
    parser.add_argument("--output-dir", type=Path,
                        help="Derived figures/statistics directory; defaults to <runs-dir>/report_figures.")
    args = parser.parse_args(argv)
    args.data_dir = args.data_dir.resolve()
    args.runs_dir = args.runs_dir.resolve()
    args.output_dir = (args.output_dir or args.runs_dir / "report_figures").resolve()
    return args


def validate_destination(destination: Path, sources: set[Path]):
    """Allow regenerating derived files, but never overwrite an input artifact."""
    if destination in {source.parent.resolve() for source in sources}:
        raise ValueError("Output directory must be separate from directories containing source artifacts")
    filenames = ["report_derived_statistics.json"] + [
        f"{name}.{extension}" for name in ("confusion_matrices", "response_lengths", "validation_curves")
        for extension in ("png", "svg")]
    for filename in filenames:
        target = destination / filename
        if target.exists() and any(target.samefile(source) for source in sources):
            raise ValueError(f"Derived output would overwrite a source artifact: {target}")


def provenance_key(path: Path, data_dir: Path, runs_dir: Path) -> str:
    """Use portable input-role paths even when inputs are outside this package."""
    if path == PROJECT / "train.py":
        return "code/train.py"
    for role, directory in (("data", data_dir), ("runs", runs_dir)):
        if path.is_relative_to(directory):
            return f"{role}/{path.relative_to(directory).as_posix()}"
    raise ValueError(f"Unrecognized analysis source: {path}")


def main(argv=None):
    args = parse_args(argv)
    SOURCES.clear()
    SOURCES.add(PROJECT / "train.py")
    metadata = read_json(args.data_dir / "metadata.json")
    names = metadata["label_names"]
    short_names = ["Claude Sonnet 4.6", "GPT-5.4", "Gemini 3.1 Pro"]
    assert names == ["claude-sonnet-4-6", "gpt-5-4", "gemini-3-1-pro"]
    records = read_jsonl(args.data_dir / "train.jsonl")
    by_model = {}
    lengths = []
    all_lengths = []
    for label, name in enumerate(names):
        rows = [r for r in records if r["model"] == name]
        assert len(rows) == metadata["splits"]["train"]["class_counts"][name]
        assert all(r["label"] == label for r in rows)
        n = len(rows)
        lens = np.array([len(train.tokenize(r["output"])) for r in rows])
        lengths.append(lens)
        all_lengths.extend(lens.tolist())
        colon_n = sum(":" in r["output"] for r in rows)
        semicolon_n = sum(";" in r["output"] for r in rows)
        both_n = sum(":" in r["output"] and ";" in r["output"] for r in rows)
        by_model[name] = {
            "n": n,
            "total_response_tokens": int(lens.sum()),
            "response_tokens": {
                "mean": float(lens.mean()),
                "std_sample": float(lens.std(ddof=1)),
                "min": int(lens.min()),
                "p05": float(np.percentile(lens, 5)),
                "q1": float(np.percentile(lens, 25)),
                "median": float(np.median(lens)),
                "q3": float(np.percentile(lens, 75)),
                "p95": float(np.percentile(lens, 95)),
                "max": int(lens.max()),
            },
            "colon": {"responses_with": colon_n, "percent": 100.0 * colon_n / n},
            "semicolon": {"responses_with": semicolon_n, "percent": 100.0 * semicolon_n / n},
            "both_colon_and_semicolon": {"responses_with": both_n, "percent": 100.0 * both_n / n},
            "responses_over_384_tokens": int((lens > 384).sum()),
        }
    assert len(records) == sum(d["n"] for d in by_model.values())

    experiments = {}
    for model in ("cnn", "lstm"):
        for mode in ("input", "output", "input_output"):
            key = f"{model}_{mode}"
            run = args.runs_dir / key
            metrics = read_json(run / "metrics.json")
            config = read_json(run / "config.json")
            vocab = read_json(run / "vocab.json")
            predictions = read_jsonl(run / "test_predictions.jsonl")
            history_path = run / "history.csv"
            SOURCES.add(history_path)
            with history_path.open(newline="", encoding="utf-8") as f:
                history = [{k: int(v) if k == "epoch" else float(v)
                            for k, v in row.items()} for row in csv.DictReader(f)]
            assert len(vocab) == config["vocab_size"]
            assert len(history) == metrics["epochs_run"]
            assert config["label_names"] == names
            gold = [p["label"] for p in predictions]
            pred = [p["prediction"] for p in predictions]
            recomputed_cm = confusion_matrix(gold, pred, labels=range(len(names)))
            assert recomputed_cm.tolist() == metrics["test"]["confusion_matrix"]
            report = classification_report(gold, pred, labels=range(len(names)),
                                           target_names=names, output_dict=True,
                                           zero_division=0)
            for name in names:
                for metric in ("precision", "recall", "f1-score", "support"):
                    assert np.isclose(report[name][metric], metrics["test"]["classification_report"][name][metric])
            assert np.isclose(report["macro avg"]["f1-score"], metrics["test"]["macro_f1"])
            assert np.isclose(report["accuracy"], metrics["test"]["accuracy"])
            best_row = next(h for h in history if h["epoch"] == metrics["best_epoch"])
            assert np.isclose(best_row["validation_macro_f1"], metrics["validation"]["macro_f1"])
            experiments[key] = {
                "model": model, "mode": mode,
                "seed": metrics["seed"],
                "vocab_size": config["vocab_size"],
                "best_epoch": metrics["best_epoch"],
                "epochs_run": metrics["epochs_run"],
                "elapsed_seconds": metrics["elapsed_seconds"],
                "elapsed_minutes": metrics["elapsed_seconds"] / 60,
                "config": config,
                "validation": metrics["validation"],
                "test": metrics["test"],
                "history": history,
                "audit": {"saved_test_predictions_reproduce_confusion_matrix": True,
                          "saved_test_predictions_reproduce_per_class_metrics": True,
                          "best_epoch_validation_f1_matches_saved_history": True},
            }

    validate_destination(args.output_dir, SOURCES)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10,
        "axes.titlesize": 10, "axes.labelsize": 10,
        "xtick.labelsize": 10, "ytick.labelsize": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#89949e", "axes.labelcolor": "#253442",
        "xtick.color": "#253442", "ytick.color": "#253442",
        "text.color": "#253442", "svg.fonttype": "none",
        "figure.facecolor": "white", "axes.facecolor": "white",
    })

    # Figure 1: cell annotations are counts; color is row-normalized percentage.
    order = ["cnn_output", "cnn_input_output", "lstm_output", "lstm_input_output"]
    titles = ["(a) CNN · response", "(b) CNN · combined",
              "(c) BiLSTM · response", "(d) BiLSTM · combined"]
    ticks = ["Claude", "GPT", "Gemini"]
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.0), layout="constrained")
    for ax, key, title in zip(axes.flat, order, titles):
        cm = np.array(experiments[key]["test"]["confusion_matrix"])
        pct = 100 * cm / cm.sum(axis=1, keepdims=True)
        im = ax.imshow(pct, cmap="Blues", vmin=0, vmax=100)
        for (r, c), count in np.ndenumerate(cm):
            ax.text(c, r, str(count), ha="center", va="center", fontsize=10,
                    color="white" if pct[r, c] > 55 else "#253442")
        ax.set(xticks=range(3), yticks=range(3), xticklabels=ticks, yticklabels=ticks,
               xlabel="Predicted model", ylabel="True model", title=title)
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
    cb = fig.colorbar(im, ax=axes, shrink=0.8, pad=0.025, aspect=32)
    cb.set_label("Share of true class (%)")
    save_figure(fig, "confusion_matrices", args.output_dir)

    # Figure 2: empirical lengths use the exact training tokenizer, without truncation.
    fig, ax = plt.subplots(figsize=(7.0, 2.7), layout="constrained")
    colors = ["#6F8DA6", "#5F9B8A", "#B69A62"]
    boxplot_parameters = inspect.signature(ax.boxplot).parameters
    orientation = {"orientation": "horizontal"} if "orientation" in boxplot_parameters else {"vert": False}
    tick_labels = {"tick_labels": short_names} if "tick_labels" in boxplot_parameters else {"labels": short_names}
    boxes = ax.boxplot(lengths, **orientation, **tick_labels,
                       widths=0.50, patch_artist=True, showmeans=True,
                       medianprops={"color": "#202D39", "linewidth": 1.6},
                       meanprops={"marker": "D", "markerfacecolor": "white", "markeredgecolor": "#202D39", "markersize": 4},
                       flierprops={"marker": ".", "markersize": 2, "alpha": 0.3},
                       boxprops={"linewidth": 0.8}, whiskerprops={"linewidth": 0.8},
                       capprops={"linewidth": 0.8})
    for patch, color in zip(boxes["boxes"], colors):
        patch.set(facecolor=color, edgecolor=color, alpha=0.68)
    ax.set(xlabel="Response length (tokens)", xlim=(0, max(300, np.ceil(max(all_lengths) / 50) * 50)))
    ax.invert_yaxis()
    ax.xaxis.set_major_locator(MultipleLocator(50))
    ax.grid(axis="x", color="#e4e8ec", linewidth=0.7)
    ax.set_axisbelow(True)
    save_figure(fig, "response_lengths", args.output_dir)

    # Figure 3: validation history, exactly as saved, for response-bearing runs.
    fig, ax = plt.subplots(figsize=(7.0, 2.7), layout="constrained")
    styles = [
        ("cnn_output", "CNN: response", "#326B9A", "-", "o"),
        ("cnn_input_output", "CNN: prompt + response", "#326B9A", "--", "s"),
        ("lstm_output", "BiLSTM: response", "#B36B37", "-", "o"),
        ("lstm_input_output", "BiLSTM: prompt + response", "#B36B37", "--", "s"),
    ]
    values = []
    for key, label, color, linestyle, marker in styles:
        exp = experiments[key]
        epochs = [h["epoch"] for h in exp["history"]]
        f1 = [h["validation_macro_f1"] for h in exp["history"]]
        values.extend(f1)
        ax.plot(epochs, f1, label=label, color=color, linestyle=linestyle,
                linewidth=1.65, marker=marker, markersize=3.5)
        best = exp["best_epoch"]
        ax.scatter([best], [f1[epochs.index(best)]], s=70, facecolors="none",
                   edgecolors=color, linewidths=1.1, zorder=5)
    lower = max(0, np.floor((min(values) - 0.02) * 10) / 10)
    last_epoch = max(row["epoch"] for exp in experiments.values() for row in exp["history"])
    ax.set(xlabel="Epoch", ylabel="Validation macro F1", xlim=(0.8, max(10, last_epoch) + 0.2), ylim=(lower, 1.0))
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
    ax.grid(axis="y", color="#e4e8ec", linewidth=0.7)
    ax.legend(loc="lower right", frameon=False, fontsize=10)
    save_figure(fig, "validation_curves", args.output_dir)

    summary = {
        "analysis_scope": "Response-style analysis uses only prepared training responses. Evaluation summaries read saved run outputs; no training, ablation, or attribution method is performed.",
        "tokenizer": "train.tokenize: lowercased Unicode words and punctuation, regex \\w+|[^\\w\\s]",
        "punctuation_definition": "Number and percentage of training responses containing at least one literal ASCII colon (:) or semicolon (;); a response may count in both categories.",
        "quantile_method": "numpy.percentile default method='linear'; standard deviation uses ddof=1.",
        "figure_notes": {
            "confusion_matrices": "Held-out test set. Combined means prompt + response. Annotations are observed counts and color is within-row percentage. Class abbreviations: Claude Sonnet 4.6, GPT-5.4, Gemini 3.1 Pro.",
            "response_lengths": "Training set only; per-model sample counts are recorded in training_response_style. Boxes show Q1-Q3, line is median, diamond is mean, whiskers extend to observations within 1.5 IQR, dots are observations beyond whiskers. Tokens use train.tokenize before truncation.",
            "validation_curves": "Saved validation macro F1 for each completed epoch. Outlined circles mark the checkpoint selected by the original run; line endpoints reflect early stopping or the ten-epoch limit.",
        },
        "label_order": names,
        "training_response_style": by_model,
        "training_aggregate": {"n": len(records), "mean_response_tokens": float(np.mean(all_lengths)),
                               "total_response_tokens": int(sum(all_lengths)), "max_response_tokens": max(all_lengths)},
        "experiments": experiments,
        "provenance": {provenance_key(p, args.data_dir, args.runs_dir): {"sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "bytes": p.stat().st_size}
                       for p in sorted(SOURCES)},
        "software": {"python": sys.version.split()[0], "numpy": np.__version__, "matplotlib": matplotlib.__version__},
    }
    (args.output_dir / "report_derived_statistics.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"training_response_style": by_model,
                      "experiments": {k: {field: v[field] for field in ("vocab_size", "best_epoch", "epochs_run", "elapsed_seconds")} for k, v in experiments.items()},
                      "artifacts": sorted(p.name for p in args.output_dir.glob("*"))}, indent=2))
    return summary


if __name__ == "__main__":
    main()
