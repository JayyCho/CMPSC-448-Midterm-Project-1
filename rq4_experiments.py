#!/usr/bin/env python3
"""RQ4: controlled response-text ablations and interpretable feature baselines.

Reuses the original fixed splits, labels, seed and neural hyperparameters. A
--plan-only run performs checks without fitting or writing results. The output
directory must not exist, so original experiments cannot be overwritten.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import re
from types import SimpleNamespace
import unicodedata
import warnings

import joblib
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
import torch

import train

FEATURE_KINDS = ("length", "structure", "lexical")
ABLATIONS = ("no_punctuation", "no_punctuation_fixed_words")
C_GRID = (0.01, 0.1, 1.0, 10.0)
LENGTH_FEATURES = ("log1p_word_count", "log1p_character_count")
STRUCTURE_FEATURES = (
    "mean_word_characters", "unique_word_fraction", "digit_word_fraction",
    "uppercase_character_fraction", "punctuation_character_fraction",
    "comma_character_fraction", "period_character_fraction", "colon_character_fraction",
    "semicolon_character_fraction", "hyphen_character_fraction",
    "parenthesis_character_fraction", "quote_character_fraction",
    "slash_character_fraction", "newline_character_fraction",
    "question_character_fraction", "exclamation_character_fraction",
    "bullet_lines_per_word", "markdown_heading_lines_per_word",
)


def word_tokens(text: str) -> list[str]:
    """Lowercase Unicode alphanumeric runs; underscore is punctuation, too."""
    return re.findall(r"[^\W_]+", text.lower(), flags=re.UNICODE)


def transform_text(text: str, condition: str, fixed_words: int = 32) -> str:
    if condition not in ABLATIONS:
        raise ValueError(f"Unknown ablation: {condition}")
    words = word_tokens(text)
    if not words:
        raise ValueError("Response contains no word tokens")
    if condition == "no_punctuation_fixed_words":
        if fixed_words < 1:
            raise ValueError("fixed_words must be positive")
        if len(words) < fixed_words:
            raise ValueError(f"Response has {len(words)} words; fixed {fixed_words}-word experiment cannot include it")
        words = words[-fixed_words:]
    return " ".join(words)


def transformed_splits(splits: dict, condition: str, fixed_words: int) -> dict:
    """Copy rows without selecting, reordering, relabelling or changing splits."""
    return {split: [dict(row, output=transform_text(row["output"], condition, fixed_words))
                    for row in rows] for split, rows in splits.items()}


def numeric_features(text: str, kind: str) -> list[float]:
    words = word_tokens(text)
    n_words, n_chars = max(len(words), 1), max(len(text), 1)
    if kind == "length":
        return [float(np.log1p(len(words))), float(np.log1p(len(text)))]
    if kind != "structure":
        raise ValueError(f"Unknown numeric feature kind: {kind}")
    fraction = lambda characters: sum(char in characters for char in text) / n_chars
    return [
        sum(map(len, words)) / n_words,
        len(set(words)) / n_words,
        sum(any(char.isdigit() for char in word) for word in words) / n_words,
        sum(char.isupper() for char in text) / n_chars,
        sum(unicodedata.category(char).startswith("P") for char in text) / n_chars,
        fraction(",，"), fraction(".。"), fraction(":："), fraction(";；"),
        fraction("-–—"), fraction("()[]{}（）"), fraction("\"'‘’“”"),
        fraction("/\\"), text.count("\n") / n_chars,
        fraction("?？"), fraction("!！"),
        len(re.findall(r"(?m)^\s*(?:[-*•]|\d+[.)])\s+", text)) / n_words,
        len(re.findall(r"(?m)^\s*#{1,6}\s+", text)) / n_words,
    ]


def feature_matrix(rows: list[dict], kind: str):
    return np.asarray([numeric_features(row["output"], kind) for row in rows], dtype=np.float64)


def fit_transformer(training_rows: list[dict], kind: str):
    """Only training records are accepted at this fitting boundary."""
    if kind == "lexical":
        transformer = TfidfVectorizer(token_pattern=r"(?u)[^\W_]+",
                                      lowercase=True, ngram_range=(1, 2),
                                      min_df=2, max_features=30000, sublinear_tf=True)
        matrix = transformer.fit_transform([row["output"] for row in training_rows])
        names = transformer.get_feature_names_out().tolist()
    elif kind in {"length", "structure"}:
        transformer = StandardScaler()
        matrix = transformer.fit_transform(feature_matrix(training_rows, kind))
        names = list(LENGTH_FEATURES if kind == "length" else STRUCTURE_FEATURES)
    else:
        raise ValueError(f"Unknown feature baseline: {kind}")
    return transformer, matrix, names


def apply_transformer(transformer, rows: list[dict], kind: str):
    values = [row["output"] for row in rows] if kind == "lexical" else feature_matrix(rows, kind)
    return transformer.transform(values)


def fit_feature_baseline(training_rows: list[dict], validation_rows: list[dict],
                         kind: str, label_names: list[str], seed: int = 42,
                         c_grid=C_GRID):
    """Test rows are deliberately absent; selection uses validation labels only."""
    transformer, training_matrix, feature_names = fit_transformer(training_rows, kind)
    validation_matrix = apply_transformer(transformer, validation_rows, kind)
    y_train = np.asarray([row["label"] for row in training_rows])
    y_validation = np.asarray([row["label"] for row in validation_rows])
    candidates, best, best_key = [], None, (-1.0, float("-inf"))
    for c_value in sorted(c_grid):
        model = LogisticRegression(C=c_value, max_iter=2000, solver="lbfgs", random_state=seed)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            model.fit(training_matrix, y_train)
        if any(issubclass(item.category, ConvergenceWarning) for item in caught):
            raise RuntimeError(f"{kind} logistic regression failed to converge at C={c_value}")
        probabilities = model.predict_proba(validation_matrix)
        metrics = train.score_predictions(y_validation, model.classes_[probabilities.argmax(1)], label_names)
        metrics["loss"] = float(log_loss(y_validation, probabilities, labels=model.classes_))
        candidates.append({"C": c_value, "validation": metrics})
        key = (metrics["macro_f1"], -metrics["loss"])
        if key > best_key:
            best, best_key = model, key
    return transformer, best, feature_names, candidates


def save_predictions(path: Path, rows: list[dict], predictions, probabilities, names):
    with path.open("w", encoding="utf-8") as handle:
        for row, prediction, probability in zip(rows, predictions, probabilities):
            handle.write(json.dumps({"id": row["id"], "keyword": row["keyword"],
                                     "label": row["label"], "model": names[row["label"]],
                                     "prediction": int(prediction), "predicted_model": names[int(prediction)],
                                     "probabilities": [float(value) for value in probability]}) + "\n")


def run_feature_baseline(kind, splits, names, output_dir, seed):
    directory = output_dir / "feature_baselines" / kind
    directory.mkdir(parents=True, exist_ok=False)
    transformer, model, feature_names, candidates = fit_feature_baseline(
        splits["train"], splits["validation"], kind, names, seed)
    test_matrix = apply_transformer(transformer, splits["test"], kind)
    probabilities = model.predict_proba(test_matrix)
    predictions = model.classes_[probabilities.argmax(1)]
    truth = [row["label"] for row in splits["test"]]
    metrics = train.score_predictions(truth, predictions, names)
    metrics["loss"] = float(log_loss(truth, probabilities, labels=model.classes_))
    selected = next(candidate for candidate in candidates if candidate["C"] == model.C)
    train.write_json(directory / "config.json", {
        "kind": kind, "selected_C": model.C, "C_grid": list(C_GRID), "seed": seed,
        "feature_names": feature_names, "selection": "validation macro-F1; validation log-loss tie break",
        "fit_split": "train", "refit_on_validation": False,
        "lexical_settings": {"ngram_range": [1, 2], "min_df": 2, "max_features": 30000,
                             "sublinear_tf": True} if kind == "lexical" else None,
    })
    train.write_json(directory / "validation_selection.json", candidates)
    train.write_json(directory / "metrics.json", {"model": f"logistic_{kind}",
                     "validation": selected["validation"], "test": metrics})
    save_predictions(directory / "test_predictions.jsonl", splits["test"], predictions, probabilities, names)
    joblib.dump({"transformer": transformer, "model": model, "feature_names": feature_names,
                 "kind": kind, "label_names": names}, directory / "model.joblib")
    with (directory / "coefficients.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["label", "model", "feature", "coefficient"])
        writer.writeheader()
        for label, coefficients in zip(model.classes_, model.coef_):
            # Numeric coefficients are on standardized inputs; lexical on TF-IDF.
            for index in np.argsort(-coefficients):
                writer.writerow({"label": int(label), "model": names[int(label)],
                                 "feature": feature_names[index], "coefficient": float(coefficients[index])})
    train.save_confusion_plot(metrics["confusion_matrix"], names, directory / "confusion_matrix.png")
    return {"condition": f"logistic_{kind}", "model": "logistic", "test_accuracy": metrics["accuracy"],
            "test_macro_f1": metrics["macro_f1"], "validation_macro_f1": selected["validation"]["macro_f1"],
            "selected_C": model.C, "best_epoch": "", "artifact_dir": str(directory)}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_baselines(baseline_dir: Path, models, splits, metadata):
    configurations, reference_rows = {}, []
    for model in models:
        directory = baseline_dir / f"{model}_output"
        config = json.loads((directory / "config.json").read_text())
        baseline_metadata = json.loads((directory / "data_metadata.json").read_text())
        if baseline_metadata != metadata:
            raise ValueError(f"Baseline {directory} metadata differs from current fixed splits")
        if config["mode"] != "output" or config["model"] != model:
            raise ValueError(f"Wrong baseline modality/model in {directory}")
        baseline_predictions = [json.loads(line) for line in
                                (directory / "test_predictions.jsonl").read_text().splitlines()]
        if [(row["id"], row["label"]) for row in baseline_predictions] != [
                (row["id"], row["label"]) for row in splits["test"]]:
            raise ValueError(f"Baseline test IDs/labels differ from current fixed test split: {directory}")
        metrics = json.loads((directory / "metrics.json").read_text())
        configurations[model] = config
        reference_rows.append({"condition": "original", "model": model,
                               "test_accuracy": metrics["test"]["accuracy"],
                               "test_macro_f1": metrics["test"]["macro_f1"],
                               "validation_macro_f1": metrics["validation"]["macro_f1"],
                               "selected_C": "", "best_epoch": metrics["best_epoch"],
                               "artifact_dir": str(directory)})
    return configurations, reference_rows


def diagnostics(splits, fixed_words):
    result = {"fixed_words_predeclared": fixed_words, "splits": {}, "transformed_cross_split_collisions": {},
              "transformed_collision_details": {}}
    for split, rows in splits.items():
        lengths = np.asarray([len(word_tokens(row["output"])) for row in rows])
        if np.any(lengths < fixed_words):
            raise ValueError(f"{split}: {int((lengths < fixed_words).sum())} responses shorter than predeclared {fixed_words} words")
        result["splits"][split] = {"rows": len(rows), "minimum_words": int(lengths.min()),
                                  "mean_words": float(lengths.mean()), "maximum_words": int(lengths.max()),
                                  "class_counts": dict(Counter(row["model"] for row in rows))}
    for condition in ABLATIONS:
        owners = defaultdict(list)
        for split, rows in splits.items():
            for row in rows:
                owners[transform_text(row["output"], condition, fixed_words)].append(
                    {"split": split, "id": row["id"], "keyword": row["keyword"], "label": row["label"]})
        collisions = [{"text_sha256": hashlib.sha256(text.encode()).hexdigest(), "rows": value}
                      for text, value in owners.items() if len({row["split"] for row in value}) > 1]
        result["transformed_cross_split_collisions"][condition] = len(collisions)
        result["transformed_collision_details"][condition] = {
            "groups": collisions,
            "groups_involving_test": sum(any(row["split"] == "test" for row in group["rows"])
                                         for group in collisions),
            "policy": "Report transformation-induced collisions; abort fitting if any exist. Preserve original fixed splits and all samples.",
        }
    return result


def make_plan(args, splits, metadata, baseline_configs):
    return {
        "research_question": "RQ4: features associated with generator identification",
        "data_dir": str(args.data_dir), "baseline_dir": str(args.baseline_dir),
        "split_hashes": {split: sha256(args.data_dir / f"{split}.jsonl") for split in splits},
        "metadata_sha256": sha256(args.data_dir / "metadata.json"),
        "label_names": metadata["label_names"], "fixed_words": args.fixed_words,
        "fixed_words_rule": "Exactly the LAST32 words by default; require every row to have >=N words, otherwise abort. No filtering or padding.",
        "fixed_word_position": "suffix",
        "protocol_revision_before_fitting": "A first32-word diagnostic found one train/validation duplicate (biomimicry/biomimetics). The suffix32 control was chosen before any new real-data performance results; original samples/splits are retained.",
        "ablations": list(ABLATIONS), "neural_models": args.models,
        "baseline_configs": baseline_configs, "feature_baselines": list(FEATURE_KINDS),
        "C_grid": list(C_GRID), "cpu_threads": args.cpu_threads,
        "neural_retraining": True, "feature_fit_split": "train", "selection_split": "validation",
        "test_use": "Final prediction and evaluation after each model is fixed; no test-based tuning",
        "exploratory_extension": "Original test scores were inspected before RQ4 was designed. The RQ4 protocol is frozen before its new test results; this is an exploratory extension, not a fresh confirmatory holdout.",
        "versions": {"torch": str(torch.__version__), "numpy": np.__version__},
        "diagnostics": diagnostics(splits, args.fixed_words),
        "interpretation": [
            "Any transformation-induced train/validation/test text collisions are disclosed in diagnostics; these are not new independent source examples.",
            "Ablations remove several interacting text cues; performance changes are not isolated causal effects.",
            "The fixed-word experiment removes total token-count variation, not lexical word length or suffix content.",
            "No-punctuation texts retain facet heading words, vocabulary, word order and response length.",
            "The fixed-word comparison discards all but the suffix; loss and position of content are confounded with length control.",
            "Structural rates are not statistically independent of response length; lexical features also encode style.",
            "The raw-text structural baseline includes capitalization/newlines that the original neural tokenizer discards; its success does not establish neural reliance on those cues.",
            "Single-seed estimates describe these model versions and scientific prompts, not universal authorship signatures.",
        ],
    }


def run_neural_ablation(condition, model, splits, names, metadata, config, args):
    transformed = transformed_splits(splits, condition, args.fixed_words)
    changes = dict(config)
    changes.update(output_dir=args.output_dir / condition, data_dir=args.data_dir,
                   models=[model], modes=["output"], overwrite=False, cpu_threads=args.cpu_threads,
                   device=args.device, num_workers=0)
    settings = SimpleNamespace(**changes)
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    vocab = train.build_vocab(transformed["train"], "output", settings.max_vocab, settings.min_freq)
    datasets = {split: train.EncodedDataset(rows, vocab, "output", settings.max_input, settings.max_output)
                for split, rows in transformed.items()}
    if condition == "no_punctuation_fixed_words":
        if settings.max_output < args.fixed_words:
            raise ValueError("Baseline max_output is below the predeclared fixed-word budget")
        if any(len(item[0]) != args.fixed_words for dataset in datasets.values() for item in dataset.items):
            raise AssertionError("Fixed-word neural examples do not have identical token counts")
    ablation_metadata = dict(metadata, rq4_condition=condition, rq4_fixed_words=args.fixed_words,
                             rq4_fixed_word_position="suffix",
                             rq4_transform="lowercase Unicode alphanumeric runs joined by single spaces")
    summary = train.run_experiment(model, "output", datasets, vocab, transformed, names, ablation_metadata, settings)
    return {"condition": condition, "model": model, "test_accuracy": summary["test_accuracy"],
            "test_macro_f1": summary["test_macro_f1"],
            "validation_macro_f1": summary["validation_macro_f1"], "selected_C": "",
            "best_epoch": summary["best_epoch"], "artifact_dir": str(settings.output_dir / f"{model}_output")}


def save_summary(output_dir, rows):
    with (output_dir / "results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/prepared"))
    parser.add_argument("--baseline-dir", type=Path, default=Path("runs"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/rq4"))
    parser.add_argument("--fixed-words", type=int, default=32)
    parser.add_argument("--models", nargs="+", choices=("cnn", "lstm"), default=["cnn", "lstm"])
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--features-only", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args(argv)
    if args.fixed_words < 1 or args.cpu_threads < 1:
        parser.error("fixed-words and cpu-threads must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable")
    args.models = list(dict.fromkeys(args.models))
    splits, names, metadata = train.read_data(args.data_dir)
    configs, results = check_baselines(args.baseline_dir, args.models, splits, metadata)
    plan = make_plan(args, splits, metadata, configs)
    plan["features_only"] = args.features_only
    if args.plan_only:
        print(json.dumps(plan, indent=2))
        return 0
    if any(plan["diagnostics"]["transformed_cross_split_collisions"].values()):
        raise ValueError("Transformation-induced cross-split duplicates found; inspect --plan-only before revising the protocol")
    # Atomic exclusive reservation; never reuse or overwrite an existing result directory.
    args.output_dir.mkdir(parents=True, exist_ok=False)
    train.write_json(args.output_dir / "plan.json", plan)
    torch.set_num_threads(args.cpu_threads)
    with threadpool_limits(limits=args.cpu_threads):
        for kind in FEATURE_KINDS:
            result = run_feature_baseline(kind, splits, names, args.output_dir, metadata["seed"])
            results.append(result)
            save_summary(args.output_dir, results)
            print(json.dumps(result), flush=True)
        if not args.features_only:
            for condition in ABLATIONS:
                for model in args.models:
                    result = run_neural_ablation(condition, model, splits, names, metadata, configs[model], args)
                    results.append(result)
                    save_summary(args.output_dir, results)
                    print(json.dumps(result), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
