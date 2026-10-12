#!/usr/bin/env python3
"""Verify RQ3/RQ4 predictions and quantify keyword-level test uncertainty.

No fitting or model selection occurs here. Run only after both experiment
runners finish. Confidence intervals condition on each saved model and split;
they do not measure training-seed or domain-definition uncertainty.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


DOMAINS = ("life_sciences", "physical_sciences")
MODELS = ("cnn", "lstm")
ABLATIONS = ("no_punctuation", "no_punctuation_fixed_words")


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def original_run_hashes(directory):
    paths = [directory / "results.csv", directory / "tokenization_stats.json"]
    for model in MODELS:
        for mode in ("input", "output", "input_output"):
            paths.extend(path for path in (directory / f"{model}_{mode}").rglob("*") if path.is_file())
    return {str(path): file_hash(path) for path in sorted(paths) if path.is_file()}


def scores_from_confusion(matrices):
    """Accept (..., actual class, predicted class); return vectorized metrics."""
    matrices = np.asarray(matrices, dtype=np.float64)
    if matrices.ndim < 2 or matrices.shape[-1] != matrices.shape[-2]:
        raise ValueError("Confusion matrices must be square")
    if np.any(matrices < 0) or not np.all(np.isfinite(matrices)):
        raise ValueError("Confusion counts must be finite and nonnegative")
    support = matrices.sum(axis=-1)
    total = support.sum(axis=-1)
    if np.any(total <= 0):
        raise ValueError("Cannot score an empty confusion matrix")
    correct = np.diagonal(matrices, axis1=-2, axis2=-1)
    denominators = support + matrices.sum(axis=-2)
    class_f1 = np.divide(2 * correct, denominators, out=np.zeros_like(correct), where=denominators > 0)
    return {"accuracy": correct.sum(axis=-1) / total,
            "macro_f1": class_f1.mean(axis=-1),
            "weighted_f1": (class_f1 * support).sum(axis=-1) / total,
            "class_f1": class_f1}


def validate_predictions(path, expected_rows, label_names):
    """Require exactly the held-out IDs, their labels and their keyword groups."""
    rows = load_jsonl(path)
    expected = {row["id"]: row for row in expected_rows}
    if len(expected) != len(expected_rows):
        raise ValueError("Expected test data contains repeated IDs")
    predictions = {}
    for row in rows:
        if row["id"] in predictions:
            raise ValueError(f"Duplicate prediction ID: {row['id']}")
        if row["id"] not in expected:
            raise ValueError(f"Unexpected prediction ID: {row['id']}")
        original = expected[row["id"]]
        for key in ("label", "keyword", "model"):
            if row[key] != original[key]:
                raise ValueError(f"Prediction {key} differs for {row['id']}")
        label, prediction = row["label"], row["prediction"]
        if type(label) is not int or type(prediction) is not int:
            raise ValueError("Labels and predictions must be integers")
        if not 0 <= label < len(label_names) or not 0 <= prediction < len(label_names):
            raise ValueError("Prediction class outside label range")
        if row["model"] != label_names[label] or row["predicted_model"] != label_names[prediction]:
            raise ValueError("Label name and index disagree")
        probabilities = np.asarray(row["probabilities"], dtype=float)
        if (probabilities.shape != (len(label_names),) or not np.all(np.isfinite(probabilities))
                or np.any(probabilities < 0) or not np.isclose(probabilities.sum(), 1, atol=1e-5)
                or int(probabilities.argmax()) != prediction):
            raise ValueError(f"Invalid probabilities for {row['id']}")
        predictions[row["id"]] = row
    if set(predictions) != set(expected):
        raise ValueError(f"Missing {len(set(expected) - set(predictions))} test predictions: {path}")
    return [predictions[row["id"]] for row in expected_rows]


def keyword_confusions(rows, n_classes):
    keywords = tuple(sorted({row["keyword"] for row in rows}))
    if len(keywords) < 2:
        raise ValueError("Keyword bootstrap requires at least two clusters")
    positions = {keyword: index for index, keyword in enumerate(keywords)}
    counts = np.zeros((len(keywords), n_classes, n_classes), dtype=np.int64)
    for row in rows:
        counts[positions[row["keyword"]], row["label"], row["prediction"]] += 1
    return keywords, counts


class KeywordBootstrap:
    """Share resampled keyword multiplicities for identical test keyword sets.

    Each distinct sorted keyword set has a deterministic independent PRNG
    stream derived from seed and SHA256. Resampling selects K whole clusters
    with replacement, retaining every example within each selected cluster.
    """
    def __init__(self, replicates=2000, seed=42):
        if replicates < 2 or seed < 0:
            raise ValueError("Need at least two replicates and a nonnegative seed")
        self.replicates, self.seed, self.draws = replicates, seed, {}

    def multiplicities(self, keywords):
        keywords = tuple(keywords)
        if keywords != tuple(sorted(set(keywords))) or len(keywords) < 2:
            raise ValueError("Keywords must be sorted and unique, with at least two groups")
        if keywords not in self.draws:
            payload = json.dumps(keywords, ensure_ascii=False, separators=(",", ":")).encode()
            digest = hashlib.sha256(payload).digest()
            entropy = [self.seed] + [int.from_bytes(digest[i:i + 4], "little") for i in range(0, 32, 4)]
            rng = np.random.default_rng(np.random.SeedSequence(entropy))
            selected = rng.integers(0, len(keywords), size=(self.replicates, len(keywords)))
            counts = np.zeros_like(selected)
            for i, draw in enumerate(selected):
                counts[i] = np.bincount(draw, minlength=len(keywords))
            self.draws[keywords] = counts
        return self.draws[keywords]

    def scores(self, keywords, confusions):
        if len(keywords) != len(confusions):
            raise ValueError("Keyword and confusion matrix counts differ")
        sampled = np.einsum("bk,kij->bij", self.multiplicities(keywords), confusions)
        return scores_from_confusion(sampled)


def percentile_interval(values):
    return [float(x) for x in np.quantile(values, [0.025, 0.975], method="linear")]


def verify_saved_metrics(point, confusion, saved, path):
    for metric in ("accuracy", "macro_f1", "weighted_f1"):
        if not np.isclose(float(point[metric]), saved[metric], atol=1e-12, rtol=0):
            raise ValueError(f"Recomputed {metric} differs from saved metrics: {path}")
    if not np.array_equal(confusion, np.asarray(saved["confusion_matrix"])):
        raise ValueError(f"Recomputed confusion matrix differs from saved metrics: {path}")


def analyze_prediction_file(path, metrics_path, metric_key, expected_rows, names, bootstrap, **identity):
    rows = validate_predictions(path, expected_rows, names)
    keywords, counts = keyword_confusions(rows, len(names))
    confusion = counts.sum(axis=0)
    point = scores_from_confusion(confusion)
    saved = load_json(metrics_path)[metric_key]
    verify_saved_metrics(point, confusion, saved, metrics_path)
    samples = bootstrap.scores(keywords, counts)
    result = dict(identity, n_examples=len(rows), n_keywords=len(keywords),
                  prediction_file=str(path), prediction_sha256=file_hash(path))
    for metric in ("accuracy", "macro_f1"):
        low, high = percentile_interval(samples[metric])
        result.update({metric: float(point[metric]), metric + "_ci_low": low, metric + "_ci_high": high})
    result["class_f1"] = {name: float(value) for name, value in zip(names, point["class_f1"])}
    result["confusion_matrix"] = confusion.tolist()
    return result, samples


def metric_contrast(left_result, left_samples, right_result, right_samples, **identity):
    """Calculate left minus right; shared resampling makes same-set gaps paired."""
    result = dict(identity)
    for metric in ("accuracy", "macro_f1"):
        low, high = percentile_interval(left_samples[metric] - right_samples[metric])
        key = metric + "_difference"
        result.update({key: left_result[metric] - right_result[metric],
                       key + "_ci_low": low, key + "_ci_high": high})
    return result


def coefficient_summaries(rq4_dir, top_n=5):
    summaries = []
    for kind in ("length", "structure", "lexical"):
        directory = rq4_dir / "feature_baselines" / kind
        config = load_json(directory / "config.json")
        by_model = defaultdict(list)
        with (directory / "coefficients.csv").open(encoding="utf-8", newline="") as stream:
            for row in csv.DictReader(stream):
                by_model[row["model"]].append({"feature": row["feature"], "coefficient": float(row["coefficient"])})
        for model, rows in sorted(by_model.items()):
            positive = sorted((r for r in rows if r["coefficient"] > 0), key=lambda r: -r["coefficient"])[:top_n]
            negative = sorted((r for r in rows if r["coefficient"] < 0), key=lambda r: r["coefficient"])[:top_n]
            summaries.append({"kind": kind, "model": model, "selected_C": config["selected_C"],
                              "positive": positive, "negative": negative,
                              "scale": "TF-IDF" if kind == "lexical" else "standardized numeric features"})
    return summaries


def write_csv(path, rows):
    scalar_rows = [{key: value for key, value in row.items() if not isinstance(value, (dict, list))} for row in rows]
    fields = list(dict.fromkeys(key for row in scalar_rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(scalar_rows)


def create_figure(summary, output_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "svg.fonttype": "none"})
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 5.5), gridspec_kw={"width_ratios": [1, 1.2]})
    colors = {"cnn": "#276998", "lstm": "#B45528", "logistic": "#3C8067"}
    rq3 = [row for row in summary["results"] if row["rq"] == "RQ3"]
    labels = []
    for i, row in enumerate(rq3):
        source = "Life" if row["source_domain"] == "life_sciences" else "Physical"
        target = "Life" if row["target_domain"] == "life_sciences" else "Physical"
        labels.append(f"{row['architecture'].upper()}: {source} → {target}")
        y = len(rq3) - 1 - i
        color = colors[row["architecture"]]
        axes[0].hlines(y, row["macro_f1_ci_low"], row["macro_f1_ci_high"], color=color, lw=2)
        axes[0].scatter(row["macro_f1"], y, color=color, marker="o" if row["evaluation"] == "in_domain" else "s", s=38)
    axes[0].set(yticks=list(reversed(range(len(rq3)))), yticklabels=labels,
                title="RQ3 · Subject-domain transfer", xlabel="Test macro-F1", xlim=(0.3, 1.005))
    rq4 = [row for row in summary["results"] if row["rq"] == "RQ4"]
    display = {"original": "Original", "no_punctuation": "No punctuation", "no_punctuation_fixed_words": "No punctuation + last 32 words",
               "logistic_length": "Length features", "logistic_structure": "Structure / diversity", "logistic_lexical": "Word TF-IDF"}
    labels = []
    for i, row in enumerate(rq4):
        architecture = "Linear" if row["architecture"] == "logistic" else row["architecture"].upper()
        labels.append(f"{architecture}: {display[row['condition']]}")
        y = len(rq4) - 1 - i
        color = colors[row["architecture"]]
        axes[1].hlines(y, row["macro_f1_ci_low"], row["macro_f1_ci_high"], color=color, lw=2)
        axes[1].scatter(row["macro_f1"], y, color=color, s=38)
    axes[1].set(yticks=list(reversed(range(len(rq4)))), yticklabels=labels,
                title="RQ4 · Information retained by each representation", xlabel="Test macro-F1", xlim=(0.3, 1.005))
    for ax in axes:
        ax.grid(axis="x", alpha=0.2)
        ax.set_axisbelow(True)
        ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    fig.text(0.5, 0.025, "95% keyword-bootstrap intervals; 2,000 replicates. Conditional on one fitted model.\n"
             "RQ3: circles = in-domain; squares = cross-domain. RQ4: last 32-word inputs also discard earlier content.",
             ha="center", va="bottom", fontsize=9, color="#454545")
    fig.tight_layout(rect=(0, 0.105, 1, 1), w_pad=2)
    for suffix in ("png", "svg"):
        fig.savefig(output_dir / f"extra_credit_comparison.{suffix}", dpi=180, bbox_inches="tight")
    plt.close(fig)


def analyze(args):
    metadata = load_json(args.data_dir / "metadata.json")
    names = metadata["label_names"]
    full_test = load_jsonl(args.data_dir / "test.jsonl")
    original_by_id = {row["id"]: row for row in full_test}
    if len(original_by_id) != len(full_test):
        raise ValueError("Original test data contains duplicate IDs")
    split_hashes = {split: file_hash(args.data_dir / f"{split}.jsonl") for split in ("train", "validation", "test")}
    rq4_plan = load_json(args.rq4_dir / "plan.json")
    if rq4_plan["split_hashes"] != split_hashes or rq4_plan["label_names"] != names:
        raise ValueError("Original data or labels differ from frozen RQ4 plan")
    rq3_manifest = load_json(args.rq3_data_dir / "manifest.json")
    if rq3_manifest["provenance"]["prepared_file_sha256"] != {f"{key}.jsonl": value for key, value in split_hashes.items()}:
        raise ValueError("Original data differ from frozen RQ3 preparation manifest")
    rq3_protocol = load_json(args.rq3_dir / "protocol.json")
    for relative, digest in rq3_protocol["files_sha256"].items():
        if relative.endswith(".json") or relative.endswith(".jsonl"):
            if file_hash(args.rq3_data_dir / relative) != digest:
                raise ValueError(f"RQ3 input differs from frozen protocol: {relative}")
    baseline_before = original_run_hashes(args.baseline_dir)
    bootstrap = KeywordBootstrap(args.replicates, args.seed)
    results, contrasts = [], []
    for source in DOMAINS:
        other = next(domain for domain in DOMAINS if domain != source)
        source_data = args.rq3_data_dir / source
        source_test, target_test = load_jsonl(source_data / "test.jsonl"), load_jsonl(source_data / "target_test.jsonl")
        for domain, subset in ((source, source_test), (other, target_test)):
            mapped_keywords = set(rq3_manifest["domain_mapping"]["domains"][domain]["keywords"])
            expected = [row for row in full_test if row["keyword"] in mapped_keywords]
            if subset != expected:
                raise ValueError("RQ3 test data do not contain exactly the original rows assigned to their domain")
        for subset in (source_test, target_test):
            if any(original_by_id.get(row["id"]) != row for row in subset):
                raise ValueError("An RQ3 test row differs from the original fixed test data")
        if {row["keyword"] for row in source_test} & {row["keyword"] for row in target_test}:
            raise ValueError("RQ3 source and target test keywords overlap")
        for model in MODELS:
            directory = args.rq3_dir / source / f"{model}_output"
            cross_metadata = load_json(directory / "cross_domain_metrics.json")
            source_metadata = load_json(directory / "metrics.json")
            if (cross_metadata["checkpoint_sha256"] != file_hash(directory / "best.pt")
                    or cross_metadata["best_epoch"] != source_metadata["best_epoch"]
                    or cross_metadata["source_domain"] != source or cross_metadata["target_domain"] != other):
                raise ValueError("RQ3 cross-domain evaluation does not match the saved source checkpoint")
            in_result, in_samples = analyze_prediction_file(directory / "test_predictions.jsonl", directory / "metrics.json", "test",
                source_test, names, bootstrap, rq="RQ3", condition="source_training", architecture=model,
                source_domain=source, target_domain=source, evaluation="in_domain")
            cross_result, cross_samples = analyze_prediction_file(directory / "cross_domain_predictions.jsonl", directory / "cross_domain_metrics.json", "target_test",
                target_test, names, bootstrap, rq="RQ3", condition="source_training", architecture=model,
                source_domain=source, target_domain=other, evaluation="cross_domain")
            results.extend([in_result, cross_result])
            contrasts.append(metric_contrast(cross_result, cross_samples, in_result, in_samples,
                rq="RQ3", architecture=model, source_domain=source, comparison="cross_domain_minus_in_domain",
                bootstrap_pairing="independent source/target domain keyword resampling"))
    originals = {}
    for model in MODELS:
        directory = args.baseline_dir / f"{model}_output"
        result, samples = analyze_prediction_file(directory / "test_predictions.jsonl", directory / "metrics.json", "test",
            full_test, names, bootstrap, rq="RQ4", condition="original", architecture=model,
            source_domain="all", target_domain="all", evaluation="fixed_test")
        originals[model] = result, samples
        results.append(result)
    for condition in ABLATIONS:
        for model in MODELS:
            directory = args.rq4_dir / condition / f"{model}_output"
            result, samples = analyze_prediction_file(directory / "test_predictions.jsonl", directory / "metrics.json", "test",
                full_test, names, bootstrap, rq="RQ4", condition=condition, architecture=model,
                source_domain="all", target_domain="all", evaluation="fixed_test")
            results.append(result)
            contrasts.append(metric_contrast(result, samples, *originals[model], rq="RQ4", architecture=model,
                source_domain="all", comparison=f"{condition}_minus_original", bootstrap_pairing="paired identical keyword resamples"))
    for kind in ("length", "structure", "lexical"):
        directory = args.rq4_dir / "feature_baselines" / kind
        result, _ = analyze_prediction_file(directory / "test_predictions.jsonl", directory / "metrics.json", "test",
            full_test, names, bootstrap, rq="RQ4", condition=f"logistic_{kind}", architecture="logistic",
            source_domain="all", target_domain="all", evaluation="fixed_test")
        result["selected_C"] = load_json(directory / "config.json")["selected_C"]
        results.append(result)
    baseline_after = original_run_hashes(args.baseline_dir)
    if baseline_before != baseline_after:
        raise RuntimeError("Original baseline artifacts changed during analysis")
    return {
        "bootstrap": {"replicates": args.replicates, "seed": args.seed, "confidence_level": 0.95,
                      "interval": "percentile", "unit": "whole keyword clusters",
                      "numpy_version": np.__version__,
                      "draw_sharing": "identical keyword sets use identical draws; disjoint domain sets use independent streams",
                      "stream_derivation": "NumPy SeedSequence([seed, SHA256(sorted keyword names) as eight little-endian uint32 words])",
                      "conditional_on": "saved model, training seed, fixed split and chosen domain mapping"},
        "label_names": names, "results": results, "contrasts": contrasts,
        "coefficients": coefficient_summaries(args.rq4_dir),
        "audit": {"all_test_ids_labels_keywords_and_predictions_verified": True,
                  "analysis_code_sha256": file_hash(Path(__file__)),
                  "all_saved_point_metrics_recomputed_and_matched": True,
                  "baseline_hashes_unchanged_during_analysis": True,
                  "baseline_hashes": baseline_before,
                  "historical_baseline_hash_snapshot_available": False,
                  "baseline_note": "No pre-extension hash snapshot exists; historical non-modification is not certified.",
                  "original_split_sha256": split_hashes,
                  "frozen_input_hashes_verified": True,
                  "rq3_protocol_sha256": file_hash(args.rq3_dir / "protocol.json"),
                  "rq4_plan_sha256": file_hash(args.rq4_dir / "plan.json")},
        "limitations": [
            "Exploratory extension designed after original test results were seen; no new scores used for tuning.",
            "Intervals describe test keyword resampling, not training-seed, domain-map, or collection uncertainty.",
            "RQ3 has only 9 physical-science and 14 life-science test keyword clusters; interval estimates are limited.",
            "RQ3 changes requested scientific subject domain within one task and prompt template, not task type.",
            "Direction comparisons differ in source training size and topic composition.",
            "No-punctuation retraining tests retained predictive information, not causal reliance of the original network.",
            "Last32 controls token count but also removes earlier content and retains word-length and lexical cues.",
            "Raw-text feature baselines include cues such as uppercase and newlines that the neural tokenizer removes.",
            "Linear coefficients are associations conditional on other features, not causal or neural explanations.",
            "Intervals are descriptive and unadjusted for multiple reported comparisons.",
        ],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/prepared"))
    parser.add_argument("--baseline-dir", type=Path, default=Path("runs"))
    parser.add_argument("--rq3-data-dir", type=Path, default=Path("data/rq3"))
    parser.add_argument("--rq3-dir", type=Path, default=Path("runs/rq3"))
    parser.add_argument("--rq4-dir", type=Path, default=Path("runs/rq4"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/extra_credit_analysis"))
    parser.add_argument("--replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        parser.error("Analysis output directory already exists; choose another directory.")
    summary = analyze(args)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    write_csv(args.output_dir / "results.csv", summary["results"])
    write_csv(args.output_dir / "contrasts.csv", summary["contrasts"])
    coefficient_rows = [dict(kind=item["kind"], model=item["model"], direction=direction, **row)
                        for item in summary["coefficients"] for direction in ("positive", "negative") for row in item[direction]]
    write_csv(args.output_dir / "top_coefficients.csv", coefficient_rows)
    create_figure(summary, args.output_dir)
    print(f"Verified {len(summary['results'])} test evaluations and wrote {len(summary['contrasts'])} contrasts to {args.output_dir}")


if __name__ == "__main__":
    main()
