#!/usr/bin/env python3
"""Evaluate simple attention-mass detectors for prompt/suffix attacks.

The detector treats rows with type=attacked as positives and clean rows as
negatives. By default it uses paired sample indices only, so each attacked row is
compared against the same sample's clean row.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Iterable

FEATURES = [
    "student",
    "suffix",
    "student_suffix",
    "instruction",
    "markup",
]


@dataclass
class DetectorResult:
    feature: str
    direction: str
    threshold: float
    auc: float
    accuracy: float
    precision: float
    recall: float
    f1: float
    tp: int
    fp: int
    tn: int
    fn: int
    clean_mean: float
    attack_mean: float
    clean_median: float
    attack_median: float
    clean_std: float
    attack_std: float
    attack_over_clean: float
    delta: float

    def to_dict(self) -> dict:
        data = self.__dict__.copy()
        for key, value in list(data.items()):
            if isinstance(value, float) and not math.isfinite(value):
                data[key] = None
        return data


def _field(row: dict, name: str) -> str:
    if name in row:
        return row[name]
    bom_name = "\ufeff" + name
    return row.get(bom_name, "")


def _float(row: dict, name: str) -> float:
    value = _field(row, name)
    return float(value) if value not in ("", None) else 0.0


def load_rows(path: Path, *, paired_only: bool) -> list[dict]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    clean_ids = {int(_field(r, "sample_idx")) for r in rows if _field(r, "type") == "clean"}
    attack_ids = {int(_field(r, "sample_idx")) for r in rows if _field(r, "type") == "attacked"}
    keep_ids = clean_ids & attack_ids if paired_only else clean_ids | attack_ids

    selected = []
    for row in rows:
        row_type = _field(row, "type")
        if row_type not in {"clean", "attacked"}:
            continue
        sample_idx = int(_field(row, "sample_idx"))
        if sample_idx in keep_ids:
            selected.append(row)
    return selected


def auc_score(labels: list[int], scores: list[float]) -> float:
    pos = sum(labels)
    neg = len(labels) - pos
    if pos == 0 or neg == 0:
        return 0.0

    order = sorted(range(len(scores)), key=lambda i: scores[i])
    rank_sum = 0.0
    rank = 1
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        avg_rank = (rank + rank + (j - i)) / 2.0
        for k in range(i, j + 1):
            if labels[order[k]] == 1:
                rank_sum += avg_rank
        rank += j - i + 1
        i = j + 1
    return (rank_sum - pos * (pos + 1) / 2.0) / (pos * neg)


def confusion(labels: list[int], scores: list[float], threshold: float) -> tuple[int, int, int, int]:
    tp = fp = tn = fn = 0
    for label, score in zip(labels, scores):
        pred = 1 if score >= threshold else 0
        if pred == 1 and label == 1:
            tp += 1
        elif pred == 1 and label == 0:
            fp += 1
        elif pred == 0 and label == 0:
            tn += 1
        else:
            fn += 1
    return tp, fp, tn, fn


def metrics_from_confusion(tp: int, fp: int, tn: int, fn: int) -> tuple[float, float, float, float]:
    total = tp + fp + tn + fn
    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return accuracy, precision, recall, f1


def best_threshold(labels: list[int], scores: list[float]) -> tuple[float, tuple[int, int, int, int], float]:
    values = sorted(set(scores))
    if not values:
        return 0.0, (0, 0, 0, 0), 0.0
    candidates = [values[0] - 1e-12]
    candidates += [(a + b) / 2.0 for a, b in zip(values, values[1:])]
    candidates.append(values[-1] + 1e-12)

    best = None
    for threshold in candidates:
        cm = confusion(labels, scores, threshold)
        accuracy, precision, recall, f1 = metrics_from_confusion(*cm)
        youden = recall + (cm[2] / (cm[2] + cm[1]) if cm[2] + cm[1] else 0.0) - 1.0
        key = (f1, youden, accuracy)
        if best is None or key > best[0]:
            best = (key, threshold, cm)
    assert best is not None
    return best[1], best[2], best[0][0]


def describe(values: Iterable[float]) -> tuple[float, float, float]:
    vals = list(values)
    if not vals:
        return 0.0, 0.0, 0.0
    return mean(vals), median(vals), pstdev(vals) if len(vals) > 1 else 0.0


def evaluate_feature(rows: list[dict], feature: str) -> DetectorResult:
    labels = [1 if _field(r, "type") == "attacked" else 0 for r in rows]
    raw_scores = [_float(r, feature) for r in rows]
    auc = auc_score(labels, raw_scores)
    direction = "higher_is_attack"
    scores = raw_scores
    if auc < 0.5:
        scores = [-s for s in raw_scores]
        auc = auc_score(labels, scores)
        direction = "lower_is_attack"

    threshold, cm, _ = best_threshold(labels, scores)
    accuracy, precision, recall, f1 = metrics_from_confusion(*cm)
    tp, fp, tn, fn = cm

    clean_vals = [v for y, v in zip(labels, raw_scores) if y == 0]
    attack_vals = [v for y, v in zip(labels, raw_scores) if y == 1]
    clean_mean, clean_median, clean_std = describe(clean_vals)
    attack_mean, attack_median, attack_std = describe(attack_vals)

    if direction == "lower_is_attack":
        display_threshold = -threshold
    else:
        display_threshold = threshold

    return DetectorResult(
        feature=feature,
        direction=direction,
        threshold=display_threshold,
        auc=auc,
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        tp=tp,
        fp=fp,
        tn=tn,
        fn=fn,
        clean_mean=clean_mean,
        attack_mean=attack_mean,
        clean_median=clean_median,
        attack_median=attack_median,
        clean_std=clean_std,
        attack_std=attack_std,
        attack_over_clean=attack_mean / clean_mean if clean_mean else math.inf,
        delta=attack_mean - clean_mean,
    )


def write_markdown(path: Path, source: Path, rows: list[dict], results: list[DetectorResult], paired_only: bool) -> None:
    clean_n = sum(1 for r in rows if _field(r, "type") == "clean")
    attack_n = sum(1 for r in rows if _field(r, "type") == "attacked")
    best = results[0]
    lines = [
        "# Attention-Based Attack Detection",
        "",
        f"Source: `{source.as_posix()}`",
        f"Pairing: `{paired_only}`",
        f"Clean rows: {clean_n}",
        f"Attacked rows: {attack_n}",
        "",
        "## Main Result",
        "",
        f"The best single-feature detector is `{best.feature}` ({best.direction}) with AUC **{best.auc:.4f}**, accuracy **{best.accuracy:.4f}**, and F1 **{best.f1:.4f}**.",
        "",
        "## Feature Comparison",
        "",
        "| Feature | Direction | Clean mean | Attack mean | Delta | Attack/Clean | AUC | Acc | Precision | Recall | F1 | Threshold | Confusion (TP/FP/TN/FN) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in results:
        lines.append(
            f"| {r.feature} | {r.direction} | {r.clean_mean:.4f} | {r.attack_mean:.4f} | "
            f"{r.delta:+.4f} | {r.attack_over_clean:.2f} | {r.auc:.4f} | {r.accuracy:.4f} | "
            f"{r.precision:.4f} | {r.recall:.4f} | {r.f1:.4f} | {r.threshold:.4f} | "
            f"{r.tp}/{r.fp}/{r.tn}/{r.fn} |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "A high AUC means the attack changes where the model attends before producing the verdict. The `student_suffix` feature is the most relevant for GCG suffix-bank attacks because it measures the whole attacker-controlled region: the original student answer plus the appended adversarial suffix.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate attention-based attack detectors.")
    parser.add_argument("--input", required=True, help="Attention CSV produced by pipeline log_attention=true.")
    parser.add_argument("--output-dir", default="result/analysis/attention_detector", help="Output directory.")
    parser.add_argument("--all-clean", action="store_true", help="Use all clean rows instead of only samples that also have attacked rows.")
    args = parser.parse_args()

    source = Path(args.input)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paired_only = not args.all_clean
    rows = load_rows(source, paired_only=paired_only)

    results = [evaluate_feature(rows, feature) for feature in FEATURES]
    results.sort(key=lambda r: (r.auc, r.f1, r.accuracy), reverse=True)

    summary = {
        "source": source.as_posix(),
        "paired_only": paired_only,
        "clean_rows": sum(1 for r in rows if _field(r, "type") == "clean"),
        "attacked_rows": sum(1 for r in rows if _field(r, "type") == "attacked"),
        "best_feature": results[0].feature,
        "best_auc": results[0].auc,
        "best_accuracy": results[0].accuracy,
        "best_f1": results[0].f1,
        "features": [r.to_dict() for r in results],
    }

    (out_dir / "attention_detector_metrics.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    with (out_dir / "attention_detector_feature_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].to_dict().keys()))
        writer.writeheader()
        for result in results:
            writer.writerow(result.to_dict())
    write_markdown(out_dir / "attention_detector_report.md", source, rows, results, paired_only)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

