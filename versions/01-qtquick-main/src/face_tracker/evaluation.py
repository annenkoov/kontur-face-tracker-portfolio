"""Small, dependency-free metrics for comparing expression classifiers."""

from __future__ import annotations

from dataclasses import dataclass
import csv
import json
import math
from pathlib import Path
from statistics import median


@dataclass(frozen=True)
class ClassMetrics:
    precision: float
    recall: float
    f1: float
    support: int


@dataclass(frozen=True)
class ClassificationReport:
    accuracy: float
    macro_f1: float
    labels: tuple[str, ...]
    per_class: dict[str, ClassMetrics]
    confusion_matrix: tuple[tuple[int, ...], ...]


def classification_report(y_true: list[str], y_pred: list[str], labels: list[str]) -> ClassificationReport:
    if len(y_true) != len(y_pred):
        raise ValueError("Количество истинных и предсказанных меток должно совпадать")
    if not labels or len(set(labels)) != len(labels):
        raise ValueError("Список классов должен быть непустым и не содержать повторов")
    label_set = set(labels)
    if any(label not in label_set for label in [*y_true, *y_pred]):
        raise ValueError("Все метки должны входить в список классов")

    matrix = [[0 for _ in labels] for _ in labels]
    indices = {label: i for i, label in enumerate(labels)}
    for actual, predicted in zip(y_true, y_pred):
        matrix[indices[actual]][indices[predicted]] += 1

    per_class: dict[str, ClassMetrics] = {}
    f1_values = []
    for i, label in enumerate(labels):
        tp = matrix[i][i]
        support = sum(matrix[i])
        predicted_count = sum(row[i] for row in matrix)
        precision = tp / predicted_count if predicted_count else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = ClassMetrics(precision, recall, f1, support)
        f1_values.append(f1)
    total = len(y_true)
    correct = sum(matrix[i][i] for i in range(len(labels)))
    return ClassificationReport(
        accuracy=correct / total if total else 0.0,
        macro_f1=sum(f1_values) / len(f1_values),
        labels=tuple(labels),
        per_class=per_class,
        confusion_matrix=tuple(tuple(row) for row in matrix),
    )


def probability_quality_report(y_true: list[str], probability_rows: list[dict[str, float]],
                              labels: list[str], bins: int = 10) -> dict[str, float | int]:
    """Descriptive confidence metrics for already-aggregated labeled units."""
    if len(y_true) != len(probability_rows) or not y_true or not labels or bins < 1:
        raise ValueError("Нужны метки, вероятности и положительное число корзин")
    label_index = {label: index for index, label in enumerate(labels)}
    if len(label_index) != len(labels) or any(label not in label_index for label in y_true):
        raise ValueError("Метки должны быть уникальными и входить в заданный список классов")
    nll_total = 0.0
    brier_total = 0.0
    ece_total = 0.0
    bin_counts = [0] * bins
    bin_confidence = [0.0] * bins
    bin_correct = [0.0] * bins
    for target, row in zip(y_true, probability_rows):
        try:
            values = [float(row[label]) for label in labels]
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Для каждого класса нужна конечная вероятность") from exc
        if any(not math.isfinite(value) or not 0 <= value <= 1 for value in values):
            raise ValueError("Вероятности должны быть конечными числами от 0 до 1")
        if not math.isclose(sum(values), 1.0, rel_tol=1e-6, abs_tol=1e-6):
            raise ValueError("Сумма вероятностей каждого наблюдения должна быть равна 1")
        target_index = label_index[target]
        predicted_index = max(range(len(values)), key=values.__getitem__)
        confidence = values[predicted_index]
        correct = float(predicted_index == target_index)
        nll_total -= math.log(max(values[target_index], 1e-12))
        brier_total += sum((value - float(index == target_index)) ** 2
                           for index, value in enumerate(values))
        bucket = min(bins - 1, int(confidence * bins))
        bin_counts[bucket] += 1
        bin_confidence[bucket] += confidence
        bin_correct[bucket] += correct
    sample_count = len(y_true)
    for count, confidence_sum, correct_sum in zip(bin_counts, bin_confidence, bin_correct):
        if count:
            ece_total += count / sample_count * abs(confidence_sum / count - correct_sum / count)
    return {
        "units": sample_count,
        "bins": bins,
        "nll": nll_total / sample_count,
        "brier_score": brier_total / sample_count,
        "top_label_ece": ece_total,
    }


def top_expressions(probabilities: dict[str, float], count: int = 3) -> list[tuple[str, float]]:
    if count < 1:
        raise ValueError("count должен быть положительным")
    ranked = sorted(probabilities.items(), key=lambda item: item[1], reverse=True)[:count]
    total = sum(max(0.0, float(value)) for _, value in probabilities.items())
    if total <= 0:
        return []
    return [(label, max(0.0, float(value)) / total) for label, value in ranked]


def _average_ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        rank = ((start + 1) + end) / 2.0
        for position in range(start, end):
            ranks[order[position]] = rank
        start = end
    return ranks


def spearman_correlation(x_values: list[float], y_values: list[float]) -> float | None:
    """Descriptive Spearman rho with average ranks for ties; no significance claim."""
    if len(x_values) != len(y_values) or len(x_values) < 2:
        return None
    try:
        xs = [float(value) for value in x_values]
        ys = [float(value) for value in y_values]
    except (TypeError, ValueError):
        return None
    if any(not math.isfinite(value) for value in (*xs, *ys)):
        return None
    rx, ry = _average_ranks(xs), _average_ranks(ys)
    mean_x, mean_y = sum(rx) / len(rx), sum(ry) / len(ry)
    covariance = sum((x - mean_x) * (y - mean_y) for x, y in zip(rx, ry))
    variance_x = sum((x - mean_x) ** 2 for x in rx)
    variance_y = sum((y - mean_y) ** 2 for y in ry)
    denominator = math.sqrt(variance_x * variance_y)
    return covariance / denominator if denominator else None


def self_rating_summary(rows: list[dict], mode: str = "blind", min_ratings: int = 10,
                         min_sessions: int = 3, min_session_ratings: int = 3) -> dict:
    """Describe equal-weight per-session rank association for one rating mode."""
    if mode not in {"blind", "visible"}:
        raise ValueError("Режим самооценки должен быть blind или visible")
    by_session: dict[int, list[tuple[float, float]]] = {}
    seen_observations: set[tuple[int, str]] = set()
    for row in rows:
        row_mode = row.get("rating_mode", "visible")
        if row_mode != mode:
            continue
        try:
            raw_rating = row["rating"]
            if isinstance(raw_rating, bool):
                continue
            rating = float(raw_rating)
            score = float(row["face_index"])
            quality = float(row["face_quality"])
            session_id = int(row["session_id"])
        except (KeyError, TypeError, ValueError):
            continue
        if (math.isfinite(rating) and math.isfinite(score) and math.isfinite(quality)
                and rating.is_integer() and 0 <= rating <= 10 and 0 <= score <= 100 and 0 <= quality <= 1
                and quality >= 0.45):
            observed_at = row.get("observed_at")
            if observed_at is not None:
                try:
                    timestamp = float(observed_at)
                except (TypeError, ValueError):
                    continue
                if not math.isfinite(timestamp):
                    continue
                observation_key = (session_id, timestamp)
                if observation_key in seen_observations:
                    continue
                seen_observations.add(observation_key)
            by_session.setdefault(session_id, []).append((rating, score))
    session_correlations: list[float] = []
    valid_count = 0
    for values in by_session.values():
        if len(values) < min_session_ratings:
            continue
        x_ranks = _average_ranks([rating for rating, _ in values])
        y_ranks = _average_ranks([score for _, score in values])
        if len(set(x_ranks)) < 2 or len(set(y_ranks)) < 2:
            continue
        session_correlations.append(spearman_correlation(
            [rating for rating, _ in values], [score for _, score in values]))
        valid_count += len(values)
    result = {"ratings": valid_count, "sessions": len(session_correlations), "rho": None,
              "session_rhos": tuple(session_correlations), "mode": mode}
    if valid_count < min_ratings or len(session_correlations) < min_sessions:
        result["reason"] = (f"Нужно не менее {min_ratings} качественных отметок и {min_sessions} сессий "
                             f"с изменением и оценки, и лицевого сигнала (минимум {min_session_ratings} "
                             "отметки в одной сессии).")
        return result
    result["rho"] = median(session_correlations)
    result["session_rho_min"] = min(session_correlations)
    result["session_rho_max"] = max(session_correlations)
    result["reason"] = None
    return result


def evaluate_rows(rows: list[dict[str, str]], labels: list[str]) -> dict:
    required = {"true_label", "predicted_label", "model_version", "preprocessing", "session_id"}
    if rows and not required.issubset(rows[0]):
        raise ValueError(f"В манифесте нужны столбцы: {', '.join(sorted(required))}")
    groups: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in rows:
        key = (row["model_version"], row["preprocessing"])
        groups.setdefault(key, []).append(row)
    results = []
    for (model_version, preprocessing), group in sorted(groups.items()):
        report = classification_report(
            [row["true_label"] for row in group],
            [row["predicted_label"] for row in group],
            labels,
        )
        results.append({
            "model_version": model_version,
            "preprocessing": preprocessing,
            "examples": len(group),
            "sessions": len({row["session_id"] for row in group}),
            "accuracy": report.accuracy,
            "macro_f1": report.macro_f1,
            "per_class": {label: vars(metric) for label, metric in report.per_class.items()},
            "confusion_matrix": report.confusion_matrix,
        })
    return {"labels": labels, "comparisons": results}


def evaluate_csv(manifest_path: str | Path, output_path: str | Path, labels: list[str]) -> dict:
    with Path(manifest_path).open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    result = evaluate_rows(rows, labels)
    Path(output_path).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result
