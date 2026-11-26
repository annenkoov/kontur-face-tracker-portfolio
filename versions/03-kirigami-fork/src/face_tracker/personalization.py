"""Conservative, validation-gated personal logit-bias adapter."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import tempfile

from .evaluation import classification_report
from .labels import LABELS, default_sample_directory


MIN_SESSIONS_PER_LABEL = 6
MIN_VALIDATION_SESSIONS = 3
MIN_LABELS = 3
MIN_MACRO_F1_GAIN = 0.03
MAX_ACCURACY_LOSS = 0.05


@dataclass(frozen=True)
class PersonalAdapter:
    model_name: str
    preprocessing: str
    biases: dict[str, float]
    trained_at: str
    training_sessions: int
    validation_sessions: int
    baseline_macro_f1: float
    adapted_macro_f1: float
    baseline_accuracy: float
    adapted_accuracy: float
    covered_labels: tuple[str, ...]

    def apply(self, probabilities: dict[str, float]) -> dict[str, float]:
        if set(probabilities) != set(LABELS):
            raise ValueError("Adapter expects all eight expression probabilities")
        values = {label: float(probabilities[label]) for label in LABELS}
        if any(not math.isfinite(value) or not 0 <= value <= 1 for value in values.values()) or sum(values.values()) <= 0:
            raise ValueError("Adapter received invalid probabilities")
        total_probability = sum(values.values())
        values = {label: value / total_probability for label, value in values.items()}
        baseline_label = max(values, key=values.get)
        if baseline_label not in self.covered_labels:
            return values
        covered_mass = sum(values[label] for label in self.covered_labels)
        logits = {label: math.log(max(1e-8, values[label])) + self.biases.get(label, 0.0)
                  for label in self.covered_labels}
        peak = max(logits.values())
        exp = {label: math.exp(value - peak) for label, value in logits.items()}
        total = sum(exp.values())
        adapted = dict(values)
        for label, value in exp.items():
            adapted[label] = covered_mass * value / total
        if max(adapted, key=adapted.get) not in self.covered_labels:
            return values
        return adapted


@dataclass(frozen=True)
class PersonalizationResult:
    activated: bool
    reason: str
    adapter: PersonalAdapter | None = None
    diagnostics: dict | None = None


def _session_units(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        label = row.get("true_label")
        session = row.get("session_id")
        probabilities = row.get("probabilities")
        if label not in LABELS or not session or not isinstance(probabilities, dict):
            continue
        if any(label_name not in LABELS for label_name in probabilities):
            continue
        try:
            values = {name: float(probabilities[name]) for name in LABELS}
        except (KeyError, TypeError, ValueError):
            continue
        if any(not math.isfinite(value) or value < 0 for value in values.values()) or sum(values.values()) <= 0:
            continue
        total = sum(values.values())
        values = {name: value / total for name, value in values.items()}
        grouped.setdefault((str(session), label), []).append({
            "probabilities": values,
            "created_at": str(row.get("created_at", "")),
        })
    units = []
    for (session, label), items in grouped.items():
        averaged = {name: sum(item["probabilities"][name] for item in items) / len(items) for name in LABELS}
        units.append({"session_id": session, "label": label, "probabilities": averaged,
                      "created_at": min(item["created_at"] for item in items)})
    return units


def _macro_f1(units: list[dict], predictions: list[str]) -> tuple[float, float, dict]:
    report = classification_report([row["label"] for row in units], predictions, list(LABELS))
    return report.macro_f1, report.accuracy, {label: asdict(metric) for label, metric in report.per_class.items()}


def _fit_biases(units: list[dict], covered_labels: list[str], regularization: float = 0.4,
                steps: int = 500, learning_rate: float = 0.12) -> dict[str, float]:
    biases = {label: 0.0 for label in LABELS}
    counts = {label: sum(row["label"] == label for row in units) for label in LABELS}
    weights = [1.0 / counts[row["label"]] for row in units]
    weight_total = sum(weights)
    for step in range(steps):
        gradient = {label: regularization * biases[label] for label in LABELS}
        for row, sample_weight in zip(units, weights):
            logits = {label: math.log(max(1e-8, row["probabilities"][label])) + biases[label]
                      for label in LABELS}
            peak = max(logits.values())
            exp = {label: math.exp(value - peak) for label, value in logits.items()}
            total = sum(exp.values())
            for label in LABELS:
                target = 1.0 if row["label"] == label else 0.0
                gradient[label] += sample_weight * (exp[label] / total - target) / weight_total
        # A diminishing fixed-step gradient is stable for this eight-parameter convex problem.
        rate = learning_rate / math.sqrt(1.0 + step / 40.0)
        for label in covered_labels:
            biases[label] -= rate * gradient[label]
    # Classes absent from personal data retain their generic output unchanged.
    for label in LABELS:
        if not counts[label]:
            biases[label] = 0.0
    return biases


def fit_personal_adapter(rows: list[dict], model_name: str = "enet_b0_8_best_vgaf",
                         preprocessing: str = "tight") -> PersonalizationResult:
    """Train on earlier capture sessions, evaluate on later whole sessions, gate activation."""
    units = _session_units([row for row in rows if row.get("model_version") == model_name
                            and row.get("preprocessing") == preprocessing])
    sessions_by_label = {label: {row["session_id"] for row in units if row["label"] == label} for label in LABELS}
    eligible_labels = [label for label, sessions in sessions_by_label.items()
                       if len(sessions) >= MIN_SESSIONS_PER_LABEL]
    if len(eligible_labels) < MIN_LABELS:
        return PersonalizationResult(False,
            f"Нужно не менее {MIN_LABELS} выражений, каждое размечено в {MIN_SESSIONS_PER_LABEL}+ независимых сессиях.")

    session_times: dict[str, str] = {}
    for row in units:
        session_times[row["session_id"]] = min(session_times.get(row["session_id"], row["created_at"]), row["created_at"])
    sessions = sorted(session_times, key=lambda session: (session_times[session], session))
    holdout_count = max(MIN_VALIDATION_SESSIONS, math.ceil(len(sessions) * 0.2))
    holdout_sessions = set(sessions[-holdout_count:])
    train = [row for row in units if row["session_id"] not in holdout_sessions]
    validation = [row for row in units if row["session_id"] in holdout_sessions]
    if not train or not validation or any(not any(row["label"] == label for row in train) or
                                           not any(row["label"] == label for row in validation)
                                           for label in eligible_labels):
        return PersonalizationResult(False,
            "Отложенные по времени сессии не покрывают те же выражения, что и обучение.")

    eligible_train = [row for row in train if row["label"] in eligible_labels]
    biases = _fit_biases(eligible_train, eligible_labels)
    baseline_predictions = [max(row["probabilities"], key=row["probabilities"].get) for row in validation]
    temporary = PersonalAdapter(model_name, preprocessing, biases, "", 1, 2, 0, 1, 0, 1,
                                tuple(eligible_labels))
    adapted_probabilities = []
    for row in validation:
        adapted_probabilities.append(temporary.apply(row["probabilities"]))
    adapted_predictions = [max(probabilities, key=probabilities.get) for probabilities in adapted_probabilities]
    baseline_f1, baseline_accuracy, baseline_per_class = _macro_f1(validation, baseline_predictions)
    adapted_f1, adapted_accuracy, adapted_per_class = _macro_f1(validation, adapted_predictions)
    diagnostics = {
        "eligible_labels": eligible_labels,
        "training_sessions": len({row["session_id"] for row in train}),
        "validation_sessions": len(holdout_sessions),
        "training_units": len(train),
        "validation_units": len(validation),
        "baseline_macro_f1": baseline_f1,
        "adapted_macro_f1": adapted_f1,
        "baseline_accuracy": baseline_accuracy,
        "adapted_accuracy": adapted_accuracy,
        "baseline_per_class": baseline_per_class,
        "adapted_per_class": adapted_per_class,
    }
    improves = adapted_f1 >= baseline_f1 + MIN_MACRO_F1_GAIN
    accuracy_safe = adapted_accuracy >= baseline_accuracy - MAX_ACCURACY_LOSS
    if not improves or not accuracy_safe:
        return PersonalizationResult(False,
            "Новая адаптация не активирована: улучшения на отложенных сессиях недостаточно.",
            diagnostics=diagnostics)
    adapter = PersonalAdapter(
        model_name, preprocessing, biases, datetime.now(timezone.utc).isoformat(timespec="seconds"),
        diagnostics["training_sessions"], diagnostics["validation_sessions"], baseline_f1,
        adapted_f1, baseline_accuracy, adapted_accuracy, tuple(eligible_labels),
    )
    return PersonalizationResult(True, "Персональная поправка прошла проверку на отложенных сессиях.",
                                 adapter, diagnostics)


def default_adapter_path() -> Path:
    return default_sample_directory().parent / "personal_adapter.json"


def save_adapter(adapter: PersonalAdapter, destination: str | Path | None = None) -> Path:
    path = Path(destination) if destination is not None else default_adapter_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix="personal-adapter-", suffix=".tmp", dir=path.parent)
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        temporary_path.write_text(json.dumps(asdict(adapter), ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temporary_path, path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise
    return path


def load_adapter(source: str | Path | None = None, expected_model: str | None = None,
                 expected_preprocessing: str | None = None) -> PersonalAdapter | None:
    path = Path(source) if source is not None else default_adapter_path()
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or not isinstance(data.get("biases"), dict):
            return None
        if type(data.get("training_sessions")) is not int or type(data.get("validation_sessions")) is not int:
            return None
        if not isinstance(data.get("covered_labels"), (list, tuple)):
            return None
        data["covered_labels"] = tuple(data["covered_labels"])
        adapter = PersonalAdapter(**data)
        if expected_model is not None and adapter.model_name != expected_model:
            return None
        if expected_preprocessing is not None and adapter.preprocessing != expected_preprocessing:
            return None
        if (set(adapter.biases) != set(LABELS) or
                any(not math.isfinite(value) or abs(value) > 5 for value in adapter.biases.values())):
            return None
        if (len(set(adapter.covered_labels)) < MIN_LABELS or
                any(label not in LABELS for label in adapter.covered_labels)):
            return None
        if any(abs(adapter.biases[label]) > 1e-9 for label in LABELS if label not in adapter.covered_labels):
            return None
        metrics = (adapter.baseline_macro_f1, adapter.adapted_macro_f1,
                   adapter.baseline_accuracy, adapter.adapted_accuracy)
        if any(not math.isfinite(value) or not 0 <= value <= 1 for value in metrics):
            return None
        if (adapter.training_sessions < MIN_VALIDATION_SESSIONS or
                adapter.validation_sessions < MIN_VALIDATION_SESSIONS):
            return None
        if adapter.adapted_macro_f1 < adapter.baseline_macro_f1 + MIN_MACRO_F1_GAIN:
            return None
        if adapter.adapted_accuracy < adapter.baseline_accuracy - MAX_ACCURACY_LOSS:
            return None
        return adapter
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return None
