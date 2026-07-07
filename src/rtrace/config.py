from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG: dict[str, Any] = {
    "benchmark": {
        "train": 360,
        "development": 90,
        "calibration": 90,
        "final_clean": 120,
        "final_hard": 120,
        "final_compositional": 120,
        "seed": 17,
    },
    "routing": {
        "c3_review_capacity": 0.20,
        "c4_review_capacity": 0.20,
        "c5_review_capacity_candidates": [0.12, 0.14, 0.16, 0.18, 0.20, 0.22, 0.24],
        "conservative_multiplier": 0.85,
        "scope_f1_tolerance": 0.02,
    },
    "runtime": {"bootstrap_samples": 2000},
    "models": {
        "lightgbm_estimators": 100,
        "random_state": 17,
        "calibration_regularization_candidates": [0.005, 0.01, 0.02, 0.05, 0.1],
    },
}


def default_config_path() -> Path:
    return Path(__file__).with_name("default.yaml")


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if key not in base:
            raise ValueError(f"unknown configuration section or key: {key}")
        if isinstance(value, dict):
            if not isinstance(base[key], dict):
                raise ValueError(f"configuration key is not a mapping: {key}")
            merged[key] = _deep_merge(base[key], value)
        else:
            merged[key] = value
    return merged


def _require_int(name: str, value: Any, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _require_fraction(name: str, value: Any, lower: float = 0.0, upper: float = 1.0) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not lower < numeric < upper:
        raise ValueError(f"{name} must be in ({lower}, {upper})")
    return numeric


def validate_config(config: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(config, dict):
        raise ValueError("configuration must be a mapping")
    result = copy.deepcopy(config)
    for section in ("benchmark", "routing", "runtime", "models"):
        if not isinstance(result.get(section), dict):
            raise ValueError(f"configuration section must be a mapping: {section}")
    benchmark = result["benchmark"]
    for key in [
        "train",
        "development",
        "calibration",
        "final_clean",
        "final_hard",
        "final_compositional",
    ]:
        benchmark[key] = _require_int(f"benchmark.{key}", benchmark[key])
    benchmark["seed"] = _require_int("benchmark.seed", benchmark["seed"], minimum=0)

    routing = result["routing"]
    for key in ["c3_review_capacity", "c4_review_capacity"]:
        routing[key] = _require_fraction(f"routing.{key}", routing[key])
    candidates = routing["c5_review_capacity_candidates"]
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("routing.c5_review_capacity_candidates must be a non-empty list")
    routing["c5_review_capacity_candidates"] = sorted(
        {_require_fraction("routing.c5_review_capacity_candidates", value) for value in candidates}
    )
    routing["conservative_multiplier"] = _require_fraction(
        "routing.conservative_multiplier", routing["conservative_multiplier"]
    )
    try:
        tolerance = float(routing["scope_f1_tolerance"])
    except (TypeError, ValueError) as exc:
        raise ValueError("routing.scope_f1_tolerance must be numeric") from exc
    if not 0.0 <= tolerance <= 0.25:
        raise ValueError("routing.scope_f1_tolerance must be in [0.0, 0.25]")
    routing["scope_f1_tolerance"] = tolerance

    runtime = result["runtime"]
    runtime["bootstrap_samples"] = _require_int(
        "runtime.bootstrap_samples", runtime["bootstrap_samples"], minimum=100
    )
    models = result["models"]
    models["lightgbm_estimators"] = _require_int(
        "models.lightgbm_estimators", models["lightgbm_estimators"], minimum=1
    )
    models["random_state"] = _require_int("models.random_state", models["random_state"], minimum=0)
    regularization = models["calibration_regularization_candidates"]
    if not isinstance(regularization, list) or not regularization:
        raise ValueError("models.calibration_regularization_candidates must be a non-empty list")
    normalized_regularization: list[float] = []
    for value in regularization:
        try:
            numeric = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "models.calibration_regularization_candidates must be numeric"
            ) from exc
        if not 0.0 < numeric <= 1.0:
            raise ValueError("models.calibration_regularization_candidates must be in (0.0, 1.0]")
        normalized_regularization.append(numeric)
    models["calibration_regularization_candidates"] = sorted(set(normalized_regularization))
    return result


def load_config(path: str | Path | None = None) -> tuple[dict[str, Any], str]:
    selected = Path(path) if path is not None else default_config_path()
    if not selected.exists():
        raise FileNotFoundError(f"configuration file does not exist: {selected}")
    try:
        loaded = yaml.safe_load(selected.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"configuration file cannot be read: {selected}") from exc
    except yaml.YAMLError as exc:
        raise ValueError(f"configuration YAML is invalid: {selected}") from exc
    if loaded is None:
        loaded = {}
    if not isinstance(loaded, dict):
        raise ValueError("top-level configuration must be a mapping")
    return validate_config(_deep_merge(DEFAULT_CONFIG, loaded)), str(selected.resolve())
