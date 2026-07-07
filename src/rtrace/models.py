from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from .features import preflight_context, runtime_feature_row
from .metrics import safe_average_precision
from .policy import PolicyEngine
from .schemas import ActionCandidate, TaskCard

LGBMClassifier: Any
try:
    from lightgbm import LGBMClassifier as _LGBMClassifier

    LGBMClassifier = _LGBMClassifier
except Exception:  # pragma: no cover
    LGBMClassifier = None


BASE_FEATURES = [
    "domain_code",
    "impact_code",
    "runtime_prior",
    "runtime_hard_deny",
    "schema_valid",
    "missing_schema_count",
    "unknown_argument_count",
    "protected_field_selected",
    "requires_confirmation",
    "verified_user_confirmation",
    "ambiguity_required",
    "clarified",
    "ambiguity_signal",
    "confidence",
    "duplicate_signal",
    "preflight_required",
    "preflight_executed",
    "preflight_succeeded",
    "target_exists",
    "trace_step",
]
POOL_FEATURES = [
    "pool_action_agreement",
    "pool_value_consensus",
    "pool_disagreement",
    "pool_support",
]
REFERENCE_FEATURES = [
    "reference_min_distance",
    "reference_mean_distance",
    "reference_safe_prior",
    "reference_action_known",
    "reference_support",
]


class _ConstantRiskModel:
    """Predictor fallback for degenerate training slices with one observed class."""

    def __init__(self, probability: float) -> None:
        self.probability = float(min(1 - 1e-5, max(1e-5, probability)))

    def predict_proba(self, matrix: pd.DataFrame) -> np.ndarray:
        positive = np.full(len(matrix), self.probability, dtype=float)
        return np.column_stack([1.0 - positive, positive])


def _vector(row: dict[str, float | int | str]) -> np.ndarray:
    return np.array([float(row[key]) for key in BASE_FEATURES], dtype=float)


class ReferenceBank:
    """Training-only safe trajectory-prefix references.

    References are indexed by observable ``(domain, candidate.action)`` rather than
    evaluator intent. Training labels choose safe prefixes, but final runtime
    features use only candidate output, policy and preflight observations.
    """

    def __init__(self, strategy: str = "farthest_first", max_prototypes: int = 5) -> None:
        if strategy not in {"farthest_first", "random"}:
            raise ValueError("strategy must be farthest_first or random")
        if max_prototypes < 1:
            raise ValueError("max_prototypes must be at least one")
        self.strategy = strategy
        self.max_prototypes = max_prototypes
        self.prototypes: dict[tuple[str, str], np.ndarray] = {}
        self.safe_prior: dict[tuple[str, str], float] = {}

    def _select(self, vectors: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        if len(vectors) <= self.max_prototypes:
            return vectors
        if self.strategy == "random":
            indices = rng.choice(len(vectors), size=self.max_prototypes, replace=False)
            return vectors[indices]
        selected = [int(np.argmin(np.linalg.norm(vectors - vectors.mean(axis=0), axis=1)))]
        while len(selected) < self.max_prototypes:
            chosen = vectors[selected]
            minimum_distance = np.min(
                np.linalg.norm(vectors[:, None, :] - chosen[None, :, :], axis=2), axis=1
            )
            selected.append(int(np.argmax(minimum_distance)))
        return vectors[selected]

    def fit(
        self,
        tasks: Iterable[TaskCard],
        actor,
        policy: PolicyEngine,
        seed: int,
    ) -> ReferenceBank:
        grouped: dict[tuple[str, str], list[np.ndarray]] = {}
        priors: dict[tuple[str, str], list[float]] = {}
        for index, task in enumerate(tasks):
            candidate = actor.propose(task, seed + index)
            preflight = preflight_context(task, actor, seed + index)
            oracle = policy.assess(task, candidate)
            runtime = policy.runtime_assess(task, candidate, preflight)
            if oracle.critical_label == 0 and not runtime.hard_deny:
                row = runtime_feature_row(task, candidate, policy, preflight)
                key: tuple[str, str] = (str(task.domain), candidate.action)
                grouped.setdefault(key, []).append(_vector(row))
                priors.setdefault(key, []).append(float(row["runtime_prior"]))
        rng = np.random.default_rng(seed)
        for key, vectors in grouped.items():
            self.prototypes[key] = self._select(np.vstack(vectors), rng)
            self.safe_prior[key] = float(np.mean(priors[key]))
        return self

    def features(
        self,
        task: TaskCard,
        candidate: ActionCandidate,
        policy: PolicyEngine,
        preflight: dict | None = None,
    ) -> dict[str, float]:
        row = runtime_feature_row(task, candidate, policy, preflight)
        vector = _vector(row)
        key: tuple[str, str] = (str(task.domain), candidate.action)
        if key not in self.prototypes:
            return {
                "reference_min_distance": 2.5,
                "reference_mean_distance": 2.5,
                "reference_safe_prior": 0.5,
                "reference_action_known": 0.0,
                "reference_support": 0.0,
            }
        references = self.prototypes[key]
        distances = np.linalg.norm(references - vector[None, :], axis=1) / max(
            1.0, np.sqrt(len(vector))
        )
        return {
            "reference_min_distance": float(min(3.0, distances.min())),
            "reference_mean_distance": float(min(3.0, distances.mean())),
            "reference_safe_prior": self.safe_prior[key],
            "reference_action_known": 1.0,
            "reference_support": float(min(1.0, len(references) / self.max_prototypes)),
        }


@dataclass
class RiskCritic:
    use_reference: bool
    use_pool_consensus: bool = False
    random_state: int = 17
    n_estimators: int = 150
    model: Any | None = None
    calibrator: Any | None = None
    features: list[str] | None = None
    calibration_regularization: float | None = None
    calibration_uses_runtime_prior: bool = False

    def fit(self, frame: pd.DataFrame) -> RiskCritic:
        self.features = (
            BASE_FEATURES
            + (REFERENCE_FEATURES if self.use_reference else [])
            + (POOL_FEATURES if self.use_pool_consensus else [])
        )
        missing = [feature for feature in self.features if feature not in frame]
        if missing:
            raise ValueError(f"training frame missing features: {missing}")
        matrix = frame[self.features].astype(float)
        labels = frame["critical_label"].astype(int)
        if labels.nunique() < 2:
            self.model = _ConstantRiskModel(float(labels.mean()))
            return self
        if LGBMClassifier is not None:
            base = LGBMClassifier(
                n_estimators=self.n_estimators,
                learning_rate=0.04,
                num_leaves=15,
                min_child_samples=12,
                subsample=0.9,
                colsample_bytree=0.9,
                random_state=self.random_state,
                verbosity=-1,
                n_jobs=1,
                force_col_wise=True,
            )
        else:
            base = HistGradientBoostingClassifier(
                max_iter=self.n_estimators,
                max_leaf_nodes=15,
                min_samples_leaf=12,
                random_state=self.random_state,
            )
        base.fit(matrix, labels)
        self.model = base
        return self

    @staticmethod
    def _calibration_matrix(raw: np.ndarray, frame: pd.DataFrame) -> np.ndarray:
        """Use an observable policy prior as a shrinkage feature for calibration.

        The learned critic estimates residual risk; ``runtime_prior`` is a
        deterministic serving-time policy signal. Combining them only in the
        calibrator keeps the base classifier interpretable while reducing extreme
        probabilities from a small calibration slice.
        """
        if "runtime_prior" not in frame:
            raise ValueError("calibration frame missing runtime_prior")
        logits = np.log(raw / (1 - raw))
        return np.column_stack([logits, frame["runtime_prior"].astype(float).to_numpy()])

    def calibrate(self, frame: pd.DataFrame, regularization: float = 0.05) -> RiskCritic:
        if self.model is None or self.features is None:
            raise RuntimeError("fit before calibrate")
        if regularization <= 0:
            raise ValueError("regularization must be positive")
        matrix = frame[self.features].astype(float)
        labels = frame["critical_label"].astype(int)
        raw = np.clip(self.model.predict_proba(matrix)[:, 1], 1e-5, 1 - 1e-5)
        self.calibration_regularization = float(regularization)
        self.calibration_uses_runtime_prior = True
        if labels.nunique() > 1:
            self.calibrator = LogisticRegression(
                C=float(regularization),
                random_state=self.random_state,
            ).fit(self._calibration_matrix(raw, frame), labels)
        return self

    def score(self, frame: pd.DataFrame) -> np.ndarray:
        if self.features is None or self.model is None:
            raise RuntimeError("critic is not fitted")
        missing = [feature for feature in self.features if feature not in frame]
        if missing:
            raise ValueError(f"scoring frame missing features: {missing}")
        raw = np.clip(
            self.model.predict_proba(frame[self.features].astype(float))[:, 1], 1e-5, 1 - 1e-5
        )
        if self.calibrator is None:
            return raw
        if self.calibration_uses_runtime_prior:
            matrix = self._calibration_matrix(raw, frame)
        else:  # Backward-compatible path for serialized legacy artifacts.
            matrix = np.log(raw / (1 - raw)).reshape(-1, 1)
        return self.calibrator.predict_proba(matrix)[:, 1]

    def ap(self, frame: pd.DataFrame) -> float:
        return safe_average_precision(frame["critical_label"].to_numpy(), self.score(frame))
