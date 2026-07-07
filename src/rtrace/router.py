from __future__ import annotations

import math
from dataclasses import dataclass

from .policy import PolicyEngine
from .schemas import ActionCandidate, Route, TaskCard


@dataclass
class DecisionRouter:
    threshold: float
    conservative_threshold: float

    def __post_init__(self) -> None:
        for value, name in (
            (self.threshold, "threshold"),
            (self.conservative_threshold, "conservative_threshold"),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")

    def route(
        self,
        task: TaskCard,
        candidate: ActionCandidate,
        risk_score: float,
        policy: PolicyEngine,
        preflight: dict | None = None,
        mode: str = "standard",
    ) -> tuple[Route, list[str]]:
        if mode not in {"standard", "conservative", "capacity"}:
            raise ValueError("mode must be standard, conservative, or capacity")
        assessment = policy.runtime_assess(task, candidate, preflight)
        if not math.isfinite(risk_score):
            return Route.BLOCK, assessment.reasons + ["non_finite_risk_score"]
        if assessment.target_exists is False:
            return Route.CLARIFY, assessment.reasons + ["target_not_found"]
        if assessment.preflight_required and not assessment.preflight_succeeded:
            return Route.BLOCK, assessment.reasons + ["preflight_not_verified"]
        if assessment.hard_deny:
            return Route.BLOCK, assessment.reasons
        threshold = (
            self.threshold if mode in {"standard", "capacity"} else self.conservative_threshold
        )
        if task.ambiguity_required and not candidate.args.get("clarified", False):
            return Route.CLARIFY, assessment.reasons + ["clarification_required"]
        if "confirmation_required" in assessment.reasons:
            return Route.CONFIRM, assessment.reasons
        if risk_score >= threshold:
            return Route.CONFIRM, assessment.reasons + ["risk_threshold"]
        return Route.ALLOW, assessment.reasons
