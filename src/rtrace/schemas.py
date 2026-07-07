from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

SplitName = Literal[
    "train",
    "development",
    "calibration",
    "final_clean",
    "final_hard",
    "final_compositional",
]


class ImpactTier(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    IRREVERSIBLE = "irreversible"


class Route(StrEnum):
    ALLOW = "ALLOW"
    CLARIFY = "CLARIFY"
    CONFIRM = "CONFIRM"
    BLOCK = "BLOCK"
    COMPENSATE = "COMPENSATE"


class TaskCard(BaseModel):
    """Scenario contract with an explicit runtime/evaluator boundary.

    ``gold_*`` and scope fields are evaluator-only. Runtime components must use only
    ``user_request``, ``policy_ids``, ``runtime_context``, ``agent_context``, state
    observations, and candidate actions. ``agent_context`` is a synthetic stand-in
    for a parser's request-grounded representation. It is deliberately separated
    from evaluator labels so local actor simulators cannot read gold fields.
    """

    task_id: str
    split: SplitName
    domain: Literal["calendar", "contacts", "payments"]
    template_family: str
    state_seed: int
    user_request: str
    policy_ids: list[str]
    impact_tier: ImpactTier
    required_scope: dict[str, Any]
    forbidden_scope: dict[str, list[Any]] = Field(default_factory=dict)
    gold_action: str
    gold_args: dict[str, Any]
    requires_confirmation: bool = False
    ambiguity_required: bool = False
    failure_family: str = "normal"
    milestones: list[str] = Field(default_factory=list)
    runtime_context: dict[str, Any] = Field(default_factory=dict)
    agent_context: dict[str, Any] = Field(default_factory=dict)
    surface_template_id: str = ""
    composition_id: str = ""
    requires_preflight: bool = False

    @field_validator("required_scope")
    @classmethod
    def nonempty_scope(cls, value: dict[str, Any]) -> dict[str, Any]:
        if not value:
            raise ValueError("required_scope must not be empty")
        return value

    @field_validator("agent_context")
    @classmethod
    def valid_agent_context(cls, value: dict[str, Any]) -> dict[str, Any]:
        """Keep evaluator targets out of the actor-visible context.

        Runtime actors parse ``user_request`` directly. ``agent_context`` is reserved
        for parser metadata and may not contain action or argument payloads.
        """
        if not value.get("parser_contract"):
            raise ValueError("agent_context.parser_contract is required")
        forbidden = {"requested_action", "requested_args", "gold_action", "gold_args"}
        present = sorted(forbidden & set(value))
        if present:
            raise ValueError(f"agent_context contains evaluator-like fields: {present}")
        return value


class ActionCandidate(BaseModel):
    task_id: str
    domain: str
    action: str
    args: dict[str, Any]
    trace_step: int = 1
    idempotency_key: str
    source: str
    confidence: float = 0.5
    ambiguity_signal: float = 0.0
    synthetic_error_tags: list[str] = Field(default_factory=list)


class PolicyAssessment(BaseModel):
    """Evaluator-side oracle assessment, never a serving-time feature source."""

    hard_deny: bool
    policy_prior: float = Field(ge=0.0, le=1.0)
    reasons: list[str]
    critical_label: int
    missing_required: list[str] = Field(default_factory=list)
    forbidden_selected: list[str] = Field(default_factory=list)


class RuntimeAssessment(BaseModel):
    """Serving-time policy assessment from observable runtime information only."""

    hard_deny: bool
    runtime_prior: float = Field(ge=0.0, le=1.0)
    reasons: list[str]
    schema_valid: bool
    missing_schema_fields: list[str] = Field(default_factory=list)
    unknown_argument_fields: list[str] = Field(default_factory=list)
    protected_field_selected: bool = False
    preflight_required: bool = False
    preflight_succeeded: bool = False
    target_exists: bool | None = None


class TraceEvent(BaseModel):
    transaction_id: str
    task_id: str
    action: str
    route: Route
    status: Literal[
        "executed",
        "observed",
        "blocked",
        "clarify",
        "confirm",
        "partial",
        "compensated",
        "error",
    ]
    state_before: str
    state_after: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ExecutionResult(BaseModel):
    success: bool
    partial: bool = False
    transaction_id: str
    message: str
    state_digest: str
    compensable: bool = False


class EvaluationRecord(BaseModel):
    task_id: str
    candidate_id: str
    candidate: str
    split: str
    domain: str
    impact_tier: str
    failure_family: str
    risk_label: int
    risk_score: float
    route: Route
    safe_completion: int
    autonomous_safe_completion: int
    assisted_safe_completion: int
    safe_intervention: int
    false_greenlight: int
    overblock: int
    confirmation_used: int
    scope_recall: float
    scope_precision: float
    unauthorized_scope: int
    harm_cost: float
    preflight_succeeded: int
    partial_failure: int = 0
    compensation_attempted: int = 0
    compensation_succeeded: int = 0
    latency_ms: float
    reason_codes: list[str]


class ReleaseDecision(BaseModel):
    verdict: Literal["PASS", "REVIEW", "BLOCK"]
    reasons: list[str]
    gates: dict[str, bool]
    recommended_mode: Literal["conservative", "standard", "capacity"]
