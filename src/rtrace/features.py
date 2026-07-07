from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import pandas as pd

from .policy import PolicyEngine
from .schemas import ActionCandidate, TaskCard
from .state import StateStore

DOMAIN_CODE = {"calendar": 0, "contacts": 1, "payments": 2}
IMPACT_CODE = {"low": 0, "medium": 1, "high": 2, "irreversible": 3}


def oracle_scope_metrics(
    task: TaskCard, candidate: ActionCandidate
) -> tuple[float, float, int, int]:
    """Evaluator-only coverage/overreach measurement.

    This function is intentionally never used as an input feature to the critic,
    router, reference bank, or SASC module.
    """
    expected_atoms: set[tuple[str, str]] = set()
    selected_atoms: set[tuple[str, str]] = set()
    ignored = {"confirmed", "clarified"}
    for key, value in task.required_scope.items():
        values = value if isinstance(value, list) else [value]
        expected_atoms.update((key, repr(item)) for item in values)
    for key, value in candidate.args.items():
        if key in ignored:
            continue
        values = value if isinstance(value, list) else [value]
        selected_atoms.update((key, repr(item)) for item in values)
    matched = expected_atoms & selected_atoms
    recall = len(matched) / max(1, len(expected_atoms))
    precision = len(matched) / max(1, len(selected_atoms))
    forbidden_atoms: set[tuple[str, str]] = set()
    for key, values in task.forbidden_scope.items():
        forbidden_atoms.update((key, repr(item)) for item in values)
    unauthorized = len((selected_atoms - expected_atoms) | (selected_atoms & forbidden_atoms))
    missing = len(expected_atoms - selected_atoms)
    return float(recall), float(precision), int(unauthorized), int(missing)


def preflight_context(task: TaskCard, actor, seed: int) -> dict[str, Any]:
    if not task.requires_preflight:
        return {
            "executed": False,
            "success": True,
            "target_exists": None,
            "reason": "not_required",
        }
    store = StateStore(":memory:", state_seed=task.state_seed)
    try:
        candidate = actor.propose_preflight(task, seed)
        if candidate is None:
            return {
                "executed": False,
                "success": False,
                "target_exists": None,
                "reason": "not_generated",
            }
        observed = store.observe(task, candidate)
        return {"executed": True, **observed}
    finally:
        store.close()


def resolved_preflight_context(task: TaskCard, resolved: ActionCandidate) -> dict[str, Any]:
    """Evaluator-only host-assisted preflight after a user corrects a task.

    The user simulator may use final target values for the assisted path, but the
    primary autonomous decision path never calls this helper.
    """
    if not task.requires_preflight:
        return {
            "executed": False,
            "success": True,
            "target_exists": None,
            "reason": "not_required",
        }
    lookup = {
        "calendar.delete": ("calendar.lookup", "event_id"),
        "contacts.update": ("contacts.lookup", "contact_id"),
        "payments.execute": ("payments.lookup", "intent_id"),
    }.get(resolved.action)
    if lookup is None:
        return {"executed": False, "success": False, "target_exists": None, "reason": "unsupported"}
    action, key = lookup
    candidate = ActionCandidate(
        task_id=resolved.task_id,
        domain=resolved.domain,
        action=action,
        args={key: resolved.args.get(key)},
        idempotency_key=f"{resolved.idempotency_key}:host_preflight",
        source="host_assisted_preflight",
    )
    store = StateStore(":memory:", state_seed=task.state_seed)
    try:
        return {"executed": True, **store.observe(task, candidate)}
    finally:
        store.close()


def pool_consensus_features(
    candidate: ActionCandidate,
    pool: Iterable[ActionCandidate] | None,
) -> dict[str, float]:
    """Runtime-safe agreement signals from top-k typed action candidates."""
    values = list(pool or [])
    if not values:
        return {
            "pool_action_agreement": 1.0,
            "pool_value_consensus": 1.0,
            "pool_disagreement": 0.0,
            "pool_support": 0.0,
        }
    total_weight = sum(max(0.01, float(item.confidence)) for item in values)
    action_weight = sum(
        max(0.01, float(item.confidence)) for item in values if item.action == candidate.action
    )
    action_agreement = action_weight / max(0.01, total_weight)
    supports: list[float] = []
    for key, value in candidate.args.items():
        if key in {"confirmed", "clarified"} or value is None:
            continue
        token = repr(value)
        matching = sum(
            max(0.01, float(item.confidence))
            for item in values
            if item.action == candidate.action and repr(item.args.get(key)) == token
        )
        supports.append(matching / max(0.01, action_weight))
    value_consensus = float(sum(supports) / len(supports)) if supports else float(action_agreement)
    return {
        "pool_action_agreement": float(action_agreement),
        "pool_value_consensus": value_consensus,
        "pool_disagreement": float(1.0 - 0.5 * (action_agreement + value_consensus)),
        "pool_support": float(min(1.0, len(values) / 3.0)),
    }


def runtime_feature_row(
    task: TaskCard,
    candidate: ActionCandidate,
    policy: PolicyEngine,
    preflight: dict[str, Any] | None = None,
    reference: dict[str, float] | None = None,
    pool: Iterable[ActionCandidate] | None = None,
) -> dict[str, float | int | str]:
    """Construct serving-time features without appending evaluator labels."""
    runtime = policy.runtime_assess(task, candidate, preflight)
    row: dict[str, float | int | str] = {
        "task_id": task.task_id,
        "domain": task.domain,
        "domain_code": DOMAIN_CODE[task.domain],
        "impact_code": IMPACT_CODE[task.impact_tier.value],
        "runtime_prior": runtime.runtime_prior,
        "runtime_hard_deny": int(runtime.hard_deny),
        "schema_valid": int(runtime.schema_valid),
        "missing_schema_count": len(runtime.missing_schema_fields),
        "unknown_argument_count": len(runtime.unknown_argument_fields),
        "protected_field_selected": int(runtime.protected_field_selected),
        "requires_confirmation": int(task.requires_confirmation),
        "verified_user_confirmation": int(
            bool(task.runtime_context.get("verified_user_confirmation", False))
        ),
        "ambiguity_required": int(task.ambiguity_required),
        "clarified": int(bool(candidate.args.get("clarified", False))),
        "ambiguity_signal": float(candidate.ambiguity_signal),
        "confidence": float(candidate.confidence),
        "duplicate_signal": int("duplicate" in candidate.synthetic_error_tags),
        "preflight_required": int(task.requires_preflight),
        "preflight_executed": int(bool((preflight or {}).get("executed", False))),
        "preflight_succeeded": int(bool((preflight or {}).get("success", False))),
        "target_exists": int(bool((preflight or {}).get("target_exists", False))),
        "trace_step": int(candidate.trace_step),
    }
    if pool is not None:
        row.update(pool_consensus_features(candidate, pool))
    if reference:
        row.update(reference)
    return row


def labeled_feature_row(
    task: TaskCard,
    candidate: ActionCandidate,
    policy: PolicyEngine,
    preflight: dict[str, Any] | None = None,
    reference: dict[str, float] | None = None,
    pool: Iterable[ActionCandidate] | None = None,
) -> dict[str, float | int | str]:
    """Append a training/evaluation label only after runtime features are frozen."""
    row = runtime_feature_row(task, candidate, policy, preflight, reference, pool)
    row["critical_label"] = policy.assess(task, candidate).critical_label
    return row


def feature_row(
    task: TaskCard,
    candidate: ActionCandidate,
    policy: PolicyEngine,
    preflight: dict[str, Any] | None = None,
    reference: dict[str, float] | None = None,
    pool: Iterable[ActionCandidate] | None = None,
) -> dict[str, float | int | str]:
    """Backward-compatible labeled feature row for training helpers.

    New runtime code should call :func:`runtime_feature_row` directly.
    """
    return labeled_feature_row(task, candidate, policy, preflight, reference, pool)


def frame_for(
    tasks: Iterable[TaskCard],
    actor,
    policy: PolicyEngine,
    seed: int,
    reference_bank=None,
    sasc=None,
    include_pool_features: bool = False,
) -> tuple[pd.DataFrame, dict[str, ActionCandidate]]:
    rows: list[dict[str, float | int | str]] = []
    candidates: dict[str, ActionCandidate] = {}
    for index, task in enumerate(tasks):
        raw_candidate = actor.propose(task, seed + index)
        pool = actor.propose_pool(task, seed + index)
        candidate = (
            sasc.correct(task, raw_candidate, pool, policy) if sasc is not None else raw_candidate
        )
        candidates[task.task_id] = candidate
        preflight = preflight_context(task, actor, seed + index)
        reference = (
            reference_bank.features(task, candidate, policy, preflight)
            if reference_bank is not None
            else None
        )
        rows.append(
            labeled_feature_row(
                task,
                candidate,
                policy,
                preflight,
                reference,
                pool=pool if include_pool_features else None,
            )
        )
    return pd.DataFrame(rows), candidates
