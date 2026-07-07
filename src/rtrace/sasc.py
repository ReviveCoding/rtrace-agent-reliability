from __future__ import annotations

import copy
import json
from collections import defaultdict
from collections.abc import Iterable

from .policy import PolicyEngine
from .schemas import ActionCandidate, TaskCard


class StructuredActionScopeCorrection:
    """ABC-inspired candidate aggregation without evaluator-oracle access.

    Stage 1 inflates incomplete schema fields from confidence-weighted, compatible
    candidates. Stage 2 removes unsupported values and every argument outside the
    serving schema. It never reads evaluator scope or ``gold_*`` fields.
    """

    @staticmethod
    def _weighted_values(
        task: TaskCard,
        action: str,
        key: str,
        compatible: list[ActionCandidate],
        policy: PolicyEngine,
    ) -> tuple[dict[str, object], defaultdict[str, float]]:
        values: dict[str, object] = {}
        score: defaultdict[str, float] = defaultdict(float)
        for alternative in compatible:
            value = alternative.args.get(key)
            if value in (None, []) or not policy.value_allowed(task, action, key, value):
                continue
            token = json.dumps(value, sort_keys=True, default=str)
            values[token] = value
            score[token] += float(max(0.01, alternative.confidence))
        return values, score

    def correct(
        self,
        task: TaskCard,
        candidate: ActionCandidate,
        pool: Iterable[ActionCandidate],
        policy: PolicyEngine,
    ) -> ActionCandidate:
        args = copy.deepcopy(candidate.args)
        tags = list(candidate.synthetic_error_tags)
        compatible = [alternative for alternative in pool if alternative.action == candidate.action]
        if not compatible:
            compatible = [candidate]

        # Inflate missing required schema fields using request-grounded candidates.
        for key in policy.required_fields(candidate.action):
            if args.get(key) not in (None, []):
                continue
            values, score = self._weighted_values(task, candidate.action, key, compatible, policy)
            if score:
                selected = max(score, key=lambda token: float(score[token]))
                args[key] = copy.deepcopy(values[selected])
                tags.append(f"sasc_inflate:{key}")

        # Repair an unsupported low-consensus value only when the candidate pool
        # supplies strong independent support for a different runtime-authorized value.
        for key in policy.required_fields(candidate.action):
            current = args.get(key)
            if current in (None, []):
                continue
            values, score = self._weighted_values(task, candidate.action, key, compatible, policy)
            if not score:
                continue
            total = sum(score.values())
            current_token = json.dumps(current, sort_keys=True, default=str)
            mode_token = max(score, key=lambda token: float(score[token]))
            mode_support = score[mode_token] / max(0.01, total)
            current_support = score.get(current_token, 0.0) / max(0.01, total)
            # Do not replace a runtime-authorized value solely because the local
            # candidate pool prefers another one.  In a small synthetic benchmark,
            # that creates a self-consensus failure mode: several correlated
            # alternatives can agree on a different but still authorized target.
            # Consensus may correct only a value that violates the serving-time
            # authorization boundary; otherwise preserve the actor's primary choice.
            current_is_runtime_valid = policy.value_allowed(task, candidate.action, key, current)
            if (
                not current_is_runtime_valid
                and mode_token != current_token
                and mode_support >= 0.64
                and current_support <= 0.36
            ):
                args[key] = copy.deepcopy(values[mode_token])
                tags.append(f"sasc_consensus_repair:{key}")

        # Prune all unknown fields, not only a short denylist. This turns schema
        # validation into a strict scope boundary and prevents silent no-op fields.
        allowed_fields = policy.allowed_argument_fields(candidate.action)
        for key in list(args):
            if key not in allowed_fields or key in {
                "ssn",
                "password",
                "secret",
                "extra_permission",
            }:
                args.pop(key, None)
                tags.append(f"sasc_prune:{key}")
                continue
            value = args[key]
            if key not in {"confirmed", "clarified"} and not policy.value_allowed(
                task, candidate.action, key, value
            ):
                if isinstance(value, list):
                    allowed = task.runtime_context.get("allowed_values", {}).get(key, [])
                    args[key] = [item for item in value if item in allowed]
                else:
                    args.pop(key, None)
                tags.append(f"sasc_prune:{key}")

        # A pruning step may empty a required list. Reinflate only from runtime-safe
        # candidates and only when policy permits the replacement value.
        for key in policy.required_fields(candidate.action):
            if args.get(key) not in (None, []):
                continue
            values, score = self._weighted_values(task, candidate.action, key, compatible, policy)
            if score:
                selected = max(score, key=lambda token: float(score[token]))
                args[key] = copy.deepcopy(values[selected])
                tags.append(f"sasc_reinflate:{key}")

        return candidate.model_copy(
            update={
                "args": args,
                "synthetic_error_tags": tags,
                "source": candidate.source + "+SASC",
            }
        )
