from __future__ import annotations

import copy
import random
from dataclasses import dataclass
from typing import Any

from .parser import RequestParseError, parse_request
from .policy import PREFLIGHT_ACTION
from .schemas import ActionCandidate, TaskCard


@dataclass
class LocalActor:
    """Deterministic local behavior simulator used only for CPU evaluation.

    The simulator parses the user-visible request text and never reads evaluator
    ``gold_*`` fields or actor-visible target mappings. It produces typed candidates
    plus observable parser/agent failure signals. It is not a substitute for a
    trained LLM; QLoRA evidence remains a separate GPU path.
    """

    name: str
    error_rate: float
    seed_offset: int = 0

    def _rng(self, task: TaskCard, seed: int, salt: int = 0) -> random.Random:
        return random.Random(
            seed + self.seed_offset + salt + sum(ord(char) for char in task.task_id)
        )

    @staticmethod
    def _request_action_and_args(task: TaskCard) -> tuple[str, dict[str, Any]]:
        parsed = parse_request(task.user_request)
        return parsed.action, copy.deepcopy(parsed.args)

    def propose(self, task: TaskCard, seed: int) -> ActionCandidate:
        rng = self._rng(task, seed)
        tags: list[str] = []
        try:
            action, args = self._request_action_and_args(task)
        except RequestParseError:
            action, args = f"{task.domain}.parse_error", {}
            tags.append("parser_error")
        if rng.random() < self.error_rate:
            error_kind = rng.choice(
                [
                    "missing",
                    "forbidden",
                    "wrong_target",
                    "missing_confirmation",
                    "duplicate",
                    "wrong_action",
                    "ambiguous",
                    "malformed",
                ]
            )
            mutable_keys = [key for key in args if key not in {"confirmed", "clarified"}]
            if error_kind == "missing" and mutable_keys:
                args.pop(rng.choice(mutable_keys), None)
                tags.append("missing_required")
            elif error_kind == "forbidden":
                if "attendees" in args and isinstance(args["attendees"], list):
                    args["attendees"] = list(args["attendees"]) + ["external@example.test"]
                elif "fields" in args and isinstance(args["fields"], list):
                    args["fields"] = list(args["fields"]) + ["ssn"]
                else:
                    args["extra_permission"] = "admin"
                tags.append("forbidden_scope")
            elif error_kind == "wrong_target":
                for key in ("contact_id", "recipient_id", "event_id", "intent_id"):
                    if key in args:
                        args[key] = f"wrong_{args[key]}"
                        break
                tags.append("wrong_target")
            elif error_kind == "missing_confirmation":
                args["confirmed"] = False
                tags.append("missing_confirmation")
            elif error_kind == "duplicate":
                tags.append("duplicate")
            elif error_kind == "wrong_action":
                action = f"{task.domain}.wrong_action"
                tags.append("wrong_action")
            elif error_kind == "ambiguous":
                args["clarified"] = False
                tags.append("ambiguity")
            elif error_kind == "malformed":
                tags.append("malformed")
        return ActionCandidate(
            task_id=task.task_id,
            domain=task.domain,
            action=action,
            args=args,
            trace_step=2 if task.requires_preflight else 1,
            idempotency_key=f"{task.task_id}:{seed}:{self.name}",
            source=self.name,
            confidence=max(0.05, min(0.98, 0.90 - self.error_rate + rng.uniform(-0.18, 0.08))),
            ambiguity_signal=0.85 if "ambiguity" in tags else rng.uniform(0.0, 0.25),
            synthetic_error_tags=tags,
        )

    def propose_pool(self, task: TaskCard, seed: int, size: int = 3) -> list[ActionCandidate]:
        """Return top-k request-grounded typed alternatives for scope correction."""
        pool = [self.propose(task, seed + 997 * rank) for rank in range(size)]
        return sorted(pool, key=lambda candidate: candidate.confidence, reverse=True)

    def propose_preflight(self, task: TaskCard, seed: int) -> ActionCandidate | None:
        if not task.requires_preflight:
            return None
        rng = self._rng(task, seed, salt=7001)
        try:
            requested_action, args = self._request_action_and_args(task)
        except RequestParseError:
            return None
        action = PREFLIGHT_ACTION.get(requested_action)
        if action is None:
            return None
        target_key = {
            "calendar.lookup": "event_id",
            "contacts.lookup": "contact_id",
            "payments.lookup": "intent_id",
        }[action]
        target_value = args.get(target_key)
        tags: list[str] = []
        if target_value is None:
            return None
        if rng.random() < self.error_rate * 0.75:
            if rng.random() < 0.5:
                target_value = f"missing_{target_value}"
                tags.append("preflight_wrong_target")
            else:
                tags.append("preflight_skipped")
        return ActionCandidate(
            task_id=task.task_id,
            domain=task.domain,
            action=action,
            args={target_key: target_value},
            trace_step=1,
            idempotency_key=f"{task.task_id}:{seed}:{self.name}:preflight",
            source=f"{self.name}:preflight",
            confidence=max(0.05, min(0.98, 0.88 - self.error_rate + rng.uniform(-0.12, 0.08))),
            ambiguity_signal=0.0,
            synthetic_error_tags=tags,
        )


class DeterministicWorkflowActor(LocalActor):
    def __init__(self) -> None:
        super().__init__(name="C0_deterministic", error_rate=0.34, seed_offset=101)


class PromptedFunctionActor(LocalActor):
    def __init__(self) -> None:
        super().__init__(name="C1_prompted", error_rate=0.26, seed_offset=201)


class GenericSFTActor(LocalActor):
    def __init__(self) -> None:
        super().__init__(name="C2_generic_sft", error_rate=0.16, seed_offset=301)
