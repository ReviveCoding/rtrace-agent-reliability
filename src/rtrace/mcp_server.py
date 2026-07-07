"""Optional FastMCP adapter for SafeAssist-MCP.

The canonical benchmark remains local and deterministic. This adapter exposes the
same SQLite-backed tools after ``pip install -e '.[mcp]'``. High-impact mutations
require a host-issued, HMAC-signed, action-bound confirmation token. The issuance
helper is deliberately not exposed as an MCP tool.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import secrets
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .policy import PREFLIGHT_ACTION
from .schemas import ActionCandidate, ImpactTier, Route, TaskCard
from .state import StateStore
from .utils import stable_hash


class ConfirmationTokenError(ValueError):
    """Raised when a host confirmation token is missing, expired, or mismatched."""


@dataclass(frozen=True)
class ConfirmationTokenIssuer:
    """Host-side confirmation token issuer for local protocol integration tests.

    The host must keep the secret outside the agent context. A token is bound to an
    exact action and canonical argument payload, so it cannot authorize another
    tool or a modified amount/recipient after issuance.
    """

    secret: bytes | str
    ttl_seconds: int = 300
    clock: Callable[[], float] = time.time

    def __post_init__(self) -> None:
        secret = self.secret.encode("utf-8") if isinstance(self.secret, str) else self.secret
        if len(secret) < 16:
            raise ValueError("confirmation secret must be at least 16 bytes")
        if self.ttl_seconds < 1:
            raise ValueError("ttl_seconds must be positive")

    @property
    def secret_bytes(self) -> bytes:
        return self.secret.encode("utf-8") if isinstance(self.secret, str) else self.secret

    @staticmethod
    def _payload_hash(action: str, args: dict[str, Any]) -> str:
        return stable_hash({"action": action, "args": args})

    def issue(self, action: str, args: dict[str, Any]) -> str:
        payload = {
            "version": 1,
            "action": action,
            "args_hash": self._payload_hash(action, args),
            "expires_at": int(self.clock()) + self.ttl_seconds,
            "nonce": secrets.token_hex(12),
        }
        encoded = (
            base64.urlsafe_b64encode(
                json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
            )
            .decode("ascii")
            .rstrip("=")
        )
        signature = hmac.new(self.secret_bytes, encoded.encode("ascii"), hashlib.sha256).hexdigest()
        return f"{encoded}.{signature}"

    def verify(self, token: str | None, action: str, args: dict[str, Any]) -> None:
        if not token or "." not in token:
            raise ConfirmationTokenError("host_verified_confirmation_token_required")
        encoded, signature = token.rsplit(".", 1)
        expected_signature = hmac.new(
            self.secret_bytes, encoded.encode("ascii"), hashlib.sha256
        ).hexdigest()
        if not hmac.compare_digest(signature, expected_signature):
            raise ConfirmationTokenError("host_confirmation_signature_invalid")
        padded = encoded + "=" * (-len(encoded) % 4)
        try:
            payload = json.loads(base64.urlsafe_b64decode(padded.encode("ascii")))
        except (ValueError, json.JSONDecodeError) as exc:
            raise ConfirmationTokenError("host_confirmation_payload_invalid") from exc
        if int(payload.get("expires_at", 0)) < int(self.clock()):
            raise ConfirmationTokenError("host_confirmation_expired")
        if payload.get("action") != action:
            raise ConfirmationTokenError("host_confirmation_action_mismatch")
        if payload.get("args_hash") != self._payload_hash(action, args):
            raise ConfirmationTokenError("host_confirmation_payload_mismatch")


def _default_issuer(secret: str | bytes | None = None) -> ConfirmationTokenIssuer:
    value = secret or os.getenv("SAFEASSIST_CONFIRMATION_SECRET")
    if value is None:
        # Ephemeral secret is safe for a one-process local demo, but a host should
        # supply a persistent secret through its own secret manager in integrations.
        raw = secrets.token_bytes(32)
    elif isinstance(value, str):
        raw = value.encode("utf-8")
    else:
        raw = value
    return ConfirmationTokenIssuer(raw)


def tool_manifest() -> dict[str, Any]:
    return {
        "name": "safeassist-mcp",
        "tools": [
            "calendar_lookup",
            "calendar_create",
            "calendar_delete",
            "contacts_lookup",
            "contacts_read",
            "contacts_update",
            "payments_lookup",
            "payments_create_intent",
            "payments_execute",
        ],
        "safety": [
            "hmac_bound_host_verified_confirmation_for_high_impact_writes",
            "host_verified_confirmation_for_high_impact_writes",
            "protected_field_denial",
            "strict_argument_schema",
            "idempotency",
            "transaction_ledger",
            "preflight_enforcement",
            "atomic_payment_execution",
        ],
        "claim_boundary": "local protocol adapter; not an externally security-audited MCP deployment",
    }


def _runtime_task(action: str, impact: ImpactTier) -> TaskCard:
    """Create a runtime-only carrier without evaluator labels or benchmark fields."""
    return TaskCard.model_construct(
        task_id=f"mcp:{action}:{uuid.uuid4().hex}",
        split="train",
        domain=action.split(".", 1)[0],
        template_family="mcp_runtime",
        state_seed=17,
        user_request="runtime_mcp_tool_call",
        policy_ids=["MCP-RUNTIME"],
        impact_tier=impact,
        required_scope={"runtime": True},
        forbidden_scope={},
        gold_action="",
        gold_args={},
        runtime_context={"verified_user_confirmation": True},
        agent_context={"parser_contract": "mcp_runtime", "parser_source": "mcp_adapter"},
        requires_confirmation=False,
        requires_preflight=action in PREFLIGHT_ACTION,
    )


def _preflight_for(
    state: StateStore, task: TaskCard, action: str, args: dict[str, Any]
) -> dict[str, Any] | None:
    lookup = PREFLIGHT_ACTION.get(action)
    if lookup is None:
        return None
    key = {
        "calendar.lookup": "event_id",
        "contacts.lookup": "contact_id",
        "payments.lookup": "intent_id",
    }[lookup]
    candidate = ActionCandidate(
        task_id=task.task_id,
        domain=task.domain,
        action=lookup,
        args={key: args.get(key)},
        idempotency_key=f"{task.task_id}:preflight:{key}",
        source="safeassist_mcp_adapter",
    )
    return {"executed": True, **state.observe(task, candidate)}


def _mutate(
    state: StateStore,
    action: str,
    args: dict[str, Any],
    idempotency_key: str,
    impact: ImpactTier,
) -> dict[str, Any]:
    task = _runtime_task(action, impact)
    candidate = ActionCandidate(
        task_id=task.task_id,
        domain=task.domain,
        action=action,
        args=args,
        idempotency_key=idempotency_key,
        source="safeassist_mcp_adapter",
    )
    preflight = _preflight_for(state, task, action, args)
    return state.execute(task, candidate, Route.ALLOW, preflight=preflight).model_dump()


def create_server(
    store: StateStore | None = None,
    confirmation_secret: str | bytes | None = None,
):
    try:
        from fastmcp import FastMCP
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Install the optional MCP dependency: pip install -e '.[mcp]'") from exc
    state = store or StateStore(":memory:")
    issuer = _default_issuer(confirmation_secret)
    mcp = FastMCP("safeassist-mcp")

    @mcp.tool()
    def calendar_lookup(event_id: str) -> dict[str, Any]:
        task = _runtime_task("calendar.lookup", ImpactTier.LOW)
        candidate = ActionCandidate(
            task_id=task.task_id,
            domain="calendar",
            action="calendar.lookup",
            args={"event_id": event_id},
            idempotency_key=f"lookup:{event_id}",
            source="safeassist_mcp_adapter",
        )
        return state.observe(task, candidate)

    @mcp.tool()
    def calendar_create(
        title: str,
        start: str,
        end: str,
        attendees: list[str],
        idempotency_key: str,
    ) -> dict[str, Any]:
        if any(not value.endswith("@example.test") for value in attendees):
            raise ValueError("runtime_authorization_violation:external_attendee")
        return _mutate(
            state,
            "calendar.create",
            {"title": title, "start": start, "end": end, "attendees": attendees},
            idempotency_key,
            ImpactTier.MEDIUM,
        )

    @mcp.tool()
    def calendar_delete(
        event_id: str,
        idempotency_key: str,
        confirmation_token: str,
    ) -> dict[str, Any]:
        args = {"event_id": event_id}
        issuer.verify(confirmation_token, "calendar.delete", args)
        return _mutate(state, "calendar.delete", args, idempotency_key, ImpactTier.HIGH)

    @mcp.tool()
    def contacts_lookup(contact_id: str) -> dict[str, Any]:
        task = _runtime_task("contacts.lookup", ImpactTier.LOW)
        candidate = ActionCandidate(
            task_id=task.task_id,
            domain="contacts",
            action="contacts.lookup",
            args={"contact_id": contact_id},
            idempotency_key=f"lookup:{contact_id}",
            source="safeassist_mcp_adapter",
        )
        return state.observe(task, candidate)

    @mcp.tool()
    def contacts_read(contact_id: str, fields: list[str]) -> dict[str, Any]:
        if any(field in {"ssn", "password", "secret"} for field in fields):
            raise ValueError("protected_field_or_privilege")
        with state._lock:  # local adapter shares StateStore's thread-safe connection.
            row = state.conn.execute(
                "SELECT email, phone FROM contacts WHERE contact_id=?", (contact_id,)
            ).fetchone()
        if row is None:
            return {"success": False, "reason": "target_not_found"}
        return {
            "success": True,
            "contact_id": contact_id,
            "data": {field: row[field] for field in fields if field in {"email", "phone"}},
        }

    @mcp.tool()
    def contacts_update(contact_id: str, phone: str, idempotency_key: str) -> dict[str, Any]:
        return _mutate(
            state,
            "contacts.update",
            {"contact_id": contact_id, "phone": phone},
            idempotency_key,
            ImpactTier.MEDIUM,
        )

    @mcp.tool()
    def payments_lookup(intent_id: str) -> dict[str, Any]:
        task = _runtime_task("payments.lookup", ImpactTier.LOW)
        candidate = ActionCandidate(
            task_id=task.task_id,
            domain="payments",
            action="payments.lookup",
            args={"intent_id": intent_id},
            idempotency_key=f"lookup:{intent_id}",
            source="safeassist_mcp_adapter",
        )
        return state.observe(task, candidate)

    @mcp.tool()
    def payments_create_intent(
        recipient_id: str,
        amount: float,
        currency: str,
        idempotency_key: str,
    ) -> dict[str, Any]:
        if amount <= 0 or currency != "USD":
            raise ValueError("invalid_payment_intent")
        return _mutate(
            state,
            "payments.create_intent",
            {"recipient_id": recipient_id, "amount": amount, "currency": currency},
            idempotency_key,
            ImpactTier.HIGH,
        )

    @mcp.tool()
    def payments_execute(
        intent_id: str,
        idempotency_key: str,
        confirmation_token: str,
    ) -> dict[str, Any]:
        args = {"intent_id": intent_id}
        issuer.verify(confirmation_token, "payments.execute", args)
        return _mutate(state, "payments.execute", args, idempotency_key, ImpactTier.IRREVERSIBLE)

    return mcp
