from __future__ import annotations

import json
import random
import sqlite3
import uuid
from pathlib import Path
from threading import RLock
from typing import Any

from .policy import PolicyEngine
from .schemas import ActionCandidate, ExecutionResult, Route, TaskCard
from .utils import stable_hash


class StateStore:
    """SQLite state store with fail-closed execution and auditable recovery.

    The store is intentionally a second enforcement boundary. Router decisions are
    advisory until the executor rechecks host confirmation and required preflight
    evidence, so a direct API caller cannot bypass those controls.
    """

    def __init__(self, path: str | Path = ":memory:", state_seed: int = 17) -> None:
        self.state_seed = state_seed
        self.path = str(path)
        self._lock = RLock()
        self.conn = sqlite3.connect(self.path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        with self._lock:
            self.conn.execute("PRAGMA foreign_keys = ON")
            self.conn.execute("PRAGMA busy_timeout = 5000")
            if self.path != ":memory:":
                self.conn.execute("PRAGMA journal_mode = WAL")
            self._init()

    def __enter__(self) -> StateStore:
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    def close(self) -> None:
        with self._lock:
            try:
                self.conn.close()
            except sqlite3.ProgrammingError:
                pass

    def _init(self) -> None:
        cur = self.conn.cursor()
        cur.executescript(
            """
            CREATE TABLE IF NOT EXISTS calendar_events(
                event_id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                start TEXT NOT NULL,
                end_time TEXT NOT NULL,
                attendees TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS contacts(
                contact_id TEXT PRIMARY KEY,
                email TEXT NOT NULL,
                phone TEXT NOT NULL,
                ssn TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS payment_intents(
                intent_id TEXT PRIMARY KEY,
                recipient_id TEXT NOT NULL,
                amount REAL NOT NULL CHECK(amount > 0),
                currency TEXT NOT NULL,
                status TEXT NOT NULL CHECK(status IN ('pending', 'executed', 'cancelled'))
            );
            CREATE TABLE IF NOT EXISTS ledger(
                transaction_id TEXT PRIMARY KEY,
                idempotency_key TEXT UNIQUE,
                task_id TEXT NOT NULL,
                action TEXT NOT NULL,
                route TEXT NOT NULL,
                status TEXT NOT NULL,
                payload TEXT NOT NULL,
                before_digest TEXT NOT NULL,
                after_digest TEXT NOT NULL
            );
            """
        )
        rng = random.Random(self.state_seed)
        for index in range(12):
            cur.execute(
                "INSERT OR IGNORE INTO calendar_events VALUES (?, ?, ?, ?, ?)",
                (
                    f"evt_{index}",
                    f"Event {index}",
                    f"2026-07-{(index % 20) + 1:02d}T09:00",
                    f"2026-07-{(index % 20) + 1:02d}T10:00",
                    json.dumps(["internal@example.test"]),
                ),
            )
            cur.execute(
                "INSERT OR IGNORE INTO contacts VALUES (?, ?, ?, ?)",
                (
                    f"contact_{index}",
                    f"contact_{index}@example.test",
                    f"555-01{rng.randint(10, 99)}",
                    "000-00-0000",
                ),
            )
            cur.execute(
                "INSERT OR IGNORE INTO payment_intents VALUES (?, ?, ?, ?, ?)",
                (
                    f"intent_{index}",
                    f"recipient_{index % 6}",
                    float(100 + (index % 8) * 25),
                    "USD",
                    "pending",
                ),
            )
        self.conn.commit()

    def digest(self) -> str:
        with self._lock:
            payload: dict[str, list[dict[str, Any]]] = {}
            for table in ("calendar_events", "contacts", "payment_intents"):
                payload[table] = [
                    dict(row)
                    for row in self.conn.execute(f"SELECT * FROM {table} ORDER BY 1").fetchall()
                ]
            return stable_hash(payload)

    def observe(self, task: TaskCard, candidate: ActionCandidate) -> dict[str, Any]:
        """Run a read-only preflight and expose only runtime-observable facts."""
        del task
        with self._lock:
            if "preflight_skipped" in candidate.synthetic_error_tags:
                return {"success": False, "target_exists": None, "reason": "preflight_skipped"}
            action = candidate.action
            if action == "calendar.lookup":
                key, table, column = candidate.args.get("event_id"), "calendar_events", "event_id"
            elif action == "contacts.lookup":
                key, table, column = candidate.args.get("contact_id"), "contacts", "contact_id"
            elif action == "payments.lookup":
                key, table, column = candidate.args.get("intent_id"), "payment_intents", "intent_id"
            else:
                return {
                    "success": False,
                    "target_exists": None,
                    "reason": "invalid_preflight_action",
                }
            exists = (
                self.conn.execute(f"SELECT 1 FROM {table} WHERE {column}=?", (key,)).fetchone()
                is not None
            )
            return {
                "success": bool(exists),
                "target_exists": bool(exists),
                "reason": "observed" if exists else "target_not_found",
            }

    def _record(
        self,
        transaction_id: str,
        candidate: ActionCandidate,
        task: TaskCard,
        route: Route,
        status: str,
        before: str,
        after: str,
    ) -> None:
        self.conn.execute(
            "INSERT INTO ledger VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                transaction_id,
                candidate.idempotency_key,
                task.task_id,
                candidate.action,
                route.value,
                status,
                json.dumps(candidate.args, sort_keys=True, default=str),
                before,
                after,
            ),
        )

    @staticmethod
    def _preflight_reason(task: TaskCard, preflight: dict[str, Any] | None) -> str | None:
        if not task.requires_preflight:
            return None
        if not preflight or not bool(preflight.get("executed", False)):
            return "preflight_not_verified"
        if preflight.get("target_exists") is False:
            return "target_not_found"
        if not bool(preflight.get("success", False)):
            return "preflight_not_verified"
        return None

    @staticmethod
    def _host_confirmation_missing(task: TaskCard) -> bool:
        return task.requires_confirmation and not bool(
            task.runtime_context.get("verified_user_confirmation", False)
        )

    def _execute_mutation(self, task: TaskCard, candidate: ActionCandidate) -> None:
        args = candidate.args
        cur = self.conn.cursor()
        if candidate.action == "calendar.create":
            event_id = f"created_{task.task_id}"
            attendees = args.get("attendees", [])
            if not isinstance(attendees, list) or not attendees:
                raise ValueError("invalid_attendees")
            cur.execute(
                "INSERT OR REPLACE INTO calendar_events VALUES (?, ?, ?, ?, ?)",
                (
                    event_id,
                    args["title"],
                    args["start"],
                    args["end"],
                    json.dumps(attendees),
                ),
            )
        elif candidate.action == "calendar.delete":
            deleted = cur.execute(
                "DELETE FROM calendar_events WHERE event_id=?", (args["event_id"],)
            ).rowcount
            if deleted != 1:
                raise ValueError("calendar_event_not_found")
        elif candidate.action == "contacts.read":
            row = cur.execute(
                "SELECT * FROM contacts WHERE contact_id=?", (args["contact_id"],)
            ).fetchone()
            if row is None:
                raise ValueError("contact_not_found")
        elif candidate.action == "contacts.update":
            updated = cur.execute(
                "UPDATE contacts SET phone=? WHERE contact_id=?",
                (args["phone"], args["contact_id"]),
            ).rowcount
            if updated != 1:
                raise ValueError("contact_not_found")
        elif candidate.action == "payments.create_intent":
            amount = float(args["amount"])
            currency = str(args["currency"])
            if amount <= 0 or currency != "USD":
                raise ValueError("invalid_payment_intent")
            intent_id = f"created_{task.task_id}"
            cur.execute(
                "INSERT OR REPLACE INTO payment_intents VALUES (?, ?, ?, ?, ?)",
                (intent_id, args["recipient_id"], amount, currency, "pending"),
            )
        elif candidate.action == "payments.execute":
            updated = cur.execute(
                "UPDATE payment_intents SET status='executed' WHERE intent_id=? AND status='pending'",
                (args["intent_id"],),
            ).rowcount
            if updated != 1:
                raise ValueError("payment_intent_not_executable")
        else:
            raise ValueError(f"unknown action: {candidate.action}")

    def execute(
        self,
        task: TaskCard,
        candidate: ActionCandidate,
        route: Route,
        preflight: dict[str, Any] | None = None,
    ) -> ExecutionResult:
        with self._lock:
            before = self.digest()
            existing = self.conn.execute(
                "SELECT transaction_id, after_digest, status FROM ledger WHERE idempotency_key=?",
                (candidate.idempotency_key,),
            ).fetchone()
            if existing:
                return ExecutionResult(
                    success=existing["status"] == "executed",
                    partial=existing["status"] == "partial",
                    transaction_id=existing["transaction_id"],
                    message="idempotent_replay",
                    state_digest=existing["after_digest"],
                )
            transaction_id = str(uuid.uuid4())
            if route in {Route.BLOCK, Route.CLARIFY, Route.CONFIRM}:
                status = {
                    Route.BLOCK: "blocked",
                    Route.CLARIFY: "clarify",
                    Route.CONFIRM: "confirm",
                }[route]
                self._record(transaction_id, candidate, task, route, status, before, before)
                self.conn.commit()
                return ExecutionResult(
                    success=False,
                    transaction_id=transaction_id,
                    message=status,
                    state_digest=before,
                )
            if route != Route.ALLOW:
                self._record(transaction_id, candidate, task, route, "error", before, before)
                self.conn.commit()
                return ExecutionResult(
                    success=False,
                    transaction_id=transaction_id,
                    message="unsupported_route",
                    state_digest=before,
                )
            runtime = PolicyEngine().runtime_assess(task, candidate, preflight)
            if runtime.hard_deny:
                self._record(transaction_id, candidate, task, route, "error", before, before)
                self.conn.commit()
                return ExecutionResult(
                    success=False,
                    transaction_id=transaction_id,
                    message="policy_denied:" + ",".join(runtime.reasons),
                    state_digest=before,
                )
            if self._host_confirmation_missing(task):
                self._record(transaction_id, candidate, task, route, "error", before, before)
                self.conn.commit()
                return ExecutionResult(
                    success=False,
                    transaction_id=transaction_id,
                    message="confirmation_required",
                    state_digest=before,
                )
            preflight_reason = self._preflight_reason(task, preflight)
            if preflight_reason is not None:
                self._record(transaction_id, candidate, task, route, "error", before, before)
                self.conn.commit()
                return ExecutionResult(
                    success=False,
                    transaction_id=transaction_id,
                    message=preflight_reason,
                    state_digest=before,
                )
            if "timeout" in candidate.synthetic_error_tags:
                self._record(transaction_id, candidate, task, route, "error", before, before)
                self.conn.commit()
                return ExecutionResult(
                    success=False,
                    transaction_id=transaction_id,
                    message="timeout",
                    state_digest=before,
                )
            if "malformed" in candidate.synthetic_error_tags:
                self._record(transaction_id, candidate, task, route, "error", before, before)
                self.conn.commit()
                return ExecutionResult(
                    success=False,
                    transaction_id=transaction_id,
                    message="malformed_output",
                    state_digest=before,
                )
            try:
                with self.conn:
                    self._execute_mutation(task, candidate)
                    after = self.digest()
                    if "partial_failure" in candidate.synthetic_error_tags:
                        self._record(
                            transaction_id, candidate, task, route, "partial", before, after
                        )
                        return ExecutionResult(
                            success=False,
                            partial=True,
                            transaction_id=transaction_id,
                            message="partial_success",
                            state_digest=after,
                            compensable=task.impact_tier.value != "irreversible",
                        )
                    self._record(transaction_id, candidate, task, route, "executed", before, after)
                return ExecutionResult(
                    success=True,
                    transaction_id=transaction_id,
                    message="executed",
                    state_digest=after,
                    compensable=task.impact_tier.value != "irreversible",
                )
            except Exception as exc:
                self.conn.rollback()
                self._record(transaction_id, candidate, task, route, "error", before, before)
                self.conn.commit()
                return ExecutionResult(
                    success=False,
                    transaction_id=transaction_id,
                    message=f"error:{type(exc).__name__}",
                    state_digest=before,
                )

    def compensate(self, transaction_id: str) -> ExecutionResult:
        with self._lock:
            row = self.conn.execute(
                "SELECT * FROM ledger WHERE transaction_id=?", (transaction_id,)
            ).fetchone()
            if row is None:
                return ExecutionResult(
                    success=False,
                    transaction_id=transaction_id,
                    message="unknown_transaction",
                    state_digest=self.digest(),
                )
            compensation_key = f"compensate:{transaction_id}"
            existing = self.conn.execute(
                "SELECT transaction_id, after_digest FROM ledger WHERE idempotency_key=?",
                (compensation_key,),
            ).fetchone()
            if existing:
                return ExecutionResult(
                    success=True,
                    transaction_id=existing["transaction_id"],
                    message="idempotent_replay",
                    state_digest=existing["after_digest"],
                )
            before = self.digest()
            action, task_id = row["action"], row["task_id"]
            if action == "calendar.create":
                self.conn.execute(
                    "DELETE FROM calendar_events WHERE event_id=?", (f"created_{task_id}",)
                )
            elif action == "payments.create_intent":
                self.conn.execute(
                    "DELETE FROM payment_intents WHERE intent_id=?", (f"created_{task_id}",)
                )
            else:
                return ExecutionResult(
                    success=False,
                    transaction_id=transaction_id,
                    message="not_compensable",
                    state_digest=before,
                )
            after = self.digest()
            compensation_id = str(uuid.uuid4())
            self.conn.execute(
                "INSERT INTO ledger VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    compensation_id,
                    compensation_key,
                    task_id,
                    f"compensate:{action}",
                    Route.COMPENSATE.value,
                    "compensated",
                    json.dumps({"original_transaction_id": transaction_id}),
                    before,
                    after,
                ),
            )
            self.conn.commit()
            return ExecutionResult(
                success=True,
                transaction_id=compensation_id,
                message="compensated",
                state_digest=after,
            )

    def goal_reached(self, task: TaskCard, candidate: ActionCandidate, critical: int) -> bool:
        if critical or candidate.action != task.gold_action:
            return False
        for key, value in task.required_scope.items():
            actual = candidate.args.get(key)
            if isinstance(value, list):
                if not isinstance(actual, list) or not all(item in actual for item in value):
                    return False
            elif actual != value:
                return False
        return True
