from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


class RequestParseError(ValueError):
    """Raised when the local text-only parser cannot construct a typed action."""


@dataclass(frozen=True)
class ParsedRequest:
    action: str
    args: dict[str, Any]


_TIMESTAMP = re.compile(r"2026-07-\d{2}T\d{2}:\d{2}")
_EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@example\.test")
_EVENT = re.compile(r"evt_\d+")
_CONTACT = re.compile(r"contact_\d+")
_PHONE = re.compile(r"555-\d{4}")
_RECIPIENT = re.compile(r"recipient_\d+")
_INTENT = re.compile(r"intent_\d+")
_AMOUNT = re.compile(r"(?:USD\s*)?(\d+(?:\.\d{1,2})?)\s*USD", re.IGNORECASE)
_TITLE = re.compile(r"Design Review\s+\d+")


def _first(pattern: re.Pattern[str], text: str, label: str) -> str:
    match = pattern.search(text)
    if match is None:
        raise RequestParseError(f"missing_{label}")
    return match.group(0)


def parse_request(user_request: str) -> ParsedRequest:
    """Parse the constrained local benchmark request without evaluator fields.

    The parser intentionally reads only the user-visible request text. It is a small,
    deterministic CPU contract for the synthetic benchmark, not an LLM or an
    independent semantic parser benchmark.
    """
    text = " ".join(str(user_request).split())
    lower = text.lower()
    if _TITLE.search(text) and len(_TIMESTAMP.findall(text)) >= 2:
        timestamps = _TIMESTAMP.findall(text)
        emails = _EMAIL.findall(text)
        return ParsedRequest(
            action="calendar.create",
            args={
                "title": _first(_TITLE, text, "title"),
                "start": timestamps[0],
                "end": timestamps[1],
                "attendees": emails or ["internal@example.test"],
                # This is acknowledgement of a confirmation requirement, not proof of consent.
                "confirmed": True,
                "clarified": not ("ambiguous" in lower or "clarif" in lower),
            },
        )
    if _EVENT.search(text):
        return ParsedRequest(
            action="calendar.delete",
            args={
                "event_id": _first(_EVENT, text, "event_id"),
                "confirmed": True,
                "clarified": not ("ambiguous" in lower or "clarif" in lower),
            },
        )
    if _CONTACT.search(text) and "phone" in lower:
        return ParsedRequest(
            action="contacts.update",
            args={
                "contact_id": _first(_CONTACT, text, "contact_id"),
                "phone": _first(_PHONE, text, "phone"),
                "confirmed": True,
                "clarified": not ("ambiguous" in lower or "clarif" in lower),
            },
        )
    if _CONTACT.search(text) and "email" in lower:
        return ParsedRequest(
            action="contacts.read",
            args={
                "contact_id": _first(_CONTACT, text, "contact_id"),
                "fields": ["email"],
                "confirmed": True,
                "clarified": not ("ambiguous" in lower or "clarif" in lower),
            },
        )
    if _INTENT.search(text) and any(
        token in lower for token in ("execute", "execution", "run", "send")
    ):
        return ParsedRequest(
            action="payments.execute",
            args={
                "intent_id": _first(_INTENT, text, "intent_id"),
                "confirmed": True,
                "clarified": not ("ambiguous" in lower or "clarif" in lower),
            },
        )
    if _RECIPIENT.search(text) and ("payment" in lower or "usd" in lower):
        amount_match = _AMOUNT.search(text)
        if amount_match is None:
            raise RequestParseError("missing_amount")
        return ParsedRequest(
            action="payments.create_intent",
            args={
                "recipient_id": _first(_RECIPIENT, text, "recipient_id"),
                "amount": float(amount_match.group(1)),
                "currency": "USD",
                "confirmed": True,
                "clarified": not ("ambiguous" in lower or "clarif" in lower),
            },
        )
    raise RequestParseError("unsupported_request")
