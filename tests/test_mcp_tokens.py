import pytest

from rtrace.mcp_server import ConfirmationTokenError, ConfirmationTokenIssuer


def test_confirmation_token_is_bound_to_exact_action_and_arguments():
    issuer = ConfirmationTokenIssuer("test-secret-at-least-16", ttl_seconds=60, clock=lambda: 100)
    token = issuer.issue("payments.execute", {"intent_id": "intent_17"})
    issuer.verify(token, "payments.execute", {"intent_id": "intent_17"})
    with pytest.raises(ConfirmationTokenError, match="payload_mismatch"):
        issuer.verify(token, "payments.execute", {"intent_id": "other"})
    with pytest.raises(ConfirmationTokenError, match="action_mismatch"):
        issuer.verify(token, "calendar.delete", {"intent_id": "intent_17"})


def test_confirmation_token_expires():
    now = [100]
    issuer = ConfirmationTokenIssuer("test-secret-at-least-16", ttl_seconds=1, clock=lambda: now[0])
    token = issuer.issue("calendar.delete", {"event_id": "evt_17"})
    now[0] = 102
    with pytest.raises(ConfirmationTokenError, match="expired"):
        issuer.verify(token, "calendar.delete", {"event_id": "evt_17"})
