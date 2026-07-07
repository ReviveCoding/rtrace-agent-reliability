import asyncio

import pytest

from rtrace.mcp_server import ConfirmationTokenError, ConfirmationTokenIssuer, create_server
from rtrace.state import StateStore


def test_fastmcp_registration_and_bound_confirmation_smoke() -> None:
    pytest.importorskip("fastmcp")

    async def scenario() -> None:
        secret = "local-test-secret-123456789"
        with StateStore(":memory:", state_seed=17) as store:
            server = create_server(store, confirmation_secret=secret)
            tools = await server.get_tools()
            assert len(tools) == 9
            assert "calendar_delete" in tools

            before = store.digest()
            with pytest.raises(ConfirmationTokenError, match="token_required"):
                await tools["calendar_delete"].run(
                    {
                        "event_id": "evt_0",
                        "idempotency_key": "mcp-forged",
                        "confirmation_token": "forged",
                    }
                )
            assert store.digest() == before

            token = ConfirmationTokenIssuer(secret).issue("calendar.delete", {"event_id": "evt_0"})
            result = await tools["calendar_delete"].run(
                {
                    "event_id": "evt_0",
                    "idempotency_key": "mcp-valid",
                    "confirmation_token": token,
                }
            )
            assert result.structured_content["success"] is True
            assert store.digest() != before

    asyncio.run(scenario())
