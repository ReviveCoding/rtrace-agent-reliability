from rtrace.mcp_server import tool_manifest


def test_mcp_manifest_declares_fully_implemented_surface_and_safety_boundary():
    manifest = tool_manifest()
    assert len(manifest["tools"]) == 9
    assert {"calendar_delete", "payments_execute", "contacts_read"}.issubset(manifest["tools"])
    assert "host_verified_confirmation_for_high_impact_writes" in manifest["safety"]
    assert "not an externally security-audited" in manifest["claim_boundary"]
