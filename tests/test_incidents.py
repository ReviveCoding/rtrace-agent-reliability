from rtrace.incidents import replay_incidents


def test_incident_replays(tmp_path):
    result = replay_incidents(tmp_path, 17)
    assert result["status"] == "PASS"
    assert result["incidents"] == 12
    assert result["failed"] == 0
