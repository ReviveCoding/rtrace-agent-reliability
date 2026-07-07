from __future__ import annotations

import json
from pathlib import Path

from rtrace.pipeline import run_all
from rtrace.utils import file_sha256


def test_end_to_end_artifacts(tmp_path: Path) -> None:
    out = tmp_path / "run"
    result = run_all(out, 17, config_path="configs/ci_smoke.yaml")
    assert (out / "metrics.json").exists()
    assert (out / "candidate_comparison.csv").exists()
    assert (out / "operating_point_selection.csv").exists()
    assert (out / "calibration_regularization_selection.csv").exists()
    assert (out / "core_artifact_manifest.json").exists()
    assert (out / "reports" / "evaluation_report.md").exists()
    assert result["release"]["verdict"] in {"PASS", "REVIEW", "BLOCK"}
    assert result["release"]["gates"]["artifact_integrity"] is True

    inventory = json.loads((out / "core_artifact_manifest.json").read_text(encoding="utf-8"))[
        "artifacts"
    ]
    assert inventory
    for record in inventory:
        path = out / record["path"]
        assert path.is_file()
        assert path.stat().st_size == record["bytes"]
        assert file_sha256(path) == record["sha256"]
