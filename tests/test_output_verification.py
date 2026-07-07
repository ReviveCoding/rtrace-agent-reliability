from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from rtrace.pipeline import run_all


def test_output_verifier_accepts_run_and_rejects_tampering(tmp_path: Path) -> None:
    output = tmp_path / "run"
    run_all(output, seed=17, config_path="configs/ci_smoke.yaml")
    command = [sys.executable, "scripts/verify_output.py", "--output", str(output)]
    completed = subprocess.run(command, check=False, text=True, capture_output=True)
    assert completed.returncode == 0, completed.stderr
    assert '"status": "PASS"' in completed.stdout

    (output / "metrics.json").write_text("{}\n", encoding="utf-8")
    tampered = subprocess.run(command, check=False, text=True, capture_output=True)
    assert tampered.returncode == 2
    assert "VERIFY_OUTPUT_FAILED" in tampered.stderr
