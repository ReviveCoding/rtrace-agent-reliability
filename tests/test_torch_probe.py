from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_torch_probe_reports_missing_torch_without_nonzero_exit() -> None:
    script = Path("scripts/probe_torch_cuda.py")
    completed = subprocess.run(
        [sys.executable, "-S", str(script), "--expected-cuda", "12.8"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["status"] == "MISSING"
    assert payload["torch_installed"] is False
    assert payload["cuda_ready"] is False
    assert payload["ready"] is False
    assert payload["expected_cuda"] == "12.8"


def test_torch_probe_rejects_missing_expected_cuda_argument() -> None:
    completed = subprocess.run(
        [sys.executable, "-S", "scripts/probe_torch_cuda.py"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode != 0
    assert "--expected-cuda" in completed.stderr
