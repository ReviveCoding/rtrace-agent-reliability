from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType


def _load_module() -> ModuleType:
    path = Path("scripts/probe_bitsandbytes.py")
    spec = importlib.util.spec_from_file_location("probe_bitsandbytes_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_bitsandbytes_probe_reports_missing_dependencies_without_nonzero_exit() -> None:
    completed = subprocess.run(
        [sys.executable, "-S", "scripts/probe_bitsandbytes.py", "--expected-cuda", "12.8"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["status"] == "MISSING"
    assert payload["ready"] is False


def test_bitsandbytes_probe_preserves_nonfatal_child_stderr_without_failing(
    monkeypatch,
) -> None:
    module = _load_module()
    monkeypatch.setattr(module.importlib.util, "find_spec", lambda _name: object())
    child_payload = {
        "status": "READY",
        "ready": True,
        "torch_cuda": "12.8",
        "bitsandbytes_version": "0.49.2",
    }
    completed = subprocess.CompletedProcess(
        args=["python"],
        returncode=0,
        stdout=json.dumps(child_payload) + "\n",
        stderr="W0000 triton not found; flop counting will not work for triton kernels\n",
    )
    monkeypatch.setattr(module.subprocess, "run", lambda *args, **kwargs: completed)
    payload = module._payload("12.8")
    assert payload["status"] == "READY"
    assert payload["ready"] is True
    assert "triton not found" in payload["child_stderr"]


def test_bitsandbytes_probe_marks_cuda_version_mismatch_not_ready(monkeypatch) -> None:
    module = _load_module()
    monkeypatch.setattr(module.importlib.util, "find_spec", lambda _name: object())
    child_payload = {"status": "READY", "ready": True, "torch_cuda": "12.6"}
    completed = subprocess.CompletedProcess(
        args=["python"], returncode=0, stdout=json.dumps(child_payload), stderr=""
    )
    monkeypatch.setattr(module.subprocess, "run", lambda *args, **kwargs: completed)
    payload = module._payload("12.8")
    assert payload["status"] == "CUDA_VERSION_MISMATCH"
    assert payload["ready"] is False


def test_bitsandbytes_probe_fails_closed_when_child_exits_nonzero(monkeypatch) -> None:
    module = _load_module()
    monkeypatch.setattr(module.importlib.util, "find_spec", lambda _name: object())
    completed = subprocess.CompletedProcess(
        args=["python"],
        returncode=7,
        stdout=json.dumps({"status": "READY", "ready": True, "torch_cuda": "12.8"}),
        stderr="native failure",
    )
    monkeypatch.setattr(module.subprocess, "run", lambda *args, **kwargs: completed)
    payload = module._payload("12.8")
    assert payload["status"] == "CHILD_EXIT_NONZERO"
    assert payload["ready"] is False
    assert payload["child_exit_code"] == 7
