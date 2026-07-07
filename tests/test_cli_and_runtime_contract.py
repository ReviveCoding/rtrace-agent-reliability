from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from rtrace.cli import main


def test_cli_validate_and_run_all_smoke(tmp_path: Path) -> None:
    validate_output = tmp_path / "validate"
    main(
        [
            "validate-data",
            "--output",
            str(validate_output),
            "--seed",
            "17",
            "--config",
            "configs/ci_smoke.yaml",
        ]
    )
    assert (validate_output / "data_quality.json").is_file()

    run_output = tmp_path / "run"
    main(
        [
            "run-all",
            "--output",
            str(run_output),
            "--seed",
            "17",
            "--config",
            "configs/ci_smoke.yaml",
        ]
    )
    assert (run_output / "run_manifest.json").is_file()
    assert (run_output / "reports" / "evaluation_report.md").is_file()


def test_docker_and_make_contracts_keep_runtime_context_lean() -> None:
    dockerfile = Path("Dockerfile").read_text(encoding="utf-8")
    dockerignore = Path(".dockerignore").read_text(encoding="utf-8")
    makefile = Path("Makefile").read_text(encoding="utf-8")

    assert "COPY docs" not in dockerfile
    assert "COPY scripts" not in dockerfile
    assert "apt-get install --yes --no-install-recommends libgomp1" in dockerfile
    assert "python -m pip install ." in dockerfile
    assert "python -m pip check" in dockerfile
    assert "import lightgbm, matplotlib, rtrace" in dockerfile
    for excluded in [
        "artifacts",
        ".venv",
        ".github",
        "*.egg-info",
        "dist",
        "tests",
        "scripts",
        "docs",
    ]:
        assert excluded in dockerignore
    assert "verify: test lint typecheck build smoke" in makefile
    assert "--overwrite" in makefile
    assert "coverage.json" in makefile
    assert "src/rtrace_agentic_evaluation.egg-info" in makefile


def test_cli_fails_cleanly_for_unsafe_resumable_output() -> None:
    command = [
        sys.executable,
        "-m",
        "rtrace.cli",
        "run-multiseed",
        "--output",
        ".",
        "--seeds",
        "17",
        "--config",
        "configs/ci_smoke.yaml",
    ]
    env = os.environ.copy()
    source_root = str((Path.cwd() / "src").resolve())
    env["PYTHONPATH"] = source_root + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    completed = subprocess.run(command, check=False, text=True, capture_output=True, env=env)
    assert completed.returncode == 2
    assert "rtrace: error:" in completed.stderr
    assert "current working directory" in completed.stderr
