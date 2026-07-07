from __future__ import annotations

from pathlib import Path

import yaml


def test_ci_workflow_has_required_quality_and_artifact_guards() -> None:
    workflow_path = Path(".github/workflows/ci.yml")
    workflow = workflow_path.read_text(encoding="utf-8")
    parsed = yaml.safe_load(workflow)

    assert isinstance(parsed, dict)
    jobs = parsed.get("jobs")
    assert isinstance(jobs, dict)
    for job_name in [
        "test",
        "quality-package-smoke",
        "mcp-adapter-smoke",
        "windows-powershell-smoke",
        "docker-smoke",
    ]:
        assert job_name in jobs
    assert "pipeline-smoke" not in jobs

    assert "actions/checkout@v6" in workflow
    assert "persist-credentials: false" in workflow
    assert "actions/setup-python@v6" in workflow
    assert "cache: pip" in workflow
    assert 'python-version: ["3.11", "3.13"]' in workflow
    assert "PIP_NO_INPUT" in workflow
    assert "python -m pip check" in workflow
    assert "python -m pytest -p pytest_cov --cov=rtrace" in workflow
    assert 'if [ "${{ matrix.python-version }}" = "3.11" ]; then' in workflow
    assert "python -m ruff check src tests scripts" in workflow
    assert "python -m mypy src/rtrace" in workflow
    assert "python -m build" in workflow
    assert "mcp-adapter-smoke:" in workflow
    assert "actions/upload-artifact@v4" in workflow
    assert "if-no-files-found: warn" in workflow
    assert "rtrace-wheel-smoke-${{ github.sha }}" in workflow
    assert "windows-powershell-smoke:" in workflow
    assert "./scripts/run_local.ps1 -Mode run" in workflow
    assert "-ConfigPath configs/ci_smoke.yaml" in workflow
    assert "docker-smoke:" in workflow
    assert "docker build --tag rtrace-agentic-evaluation:ci ." in workflow
    assert ".wheel-venv/bin/rtrace run-all" in workflow
    assert "--config configs/ci_smoke.yaml" in workflow
    assert (
        ".wheel-venv/bin/python scripts/verify_output.py --output wheel-artifacts/run" in workflow
    )
    assert "docker cp rtrace-container:/app/artifacts/ci/. docker-artifacts" in workflow
    assert "python scripts/verify_output.py --output docker-artifacts" in workflow
    assert "PYTEST_DISABLE_PLUGIN_AUTOLOAD" in workflow
    assert "OMP_NUM_THREADS" in workflow
    assert "timeout-minutes: 20" in workflow


def test_docker_smoke_pins_the_host_python_used_for_artifact_verification() -> None:
    workflow = Path(".github/workflows/ci.yml").read_text(encoding="utf-8")
    docker_job = workflow.split("  docker-smoke:", maxsplit=1)[1]

    assert "actions/setup-python@v6" in docker_job
    assert 'python-version: "3.13"' in docker_job
    assert "python scripts/verify_output.py --output docker-artifacts" in docker_job
