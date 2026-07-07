#!/usr/bin/env python3
"""Fail-closed verification for a materialized R-TRACE ``run-all`` output directory."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

REQUIRED_FILES = (
    "data_quality.json",
    "metrics.json",
    "candidate_comparison.csv",
    "slice_metrics.csv",
    "predictions.csv",
    "calibration_table.csv",
    "threshold_frontier.csv",
    "reference_selection.csv",
    "reference_ablation.csv",
    "calibration_regularization_selection.csv",
    "operating_point_selection.csv",
    "paired_comparison.csv",
    "failure_cases.csv",
    "decision_scenarios.csv",
    "operational_metrics.json",
    "core_artifact_manifest.json",
    "run_manifest.json",
    "incident_replays/incident_replay_summary.json",
    "reports/evaluation_report.md",
    "reports/release_card.md",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def verify(output: Path) -> dict[str, Any]:
    if not output.exists() or not output.is_dir():
        raise ValueError(f"output directory does not exist: {output}")
    missing = [relative for relative in REQUIRED_FILES if not (output / relative).is_file()]
    if missing:
        raise ValueError(f"missing required output files: {missing}")

    data_quality = load_json(output / "data_quality.json")
    if data_quality.get("status") != "PASS":
        raise ValueError("data quality status is not PASS")
    run_manifest = load_json(output / "run_manifest.json")
    if run_manifest.get("core_artifact_integrity") is not True:
        raise ValueError("run manifest does not attest core artifact integrity")
    incident_summary = load_json(output / "incident_replays" / "incident_replay_summary.json")
    if incident_summary.get("status") != "PASS":
        raise ValueError("incident replay status is not PASS")

    core_manifest = load_json(output / "core_artifact_manifest.json")
    records = core_manifest.get("artifacts")
    if not isinstance(records, list) or not records:
        raise ValueError("core artifact manifest is missing artifact records")
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("invalid core artifact record")
        relative = record.get("path")
        expected_size = record.get("bytes")
        expected_hash = record.get("sha256")
        if (
            not isinstance(relative, str)
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
        ):
            raise ValueError(f"unsafe core artifact path: {relative!r}")
        path = output / relative
        if not path.is_file():
            raise ValueError(f"core artifact missing: {relative}")
        if path.stat().st_size != expected_size:
            raise ValueError(f"core artifact size mismatch: {relative}")
        if sha256(path) != expected_hash:
            raise ValueError(f"core artifact SHA-256 mismatch: {relative}")

    return {
        "status": "PASS",
        "output": str(output.resolve()),
        "run_id": run_manifest.get("run_id"),
        "seed": run_manifest.get("seed"),
        "release_verdict": run_manifest.get("release", {}).get("verdict"),
        "verified_core_artifacts": len(records),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        result = verify(args.output)
    except (OSError, ValueError) as exc:
        print(f"VERIFY_OUTPUT_FAILED: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
