from pathlib import Path

import pytest

from rtrace.config import load_config
from rtrace.pipeline import run_all, run_multiseed


def _small_config(path: Path) -> Path:
    path.write_text(
        """
benchmark:
  train: 24
  development: 6
  calibration: 6
  final_clean: 8
  final_hard: 8
  final_compositional: 8
  seed: 41
routing:
  c3_review_capacity: 0.20
  c4_review_capacity: 0.20
  c5_review_capacity_candidates: [0.16, 0.20, 0.24]
  conservative_multiplier: 0.85
  scope_f1_tolerance: 0.02
runtime:
  bootstrap_samples: 100
models:
  lightgbm_estimators: 10
  random_state: 41
""".strip(),
        encoding="utf-8",
    )
    return path


def test_config_is_loaded_and_traced(tmp_path: Path):
    config_path = _small_config(tmp_path / "small.yaml")
    result = run_all(tmp_path / "run", config_path=config_path)
    manifest = result["manifest"]
    assert manifest["seed"] == 41
    assert manifest["benchmark_counts"]["final_clean"] == 8
    assert manifest["config_fingerprint"]
    assert (tmp_path / "run" / "incident_replays" / "incident_replay_summary.json").exists()


def test_invalid_config_is_rejected(tmp_path: Path):
    path = tmp_path / "bad.yaml"
    path.write_text("benchmark:\n  unknown: 1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unknown"):
        load_config(path)


def test_multiseed_rejects_artifacts_from_another_config(tmp_path: Path):
    config_a = _small_config(tmp_path / "a.yaml")
    root = tmp_path / "multiseed"
    run_multiseed(root, seeds=[41], config_path=config_a)
    config_b = _small_config(tmp_path / "b.yaml")
    config_b.write_text(
        config_b.read_text(encoding="utf-8").replace(
            "c5_review_capacity_candidates: [0.16, 0.20, 0.24]",
            "c5_review_capacity_candidates: [0.14, 0.18]",
        ),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="stale artifacts"):
        run_multiseed(root, seeds=[41], config_path=config_b)


def test_multiseed_recovers_an_incomplete_seed_directory(tmp_path: Path):
    config_path = _small_config(tmp_path / "small.yaml")
    root = tmp_path / "multiseed"
    incomplete = root / "seed_41"
    incomplete.mkdir(parents=True)
    (incomplete / "data_quality.json").write_text("{}", encoding="utf-8")
    result = run_multiseed(root, seeds=[41], config_path=config_path)
    assert result["seeds"] == [41]
    assert (incomplete / "run_manifest.json").exists()
