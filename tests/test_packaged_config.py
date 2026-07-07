from pathlib import Path

from rtrace.config import default_config_path, load_config


def test_packaged_default_config_is_available_and_valid() -> None:
    path = default_config_path()
    assert isinstance(path, Path)
    assert path.name == "default.yaml"
    assert path.exists()
    config, resolved = load_config()
    assert Path(resolved) == path.resolve()
    assert config["benchmark"]["train"] > 0


def test_ci_smoke_config_is_available_and_valid() -> None:
    path = Path("configs/ci_smoke.yaml")
    assert path.exists()
    config, _ = load_config(path)
    assert config["benchmark"]["train"] < 360
    assert config["runtime"]["bootstrap_samples"] == 100
