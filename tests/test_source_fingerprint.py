from pathlib import Path

from rtrace.utils import hardware_metadata, source_fingerprint


def test_source_fingerprint_covers_project_contract(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "module.py").write_text("value = 1\n", encoding="utf-8")
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "default.yaml").write_text("run: 1\n", encoding="utf-8")
    (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n", encoding="utf-8")
    first = source_fingerprint(tmp_path)
    (tmp_path / "src" / "module.py").write_text("value = 2\n", encoding="utf-8")
    second = source_fingerprint(tmp_path)
    assert first != second
    assert first != "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"


def test_source_fingerprint_ignores_generated_egg_metadata(tmp_path: Path) -> None:
    package = tmp_path / "src" / "rtrace"
    package.mkdir(parents=True)
    (package / "module.py").write_text("value = 1\n", encoding="utf-8")
    metadata = tmp_path / "src" / "rtrace_agentic_evaluation.egg-info"
    metadata.mkdir()
    (metadata / "PKG-INFO").write_text("version = 1\n", encoding="utf-8")

    first = source_fingerprint(tmp_path)
    (metadata / "PKG-INFO").write_text("version = 2\n", encoding="utf-8")
    second = source_fingerprint(tmp_path)
    assert first == second

    (package / "module.py").write_text("value = 2\n", encoding="utf-8")
    assert source_fingerprint(tmp_path) != second


def test_hardware_metadata_records_core_package_versions() -> None:
    metadata = hardware_metadata()
    versions = metadata["package_versions"]
    assert isinstance(versions, dict)
    for package_name in [
        "numpy",
        "pandas",
        "scikit-learn",
        "lightgbm",
        "matplotlib",
        "pydantic",
        "PyYAML",
    ]:
        assert package_name in versions
