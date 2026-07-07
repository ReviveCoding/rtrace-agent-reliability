import os
from pathlib import Path

import pytest

from rtrace.utils import prepare_output_dir, prepare_resume_output_dir


def _windows_symlink_privilege_unavailable(exc: OSError) -> bool:
    return os.name == "nt" and getattr(exc, "winerror", None) == 1314


def test_output_directory_fails_closed_without_overwrite(tmp_path: Path):
    output = tmp_path / "output"
    output.mkdir()
    (output / "existing.txt").write_text("preserve", encoding="utf-8")
    with pytest.raises(FileExistsError):
        prepare_output_dir(output)
    prepared = prepare_output_dir(output, overwrite=True)
    assert prepared == output.resolve()
    assert not (output / "existing.txt").exists()


def test_output_directory_rejects_non_directory_and_simulated_symlink(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    file_target = tmp_path / "file-output"
    file_target.write_text("not-a-directory", encoding="utf-8")
    with pytest.raises(ValueError, match="not a directory"):
        prepare_output_dir(file_target)

    # This branch must be tested even on standard Windows accounts, where creating
    # an actual symlink requires Developer Mode or SeCreateSymbolicLinkPrivilege.
    synthetic_link = tmp_path / "synthetic-link"
    original_is_symlink = Path.is_symlink

    def simulated_is_symlink(candidate: Path) -> bool:
        return candidate == synthetic_link or original_is_symlink(candidate)

    monkeypatch.setattr(Path, "is_symlink", simulated_is_symlink)
    with pytest.raises(ValueError, match="symlink"):
        prepare_output_dir(synthetic_link)


def test_output_directory_rejects_real_symlink_when_platform_allows_it(tmp_path: Path):
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    try:
        link.symlink_to(real, target_is_directory=True)
    except OSError as exc:
        if _windows_symlink_privilege_unavailable(exc):
            pytest.skip(
                "Windows symlink privilege is unavailable; simulated guard branch is tested above."
            )
        raise

    with pytest.raises(ValueError, match="symlink"):
        prepare_output_dir(link)


def test_windows_symlink_privilege_error_is_treated_as_an_optional_integration_skip(
    monkeypatch: pytest.MonkeyPatch,
):
    error = OSError("symlink privilege unavailable")
    error.winerror = 1314  # type: ignore[attr-defined]
    monkeypatch.setattr(os, "name", "nt")
    assert _windows_symlink_privilege_unavailable(error)


def test_output_directory_rejects_project_root(tmp_path: Path):
    project = tmp_path / "project"
    project.mkdir()
    (project / "pyproject.toml").write_text("[project]\nname='x'\n", encoding="utf-8")
    with pytest.raises(ValueError, match="project root"):
        prepare_output_dir(project)


def test_resume_output_directory_uses_same_safety_boundary(tmp_path: Path, monkeypatch):
    project = tmp_path / "project"
    project.mkdir()
    (project / "pyproject.toml").write_text("[project]\nname='x'\n", encoding="utf-8")
    monkeypatch.chdir(project)

    with pytest.raises(ValueError, match="current working directory"):
        prepare_resume_output_dir(project)
    with pytest.raises(ValueError, match="current working directory"):
        prepare_resume_output_dir(project.parent)

    output = prepare_resume_output_dir(project / "artifacts" / "multiseed")
    (output / "seed_17").mkdir()
    assert prepare_resume_output_dir(output) == output.resolve()
