from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import shutil
import subprocess
import tempfile
import time
from collections.abc import Iterable
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as distribution_version
from pathlib import Path
from typing import Any

import numpy as np


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def stable_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, default=str, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: str | Path) -> str:
    """Return the SHA-256 of a materialized artifact without loading it all at once."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ensure_dir(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _safe_output_path(path: str | Path) -> Path:
    """Resolve an output path while rejecting dangerous repository/system targets.

    The check is shared by one-shot and resumable commands.  It prevents accidental
    use of the repository root, its ancestors, a filesystem root, or a symlink as an
    output destination while still permitting a normal child such as ``artifacts/run``.
    """
    requested = Path(path).expanduser()
    if requested.is_symlink():
        raise ValueError("refusing to use a symlink as an output directory")
    output = requested.resolve()
    cwd = Path.cwd().resolve()
    if output == output.parent:
        raise ValueError("refusing to use a filesystem root as an output directory")
    if output == cwd or output in cwd.parents:
        raise ValueError(
            "refusing to use the current working directory or one of its ancestors as output"
        )
    if (output / "pyproject.toml").exists() or (output / ".git").exists():
        raise ValueError("refusing to use a project root as an output directory")
    if output.exists() and not output.is_dir():
        raise ValueError(f"output path exists but is not a directory: {output}")
    return output


def prepare_output_dir(path: str | Path, overwrite: bool = False) -> Path:
    """Create a clean output directory without silently clobbering source or system paths.

    ``--overwrite`` is intentionally fail-closed for the current working directory,
    its ancestors, a project root, filesystem roots, and symlinks. This keeps a
    mistaken ``--output .. --overwrite`` from deleting the repository parent.
    """
    output = _safe_output_path(path)
    if output.exists() and any(output.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"output directory is not empty: {output}. Use a fresh path or --overwrite."
            )
        shutil.rmtree(output)
    return ensure_dir(output)


def prepare_resume_output_dir(path: str | Path) -> Path:
    """Prepare a safe output root for resumable workflows without deleting contents.

    Multi-seed evaluation intentionally reuses compatible completed seed directories.
    It must therefore allow an existing non-empty directory, but it still applies the
    same project-root, ancestor, root, symlink, and non-directory protections as the
    one-shot path.
    """
    return ensure_dir(_safe_output_path(path))


def atomic_write_text(path: str | Path, content: str) -> None:
    destination = Path(path)
    ensure_dir(destination.parent)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=destination.parent, delete=False
    ) as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    os.replace(temporary, destination)


def write_json(path: str | Path, payload: Any) -> None:
    atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")


def now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _iter_source_paths(root: Path, include: Iterable[str]) -> list[Path]:
    paths: list[Path] = []
    for item in include:
        target = root / item
        if target.is_file():
            paths.append(target)
        elif target.is_dir():
            paths.extend(
                path
                for path in target.rglob("*")
                if path.is_file()
                and "__pycache__" not in path.parts
                and not any(part.endswith(".egg-info") for part in path.parts)
            )
    return sorted(paths, key=lambda path: path.relative_to(root).as_posix())


def source_fingerprint(root: str | Path | None = None) -> str:
    """Fingerprint the executable contract in editable and wheel installations.

    Editable/source runs include repository contracts such as CI and Docker files.
    Installed wheels do not have a repository root, so the function fingerprints the
    installed package modules and packaged default configuration instead of silently
    returning the empty SHA-256 digest.
    """
    digest = hashlib.sha256()
    package_root = Path(__file__).resolve().parent
    project_root: Path | None
    if root is not None:
        project_root = Path(root)
    else:
        candidate_root = package_root.parents[1]
        project_root = candidate_root if (candidate_root / "pyproject.toml").exists() else None

    if project_root is not None:
        source_tree = "src/rtrace" if (project_root / "src" / "rtrace").is_dir() else "src"
        include = (
            source_tree,
            "configs",
            "scripts",
            ".github",
            "pyproject.toml",
            "Makefile",
            "Dockerfile",
        )
        paths = _iter_source_paths(project_root, include)
        relative_root = project_root
    else:
        paths = sorted(
            [path for path in package_root.rglob("*.py") if path.is_file()]
            + [package_root / "default.yaml", package_root / "py.typed"],
            key=lambda path: path.relative_to(package_root).as_posix(),
        )
        relative_root = package_root

    for path in paths:
        if not path.exists():
            continue
        relative = path.relative_to(relative_root).as_posix().encode("utf-8")
        digest.update(relative)
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def hardware_metadata() -> dict[str, Any]:
    result: dict[str, Any] = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu_count": os.cpu_count(),
    }
    package_versions: dict[str, str | None] = {}
    for package_name in (
        "numpy",
        "pandas",
        "scikit-learn",
        "lightgbm",
        "matplotlib",
        "pydantic",
        "PyYAML",
    ):
        try:
            package_versions[package_name] = distribution_version(package_name)
        except PackageNotFoundError:
            package_versions[package_name] = None
    result["package_versions"] = package_versions
    try:
        import torch

        result["torch"] = torch.__version__
        result["cuda_available"] = torch.cuda.is_available()
        result["cuda_device"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception:
        result["torch"] = None
        result["cuda_available"] = False
        result["cuda_device"] = None
    try:
        result["git_revision"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        result["git_revision"] = "unavailable"
    return result
