from __future__ import annotations

import re
from pathlib import Path

import rtrace


def test_import_version_matches_project_metadata() -> None:
    project = Path("pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r'^version = "([^"]+)"$', project, flags=re.MULTILINE)
    assert match is not None
    assert rtrace.__version__ == match.group(1)
