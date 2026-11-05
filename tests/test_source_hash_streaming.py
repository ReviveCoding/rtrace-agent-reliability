import hashlib
from pathlib import Path

import pytest

from rtrace.utils import source_fingerprint


def test_source_fingerprint_keeps_boundaries_without_unbounded_reads(
    tmp_path: Path, monkeypatch
) -> None:
    source = tmp_path / "src" / "rtrace"
    source.mkdir(parents=True)
    content = b"source" * 400_000
    (source / "model.py").write_bytes(content)
    (tmp_path / "pyproject.toml").write_text("project")
    digest = hashlib.sha256()
    for name, value in [(b"pyproject.toml", b"project"), (b"src/rtrace/model.py", content)]:
        digest.update(name + b"\0" + value + b"\0")
    monkeypatch.setattr(Path, "read_bytes", lambda self: pytest.fail("unbounded read"))
    assert source_fingerprint(tmp_path) == digest.hexdigest()
