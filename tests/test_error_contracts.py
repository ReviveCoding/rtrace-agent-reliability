from __future__ import annotations

from pathlib import Path

import pytest

from rtrace.cli import main
from rtrace.config import load_config


def test_non_mapping_config_section_fails_as_value_error(tmp_path: Path) -> None:
    path = tmp_path / "invalid.yaml"
    path.write_text("benchmark: []\n", encoding="utf-8")
    with pytest.raises(ValueError, match="configuration section must be a mapping: benchmark"):
        load_config(path)


def test_cli_invalid_config_returns_clean_exit_code(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "invalid.yaml"
    path.write_text("routing: null\n", encoding="utf-8")
    with pytest.raises(SystemExit) as raised:
        main(["validate-data", "--output", str(tmp_path / "out"), "--config", str(path)])
    captured = capsys.readouterr()
    assert raised.value.code == 2
    assert "rtrace: error:" in captured.err


def test_cli_invalid_seed_list_returns_clean_exit_code(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as raised:
        main(
            [
                "run-multiseed",
                "--output",
                str(tmp_path / "out"),
                "--config",
                "configs/ci_smoke.yaml",
                "--seeds",
                "17,nope",
            ]
        )
    captured = capsys.readouterr()
    assert raised.value.code == 2
    assert "rtrace: error:" in captured.err
