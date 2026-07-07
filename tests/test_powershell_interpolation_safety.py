from __future__ import annotations

import re
from pathlib import Path

SCOPED_VARIABLES = {"env", "global", "local", "private", "script", "using", "workflow"}
UNSAFE_VARIABLE_COLON = re.compile(r"(?<![${])\$([A-Za-z_][A-Za-z0-9_]*):")


def _double_quoted_strings(source: str) -> list[str]:
    """Extract ordinary double-quoted PowerShell strings for interpolation checks.

    The project entrypoint scripts intentionally do not use PowerShell here-strings.
    A backtick escapes the next character inside a double-quoted string.
    """
    strings: list[str] = []
    current: list[str] = []
    inside = False
    escaped = False
    for char in source:
        if inside:
            if escaped:
                current.append(char)
                escaped = False
            elif char == "`":
                current.append(char)
                escaped = True
            elif char == '"':
                strings.append("".join(current))
                current = []
                inside = False
            else:
                current.append(char)
        elif char == '"':
            inside = True
    if inside:
        raise AssertionError("unterminated double-quoted PowerShell string")
    return strings


def _unsafe_variable_colon_references(source: str) -> list[str]:
    unsafe: list[str] = []
    for text in _double_quoted_strings(source):
        for match in UNSAFE_VARIABLE_COLON.finditer(text):
            if match.group(1).lower() not in SCOPED_VARIABLES:
                unsafe.append(match.group(0))
    return unsafe


def test_powershell_entrypoints_do_not_use_unbraced_variable_colon_interpolation() -> None:
    scripts = sorted(Path("scripts").glob("*.ps1"))
    assert scripts, "expected PowerShell entrypoints"
    violations = {
        script.name: _unsafe_variable_colon_references(script.read_text(encoding="utf-8"))
        for script in scripts
    }
    assert violations == {script.name: [] for script in scripts}


def test_regression_examples_are_safe_only_when_brace_delimited() -> None:
    assert _unsafe_variable_colon_references('throw "bad $PathValue: details"') == ["$PathValue:"]
    assert _unsafe_variable_colon_references('throw "good ${PathValue}: details"') == []
    assert _unsafe_variable_colon_references('"$env:PYTHONUTF8"') == []
