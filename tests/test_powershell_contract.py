from pathlib import Path


def test_powershell_runner_accepts_and_forwards_optional_config() -> None:
    script = Path("scripts/run_local.ps1").read_text(encoding="utf-8")
    assert "[string]$ConfigPath" in script
    assert '"--config", $ConfigPath' in script
    assert '"run-all" "--output" (Join-Path $OutputRoot "run") "--seed" $Seed' in script
    assert (
        '"run-multiseed" "--output" (Join-Path $OutputRoot "multiseed") "--seeds" "11,17,23,29,31"'
        in script
    )
    assert "@ConfigArg @OverwriteArg" in script


def test_powershell_runner_checks_native_command_exit_codes() -> None:
    script = Path("scripts/run_local.ps1").read_text(encoding="utf-8")
    assert "function Invoke-Checked" in script
    assert "if ($LASTEXITCODE -ne 0)" in script
    assert "scripts/verify_output.py" in script


def test_local_powershell_runner_falls_back_when_py_launcher_is_stale() -> None:
    script = Path("scripts/run_local.ps1").read_text(encoding="utf-8")
    assert "function Resolve-BootstrapPython" in script
    assert "assert (3, 11) <= sys.version_info[:2] < (3, 14)" in script
    assert "PYTEST_DISABLE_PLUGIN_AUTOLOAD" in script


def test_local_powershell_runner_explicitly_loads_pytest_cov_when_plugin_autoload_is_disabled() -> (
    None
):
    script = Path("scripts/run_local.ps1").read_text(encoding="utf-8")
    assert "PYTEST_DISABLE_PLUGIN_AUTOLOAD" in script
    assert '"-p" "pytest_cov" "--cov=rtrace"' in script


def test_windows_parser_error_regression_is_brace_delimited() -> None:
    gpu = Path("scripts/run_gpu_local.ps1").read_text(encoding="utf-8")
    local = Path("scripts/run_local.ps1").read_text(encoding="utf-8")
    assert '"PowerShell parse failure in ${PathValue}: $details"' in gpu
    assert '"Command failed with exit code ${LASTEXITCODE}: $FilePath' in local
    assert '"PowerShell parse failure in $PathValue: $details"' not in gpu
    assert '"Command failed with exit code $LASTEXITCODE: $FilePath' not in local
