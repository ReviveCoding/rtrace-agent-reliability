[CmdletBinding()]
param(
    [Alias("Command")]
    [ValidateSet("validate", "test", "quality", "run", "multiseed")]
    [string]$Mode = "run",
    [string]$OutputRoot = "artifacts/local",
    [int]$Seed = 17,
    [string]$ConfigPath = "",
    [switch]$Overwrite
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$env:PYTHONUTF8 = "1"
$env:PYTHONHASHSEED = "0"
$env:MPLBACKEND = "Agg"
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD = "1"
$env:OMP_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
$env:OPENBLAS_NUM_THREADS = "1"
$env:NUMEXPR_NUM_THREADS = "1"

function Invoke-Checked {
    param(
        [Parameter(Mandatory = $true, Position = 0)]
        [string]$FilePath,
        [Parameter(ValueFromRemainingArguments = $true)]
        [string[]]$Arguments
    )

    & $FilePath @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code ${LASTEXITCODE}: $FilePath $($Arguments -join ' ')"
    }
}

$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $RepoRoot

function Resolve-BootstrapPython {
    $PyLauncher = Get-Command py -ErrorAction SilentlyContinue
    if ($PyLauncher) {
        & py -3.11 -c "import sys; assert sys.version_info[:2] == (3, 11); print(sys.executable)" | Out-Null
        if ($LASTEXITCODE -eq 0) {
            return @{ FilePath = "py"; Prefix = @("-3.11") }
        }
    }
    $Python = Get-Command python -ErrorAction SilentlyContinue
    if ($Python) {
        & python -c "import sys; assert (3, 11) <= sys.version_info[:2] < (3, 14); print(sys.executable)" | Out-Null
        if ($LASTEXITCODE -eq 0) {
            return @{ FilePath = "python"; Prefix = @() }
        }
    }
    throw "No compatible Python was found. Install Python 3.11-3.13 or ensure it is on PATH."
}

$Venv = Join-Path $RepoRoot ".venv"
$VenvPython = Join-Path $Venv "Scripts\python.exe"

if (-not (Test-Path -LiteralPath $VenvPython)) {
    $BootstrapPython = Resolve-BootstrapPython
    $CreateArguments = @() + $BootstrapPython.Prefix + @("-m", "venv", $Venv)
    Invoke-Checked $BootstrapPython.FilePath @CreateArguments
}

$PythonVersion = & $VenvPython -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
if ($LASTEXITCODE -ne 0) {
    throw "Unable to resolve the virtual-environment Python version."
}
$VersionParts = $PythonVersion.Split('.')
if (($VersionParts.Length -ne 2) -or ([int]$VersionParts[0] -ne 3) -or ([int]$VersionParts[1] -lt 11) -or ([int]$VersionParts[1] -ge 14)) {
    throw "R-TRACE requires Python >=3.11,<3.14; resolved interpreter is $PythonVersion. Install/select Python 3.11-3.13, remove .venv, and rerun."
}

Invoke-Checked $VenvPython "-m" "pip" "install" "--upgrade" "pip"
Invoke-Checked $VenvPython "-m" "pip" "install" "-e" ".[dev]"
Invoke-Checked $VenvPython "-m" "pip" "check"

$OverwriteArg = @()
if ($Overwrite) { $OverwriteArg = @("--overwrite") }
$ConfigArg = @()
if ($ConfigPath) {
    if (-not (Test-Path -LiteralPath $ConfigPath)) { throw "ConfigPath does not exist: $ConfigPath" }
    $ConfigArg = @("--config", $ConfigPath)
}

switch ($Mode) {
    "validate" {
        Invoke-Checked $VenvPython "-m" "rtrace.cli" "validate-data" "--output" (Join-Path $OutputRoot "validate") "--seed" $Seed @ConfigArg @OverwriteArg
    }
    "test" {
        # PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 keeps third-party machine plugins from
        # changing the test environment. Explicitly load pytest-cov for coverage.
        Invoke-Checked $VenvPython "-m" "pytest" "-p" "pytest_cov" "--cov=rtrace" "--cov-report=term-missing" "--cov-report=xml"
    }
    "quality" {
        Invoke-Checked $VenvPython "-m" "ruff" "check" "src" "tests" "scripts"
        Invoke-Checked $VenvPython "-m" "ruff" "format" "--check" "src" "tests" "scripts"
        Invoke-Checked $VenvPython "-m" "mypy" "src/rtrace"
        Invoke-Checked $VenvPython "-m" "build"
    }
    "run" {
        Invoke-Checked $VenvPython "-m" "rtrace.cli" "run-all" "--output" (Join-Path $OutputRoot "run") "--seed" $Seed @ConfigArg @OverwriteArg
        Invoke-Checked $VenvPython "scripts/verify_output.py" "--output" (Join-Path $OutputRoot "run")
    }
    "multiseed" {
        Invoke-Checked $VenvPython "-m" "rtrace.cli" "run-multiseed" "--output" (Join-Path $OutputRoot "multiseed") "--seeds" "11,17,23,29,31" @ConfigArg @OverwriteArg
    }
}
