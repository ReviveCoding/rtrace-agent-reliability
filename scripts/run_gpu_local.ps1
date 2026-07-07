[CmdletBinding()]
param(
    [ValidateSet("Verify", "Full")]
    [string]$Mode = "Verify",
    [string]$RepoRoot = "",
    [string]$RunRootBase = "C:\Users\bjw-0\ML_Outputs\Apple02_rtrace_agentic_ai_reliability",
    [string]$ModelId = "Qwen/Qwen3-4B",
    [string]$ModelRevision = "",
    [int]$Seed = 17,
    [int]$QloraMaxSteps = 120,
    [int]$GradientAccumulation = 8,
    [int]$MaxLength = 1024,
    [switch]$IncludeOptionalMcp,
    [ValidateSet("cu126", "cu128")]
    [string]$TorchCudaWheel = "cu128"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"
$env:PYTHONUTF8 = "1"
$env:PYTHONHASHSEED = "0"
$env:MPLBACKEND = "Agg"
$env:TOKENIZERS_PARALLELISM = "false"
$env:HF_HUB_DISABLE_TELEMETRY = "1"
$env:WANDB_DISABLED = "true"
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD = "1"
$env:OMP_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
$env:OPENBLAS_NUM_THREADS = "1"
$env:NUMEXPR_NUM_THREADS = "1"

function Write-Stage {
    param([Parameter(Mandatory = $true)][string]$Title)
    Write-Host ""
    Write-Host "================================================================" -ForegroundColor Cyan
    Write-Host $Title -ForegroundColor Cyan
    Write-Host "================================================================" -ForegroundColor Cyan
}

function Invoke-Checked {
    param(
        [Parameter(Mandatory = $true)][string]$Label,
        [Parameter(Mandatory = $true)][string]$FilePath,
        [Parameter(ValueFromRemainingArguments = $true)][string[]]$CommandArguments
    )
    Write-Host "COMMAND [$Label]: $FilePath $($CommandArguments -join ' ')" -ForegroundColor DarkGray
    & $FilePath @CommandArguments
    if ($LASTEXITCODE -ne 0) {
        throw "FAILED [$Label] with exit code $LASTEXITCODE."
    }
}

function Require-Path {
    param(
        [Parameter(Mandatory = $true)][string]$PathValue,
        [Parameter(Mandatory = $true)][string]$Label
    )
    if (-not (Test-Path -LiteralPath $PathValue)) {
        throw "$Label not found: $PathValue"
    }
}

function Test-PowerShellParse {
    param([Parameter(Mandatory = $true)][string]$PathValue)
    $tokens = $null
    $errors = $null
    [void][System.Management.Automation.Language.Parser]::ParseFile(
        $PathValue, [ref]$tokens, [ref]$errors
    )
    if ($errors.Count -gt 0) {
        $details = ($errors | ForEach-Object { $_.Message }) -join "; "
        throw "PowerShell parse failure in ${PathValue}: $details"
    }
}

function Resolve-Python311 {
    $launcher = Get-Command py -ErrorAction SilentlyContinue
    if ($launcher) {
        & py -3.11 -c "import sys; assert sys.version_info[:2] == (3, 11); print(sys.executable)" | Out-Null
        if ($LASTEXITCODE -eq 0) {
            return @{ FilePath = "py"; Prefix = @("-3.11") }
        }
    }

    $python = Get-Command python -ErrorAction SilentlyContinue
    if ($python) {
        & python -c "import sys; assert sys.version_info[:2] == (3, 11); print(sys.executable)" | Out-Null
        if ($LASTEXITCODE -eq 0) {
            return @{ FilePath = "python"; Prefix = @() }
        }
    }

    throw "Python 3.11 x64 was not found. Install it and ensure either 'py -3.11' or 'python' resolves to Python 3.11."
}

function Write-JsonFile {
    param(
        [Parameter(Mandatory = $true)][object]$Value,
        [Parameter(Mandatory = $true)][string]$PathValue
    )
    $Value | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $PathValue -Encoding utf8
}

function Get-TorchCudaProbe {
    param(
        [Parameter(Mandatory = $true)][string]$PythonExe,
        [Parameter(Mandatory = $true)][string]$ExpectedCudaPrefix
    )
    $Expected = if ($ExpectedCudaPrefix -eq "cu128") { "12.8" } else { "12.6" }
    $ProbePath = Join-Path $RepoRoot "scripts\probe_torch_cuda.py"
    Require-Path $ProbePath "PyTorch CUDA probe script"

    # A first run normally has no torch package. The helper always returns a JSON
    # state with exit code 0 for missing/import-failed/CPU-only torch, so this probe
    # cannot terminate Verify before the CUDA-wheel installation fallback.
    $PreviousErrorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        $ProbeOutput = & $PythonExe $ProbePath "--expected-cuda" $Expected 2>&1
        $ProbeExit = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $PreviousErrorActionPreference
    }
    if ($ProbeExit -ne 0) {
        $Rendered = ($ProbeOutput | Out-String).Trim()
        throw "PyTorch CUDA probe failed with exit code ${ProbeExit}: $Rendered"
    }
    $JsonLine = $ProbeOutput | ForEach-Object { [string]$_ } | Where-Object { $_.Trim().StartsWith("{") } | Select-Object -Last 1
    if ($null -eq $JsonLine) {
        $Rendered = ($ProbeOutput | Out-String).Trim()
        throw "PyTorch CUDA probe did not return JSON: $Rendered"
    }
    try {
        return ($JsonLine | ConvertFrom-Json)
    }
    catch {
        throw "PyTorch CUDA probe returned invalid JSON: $JsonLine"
    }
}

function Get-BitsAndBytesProbe {
    param(
        [Parameter(Mandatory = $true)][string]$PythonExe,
        [Parameter(Mandatory = $true)][string]$ExpectedCudaPrefix
    )
    $Expected = if ($ExpectedCudaPrefix -eq "cu128") { "12.8" } else { "12.6" }
    $ProbePath = Join-Path $RepoRoot "scripts\probe_bitsandbytes.py"
    Require-Path $ProbePath "bitsandbytes 4-bit probe script"

    # The helper captures child stderr internally. Keep Continue only around this
    # native invocation as a final Windows PowerShell 5.1 safeguard.
    $PreviousErrorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        $ProbeOutput = & $PythonExe $ProbePath "--expected-cuda" $Expected 2>&1
        $ProbeExit = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $PreviousErrorActionPreference
    }
    if ($ProbeExit -ne 0) {
        $Rendered = ($ProbeOutput | Out-String).Trim()
        throw "bitsandbytes 4-bit probe failed with exit code ${ProbeExit}: $Rendered"
    }
    $JsonLine = $ProbeOutput | ForEach-Object { [string]$_ } | Where-Object { $_.Trim().StartsWith("{") } | Select-Object -Last 1
    if ($null -eq $JsonLine) {
        $Rendered = ($ProbeOutput | Out-String).Trim()
        throw "bitsandbytes 4-bit probe did not return JSON: $Rendered"
    }
    try {
        return ($JsonLine | ConvertFrom-Json)
    }
    catch {
        throw "bitsandbytes 4-bit probe returned invalid JSON: $JsonLine"
    }
}

if (-not $RepoRoot) {
    $RepoRoot = Split-Path -Parent $PSScriptRoot
}
$RepoRoot = (Resolve-Path -LiteralPath $RepoRoot).Path
$Timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
$RunRoot = Join-Path $RunRootBase "gpu-$($Mode.ToLowerInvariant())-$Timestamp"
$LogRoot = Join-Path $RunRoot "logs"
$HfRoot = "C:\hf-rtrace"
$Venv = Join-Path $RepoRoot ".venv"
$VenvPython = Join-Path $Venv "Scripts\python.exe"

try {
    Write-Stage "[0/10] Source, PowerShell, output-path, and disk preflight"
    Require-Path (Join-Path $RepoRoot "pyproject.toml") "pyproject.toml"
    Require-Path (Join-Path $RepoRoot "scripts\train_qlora.py") "QLoRA training script"
    Require-Path (Join-Path $RepoRoot "scripts\raw_adapter_inference.py") "adapter inference script"
    Require-Path (Join-Path $RepoRoot "scripts\probe_torch_cuda.py") "PyTorch CUDA probe script"
    Require-Path (Join-Path $RepoRoot "scripts\probe_bitsandbytes.py") "bitsandbytes 4-bit probe script"
    Require-Path (Join-Path $RepoRoot "configs\ci_smoke.yaml") "CI smoke configuration"
    Test-PowerShellParse $PSCommandPath
    Test-PowerShellParse (Join-Path $RepoRoot "scripts\run_local.ps1")

    New-Item -ItemType Directory -Force -Path $RunRoot, $LogRoot, $HfRoot | Out-Null
    $env:HF_HOME = $HfRoot
    $env:HF_HUB_CACHE = Join-Path $HfRoot "hub"
    $env:HF_DATASETS_CACHE = Join-Path $HfRoot "datasets"
    New-Item -ItemType Directory -Force -Path $env:HF_HUB_CACHE, $env:HF_DATASETS_CACHE | Out-Null

    $driveName = ([IO.Path]::GetPathRoot($HfRoot)).TrimEnd("\").TrimEnd(":")
    $freeGiB = [math]::Round((Get-PSDrive -Name $driveName).Free / 1GB, 1)
    if ($freeGiB -lt 30) {
        throw "At least 30 GiB free space is required on $driveName`: for model cache and outputs. Found $freeGiB GiB."
    }

    Set-Location -LiteralPath $RepoRoot

    Write-Stage "[1/10] Resolve Python 3.11 and create the repository virtual environment"
    $BootstrapPython = Resolve-Python311
    $CheckArgs = @() + $BootstrapPython.Prefix + @("-c", "import sys; print(sys.executable); print(sys.version)")
    Invoke-Checked "python-3.11-check" $BootstrapPython.FilePath @CheckArgs
    if (-not (Test-Path -LiteralPath $VenvPython)) {
        $CreateArgs = @() + $BootstrapPython.Prefix + @("-m", "venv", $Venv)
        Invoke-Checked "create-venv" $BootstrapPython.FilePath @CreateArgs
    }
    Invoke-Checked "venv-version" $VenvPython "-c" "import sys; assert sys.version_info[:2] == (3, 11); print(sys.version)"

    Write-Stage "[2/10] Install base, test, lint, and package dependencies"
    # torch 2.11.0+cu128 currently declares setuptools<82. Never let the generic
    # bootstrap upgrade pull an incompatible setuptools release into an already
    # CUDA-ready environment, especially when Full is run after Verify.
    Invoke-Checked "upgrade-pip-build-tools" $VenvPython "-m" "pip" "install" "--upgrade" "pip" "setuptools>=68,<82" "wheel"
    Invoke-Checked "verify-setuptools-compatible" $VenvPython "-c" "import importlib.metadata as m; v=m.version('setuptools'); major=int(v.split('.', 1)[0]); assert 68 <= major < 82, v; print(f'setuptools={v}')"
    Invoke-Checked "install-project-dev" $VenvPython "-m" "pip" "install" "-e" ".[dev]"
    Invoke-Checked "pip-check-base" $VenvPython "-m" "pip" "check"

    Write-Stage "[3/10] Validate source, CI contracts, and the native smoke pipeline"
    Invoke-Checked "pytest" $VenvPython "-m" "pytest" "-q"
    Invoke-Checked "ruff-check" $VenvPython "-m" "ruff" "check" "src" "tests" "scripts"
    Invoke-Checked "ruff-format" $VenvPython "-m" "ruff" "format" "--check" "src" "tests" "scripts"
    Invoke-Checked "mypy" $VenvPython "-m" "mypy" "src/rtrace"
    Invoke-Checked "compileall" $VenvPython "-m" "compileall" "-q" "src" "scripts"

    $SmokeRoot = Join-Path $RunRoot "native_smoke"
    Invoke-Checked "smoke-validate" $VenvPython "-m" "rtrace.cli" "validate-data" "--output" (Join-Path $SmokeRoot "validate") "--seed" $Seed "--config" "configs/ci_smoke.yaml" "--overwrite"
    Invoke-Checked "smoke-run" $VenvPython "-m" "rtrace.cli" "run-all" "--output" (Join-Path $SmokeRoot "run") "--seed" $Seed "--config" "configs/ci_smoke.yaml" "--overwrite"
    Invoke-Checked "smoke-verify" $VenvPython "scripts/verify_output.py" "--output" (Join-Path $SmokeRoot "run")
    Invoke-Checked "smoke-incidents" $VenvPython "-m" "rtrace.cli" "replay-incidents" "--output" (Join-Path $SmokeRoot "incidents") "--seed" $Seed "--overwrite"

    $McpStatus = "SKIPPED"
    if ($IncludeOptionalMcp) {
        Write-Stage "[4/10] Validate the optional FastMCP adapter"
        Invoke-Checked "install-mcp" $VenvPython "-m" "pip" "install" "-e" ".[mcp]"
        Invoke-Checked "pip-check-mcp" $VenvPython "-m" "pip" "check"
        Invoke-Checked "mcp-smoke" $VenvPython "-m" "pytest" "tests/test_mcp_runtime_optional.py" "-q"
        $McpStatus = "PASS"
    }
    else {
        Write-Host "Optional FastMCP smoke skipped. Run with -IncludeOptionalMcp to validate it locally." -ForegroundColor Yellow
    }

    Write-Stage "[5/10] Install the Windows QLoRA GPU stack ($TorchCudaWheel)"
    $TorchProbeBefore = Get-TorchCudaProbe $VenvPython $TorchCudaWheel
    Write-JsonFile $TorchProbeBefore (Join-Path $LogRoot "torch_probe_before_install.json")
    if (-not [bool]$TorchProbeBefore.ready) {
        Write-Host "PyTorch CUDA probe state: $($TorchProbeBefore.status). Installing requested $TorchCudaWheel wheel." -ForegroundColor Yellow
        Invoke-Checked "remove-mixed-torch" $VenvPython "-m" "pip" "uninstall" "-y" "torch" "torchvision" "torchaudio"
        Invoke-Checked "install-torch-$TorchCudaWheel" $VenvPython "-m" "pip" "install" "--upgrade" "--force-reinstall" "torch" "--index-url" "https://download.pytorch.org/whl/$TorchCudaWheel"
    }
    Invoke-Checked "install-qlora-stack" $VenvPython "-m" "pip" "install" "-e" ".[qlora]"
    Invoke-Checked "pip-check-gpu" $VenvPython "-m" "pip" "check"
    $TorchProbeAfter = Get-TorchCudaProbe $VenvPython $TorchCudaWheel
    Write-JsonFile $TorchProbeAfter (Join-Path $LogRoot "torch_probe_after_install.json")
    if (-not [bool]$TorchProbeAfter.ready) {
        throw "Requested PyTorch CUDA wheel did not produce a ready CUDA runtime. Inspect $(Join-Path $LogRoot 'torch_probe_after_install.json')."
    }
    (& $VenvPython -m pip freeze) | Set-Content -LiteralPath (Join-Path $RunRoot "requirements.freeze.txt") -Encoding utf8

    Write-Stage "[6/10] Verify NVIDIA, PyTorch CUDA, BF16, and bitsandbytes"
    if (-not (Get-Command nvidia-smi -ErrorAction SilentlyContinue)) {
        throw "nvidia-smi was not found. Install or update the NVIDIA driver and rerun."
    }
    $NvidiaOutput = & nvidia-smi 2>&1
    $NvidiaExit = $LASTEXITCODE
    $NvidiaOutput | Set-Content -LiteralPath (Join-Path $LogRoot "nvidia_smi.txt") -Encoding utf8
    if ($NvidiaExit -ne 0) {
        throw "nvidia-smi failed with exit code $NvidiaExit. Inspect $(Join-Path $LogRoot 'nvidia_smi.txt')."
    }
    $ExpectedCudaVersion = if ($TorchCudaWheel -eq "cu128") { "12.8" } else { "12.6" }
    & $VenvPython -c "import json, torch; assert torch.cuda.is_available(); assert str(torch.version.cuda).startswith('$ExpectedCudaVersion'), torch.version.cuda; p=torch.cuda.get_device_properties(0); assert p.total_memory >= 12*1024**3; assert torch.cuda.is_bf16_supported(); print(json.dumps({'torch':torch.__version__,'cuda':torch.version.cuda,'device':torch.cuda.get_device_name(0),'vram_gib':round(p.total_memory/1024**3,2),'bf16':torch.cuda.is_bf16_supported()}, indent=2))"
    if ($LASTEXITCODE -ne 0) {
        throw "PyTorch CUDA/BF16 preflight failed."
    }
    $BnbProbe = Get-BitsAndBytesProbe $VenvPython $TorchCudaWheel
    Write-JsonFile $BnbProbe (Join-Path $LogRoot "bitsandbytes_probe.json")
    [string]$BnbStderr = $BnbProbe.child_stderr
    $BnbStderr | Set-Content -LiteralPath (Join-Path $LogRoot "bitsandbytes_diagnostic.txt") -Encoding utf8
    if (-not [bool]$BnbProbe.ready) {
        throw "bitsandbytes 4-bit operational probe failed with status $($BnbProbe.status). Inspect $(Join-Path $LogRoot 'bitsandbytes_probe.json') and $(Join-Path $LogRoot 'bitsandbytes_diagnostic.txt')."
    }
    if ($BnbStderr) {
        Write-Host "bitsandbytes emitted a non-fatal diagnostic; it was recorded in bitsandbytes_diagnostic.txt." -ForegroundColor Yellow
    }

    if ($Mode -eq "Verify") {
        Write-JsonFile @{
            status = "VERIFY_PASS"
            repo_root = $RepoRoot
            run_root = $RunRoot
            qlora_model = $ModelId
            torch_cuda_wheel = $TorchCudaWheel
            optional_mcp = $McpStatus
            note = "No foundation-model weights were downloaded and no GPU training was started."
        } (Join-Path $RunRoot "verification_manifest.json")
        Write-Stage "Verification complete"
        Write-Host "No source, package, small-data pipeline, or GPU-runtime preflight blocker was found." -ForegroundColor Green
        Write-Host "Next command: .\scripts\run_gpu_local.ps1 -Mode Full" -ForegroundColor Green
        exit 0
    }

    Write-Stage "[7/10] Full native benchmark and five-seed reliability evidence"
    $NativeRoot = Join-Path $RunRoot "native"
    Invoke-Checked "full-run" $VenvPython "-m" "rtrace.cli" "run-all" "--output" (Join-Path $NativeRoot "seed_$Seed") "--seed" $Seed "--overwrite"
    Invoke-Checked "full-verify" $VenvPython "scripts/verify_output.py" "--output" (Join-Path $NativeRoot "seed_$Seed")
    Invoke-Checked "multiseed" $VenvPython "-m" "rtrace.cli" "run-multiseed" "--output" (Join-Path $NativeRoot "multiseed") "--seeds" "11,17,23,29,31" "--overwrite"

    Write-Stage "[8/10] Qwen3-4B QLoRA training and raw adapter inference"
    $QloraRoot = Join-Path $RunRoot "qlora"
    $TrainArgs = @(
        "scripts/train_qlora.py",
        "--model", $ModelId,
        "--output", $QloraRoot,
        "--max-steps", $QloraMaxSteps,
        "--batch-size", "1",
        "--gradient-accumulation", $GradientAccumulation,
        "--max-length", $MaxLength,
        "--seed", $Seed
    )
    if ($ModelRevision) {
        $TrainArgs += @("--revision", $ModelRevision)
    }
    Invoke-Checked "qlora-train" $VenvPython @TrainArgs
    Require-Path (Join-Path $QloraRoot "adapter\adapter_config.json") "QLoRA adapter config"
    Require-Path (Join-Path $QloraRoot "training_summary.json") "QLoRA summary"

    $InferenceArgs = @(
        "scripts/raw_adapter_inference.py",
        "--model", $ModelId,
        "--adapter", (Join-Path $QloraRoot "adapter"),
        "--output", (Join-Path $QloraRoot "raw_adapter_inference.json")
    )
    if ($ModelRevision) {
        $InferenceArgs += @("--revision", $ModelRevision)
    }
    Invoke-Checked "adapter-inference" $VenvPython @InferenceArgs
    Require-Path (Join-Path $QloraRoot "raw_adapter_inference.json") "raw adapter inference output"
    Invoke-Checked "adapter-inference-contract" $VenvPython "-c" "import json, pathlib; p=pathlib.Path(r'$QloraRoot')/'raw_adapter_inference.json'; d=json.loads(p.read_text(encoding='utf-8')); assert d['status']=='RAW_ADAPTER_INFERENCE_COMPLETED'; assert d['completion_nonempty'], d"

    Write-Stage "[9/10] Clean-wheel package validation"
    $WheelRoot = Join-Path $RunRoot "clean_wheel"
    $Dist = Join-Path $WheelRoot "dist"
    New-Item -ItemType Directory -Force -Path $Dist | Out-Null
    Invoke-Checked "build-wheel" $VenvPython "-m" "build" "--wheel" "--outdir" $Dist
    $Wheel = Get-ChildItem -LiteralPath $Dist -Filter "*.whl" | Select-Object -First 1
    if ($null -eq $Wheel) {
        throw "No wheel was produced."
    }
    $WheelVenv = Join-Path $WheelRoot "venv"
    $WheelCreateArgs = @() + $BootstrapPython.Prefix + @("-m", "venv", $WheelVenv)
    Invoke-Checked "wheel-venv" $BootstrapPython.FilePath @WheelCreateArgs
    $WheelPython = Join-Path $WheelVenv "Scripts\python.exe"
    Invoke-Checked "wheel-upgrade-pip" $WheelPython "-m" "pip" "install" "--upgrade" "pip"
    Invoke-Checked "wheel-install" $WheelPython "-m" "pip" "install" $Wheel.FullName
    Invoke-Checked "wheel-pip-check" $WheelPython "-m" "pip" "check"
    Invoke-Checked "wheel-run" $WheelPython "-m" "rtrace.cli" "run-all" "--output" (Join-Path $WheelRoot "run") "--seed" $Seed "--config" (Join-Path $RepoRoot "configs\ci_smoke.yaml") "--overwrite"
    Invoke-Checked "wheel-verify" $VenvPython "scripts/verify_output.py" "--output" (Join-Path $WheelRoot "run")

    Write-Stage "[10/10] Completed"
    Write-JsonFile @{
        status = "FULL_COMPLETED"
        run_root = $RunRoot
        model = $ModelId
        revision = $ModelRevision
        seed = $Seed
        torch_cuda_wheel = $TorchCudaWheel
        optional_mcp = $McpStatus
        claim_boundary = "QLoRA uses repository-generated SafeAssist data. It is not API-Bank training and is not wired into the deterministic C0-C5 release evaluator."
    } (Join-Path $RunRoot "execution_manifest.json")
    Write-Host "Run root: $RunRoot" -ForegroundColor Green
}
catch {
    if (Test-Path -LiteralPath $RunRoot) {
        Write-JsonFile @{
            status = "FAILED"
            timestamp_utc = (Get-Date).ToUniversalTime().ToString("o")
            mode = $Mode
            run_root = $RunRoot
            error = $_.Exception.Message
        } (Join-Path $RunRoot "failure_manifest.json")
    }
    Write-Host "R-TRACE execution stopped: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host "Retained diagnostics: $RunRoot" -ForegroundColor Yellow
    exit 1
}
