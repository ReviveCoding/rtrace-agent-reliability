# R-TRACE Agentic AI Evaluation & Release Reliability Framework

**Version 0.6.11 | Local-first, stateful tool-use evaluation with pre-execution GPU safeguards**

R-TRACE evaluates tool-using agent proposals before they mutate persistent state. It combines a deterministic synthetic SafeAssist-MCP benchmark, SQLite-backed execution, host-mediated consent, calibrated risk routing, a PHT-derived reference-conditioned critic, and an ABC-inspired structured action-scope correction layer.

The executable pre-action outcomes are **ALLOW**, **CLARIFY**, **CONFIRM**, and **BLOCK**. **COMPENSATE** is recorded only after a partial mutation.

## What the repository contains

- Calendar, contacts, and payment-intent stateful local workflows.
- Schema validation, preflight checks, authorization, idempotency, transaction logging, atomic payment transitions, partial-failure compensation, and fail-closed execution.
- Host-issued, action-bound, expiring confirmation tokens. A model cannot self-attest user consent.
- C0-C5 comparison path: deterministic baseline, prompted actor, generic SFT behavior simulator, calibrated critic, R-TRACE reference-conditioned critic, and C5 with structured action-scope correction.
- Train/calibration/development/final split separation for reference selection, calibration, and routing selection.
- CLI, PowerShell wrapper, Docker image, tests, CI workflow, reports, figures, and optional FastMCP in-process smoke path.

## Claim boundary

This repository produces **local synthetic benchmark evidence**. It does not by itself establish production-agent safety, real-user benefit, trained QLoRA improvement, external MCP interoperability, GitHub-hosted CI success, native Windows success, Docker success, or external benchmark transfer.


## Windows test portability

The local verify workflow does not require Administrator elevation or Windows Developer Mode. The output-directory guard is tested without creating a symlink on accounts that lack symlink privilege; if the platform permits it, an additional real-symlink integration test also runs.

## Windows GPU preflight before model download

Run the GPU preflight first. It parses both PowerShell scripts, creates the Python 3.11 environment, validates tests/lint/type checks, runs the native small-data smoke pipeline, optionally validates the FastMCP adapter when `-IncludeOptionalMcp` is supplied, and probes the NVIDIA/PyTorch/bitsandbytes stack. It does **not** download Qwen weights or train.

```powershell
.\scripts\run_gpu_local.ps1 -Mode Verify
# Optional local FastMCP adapter smoke:
# .\scripts\run_gpu_local.ps1 -Mode Verify -IncludeOptionalMcp
```

Only after `VERIFY_PASS` should you launch the full GPU workflow:

```powershell
.\scripts\run_gpu_local.ps1 -Mode Full
```

The GPU runner installs or validates a CUDA 12.8 PyTorch wheel by default before the QLoRA extras, then validates an actual tiny NF4 bitsandbytes CUDA forward/backward operation plus a PyTorch AdamW update through a JSON probe. Its bootstrap pins `setuptools>=68,<82`, matching the installed PyTorch 2.11 CUDA wheel requirement so a `Full` run remains safe after an earlier `Verify` run. It can use `-TorchCudaWheel cu126` only when that compatibility path is needed. It verifies CUDA, BF16, `nvidia-smi`, and `bitsandbytes`, then starts model download only in `Full` mode. A newer NVIDIA driver can run that wheel; the local CUDA Toolkit version is not used by the prebuilt PyTorch wheel. The QLoRA path uses the model tokenizer's chat template, disables Qwen3 thinking for structured JSON generation, applies completion-only loss masking, enables gradient checkpointing, pins the base-model revision through adapter inference when supplied, and uses standard PyTorch AdamW (`adamw_torch`) for the LoRA parameters while retaining bitsandbytes only for NF4 base-model quantization.

## Quick start

### Linux or macOS

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
make verify
```

`make verify` runs the test suite, lint/type checks, package build, and the small deterministic end-to-end smoke topology. Outputs are generated under `artifacts/smoke/` and verified for required files, data-quality status, incident replay status, and SHA-256 integrity.

For the default benchmark size:

```bash
make run
```

### Windows PowerShell

```powershell
./scripts/run_local.ps1 -Mode validate
./scripts/run_local.ps1 -Mode test
./scripts/run_local.ps1 -Mode quality
./scripts/run_local.ps1 -Mode run -ConfigPath configs/ci_smoke.yaml -Overwrite
```

The wrapper creates `.venv`, validates Python `>=3.11,<3.14`, installs local dependencies, forwards the optional config, and checks every native command exit code. It deliberately disables ambient pytest-plugin autoload for reproducibility, then explicitly loads `pytest_cov` in coverage mode so the coverage command cannot fail because its own plugin was suppressed.

### Docker smoke path

```bash
make docker-smoke
```

The Docker image contains only runtime metadata, `src/`, and `configs/`; generated evidence, virtual environments, tests, and historical reports are excluded from the build context.

## Canonical commands

```bash
rtrace validate-data --output artifacts/validate --seed 17 --overwrite
rtrace run-all --output artifacts/demo --seed 17 --overwrite
python scripts/verify_output.py --output artifacts/demo
rtrace replay-incidents --output artifacts/replays --seed 17 --overwrite
rtrace run-multiseed --output artifacts/multiseed --seeds 11,17,23,29,31 --overwrite
```

The fast CI topology is intentionally smaller while preserving the complete graph:

```bash
rtrace run-all --output artifacts/smoke --seed 17 --config configs/ci_smoke.yaml --overwrite
python scripts/verify_output.py --output artifacts/smoke
```

## Output contract

Each `run-all` output includes data quality, candidate comparison, slice metrics, predictions, calibration and threshold tables, failure cases, decision scenarios, incident replay results, figures, reports, a `run_manifest.json`, and a SHA-256 `core_artifact_manifest.json`. The `verify_output.py` helper fails closed if required files, PASS statuses, sizes, or hashes do not match.

## CI contract

GitHub Actions is configured for Python 3.11/3.13 tests, quality/type checks, clean-wheel installation plus deterministic pipeline smoke, optional FastMCP in-process smoke, Windows PowerShell smoke, Docker smoke, and artifact upload. The workflow must still be executed on GitHub to claim hosted CI evidence.

## Documentation

See `PRE_EXECUTION_VERIFICATION_v0.6.11.md`, `VERIFICATION_REPORT_V0.6.11.md`, `docs/preflight_hardening_v0.6.11.md`, `docs/validation_contract.md`, `docs/monitoring_spec.md`, and `docs/operations.md`.
