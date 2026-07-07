# R-TRACE v0.6.11 verification report

## Native trigger and scope

A native Windows `-Mode Full` run completed source validation, CUDA/BF16 checks, the deterministic benchmark, and Qwen3-4B weight loading. It stopped at the first QLoRA trainer step with:

```text
Error initialization error at line 670 in file ... bitsandbytes ... pythonInterface.cpp
```

The prior GPU verification checked an isolated NF4 forward operation but did not exercise a Qwen+LoRA backward pass or the configured `paged_adamw_8bit` optimizer. Therefore it was insufficient to claim the full training path had been checked.

## v0.6.11 remediation

- Replaced `optim="paged_adamw_8bit"` with `optim="adamw_torch"`. bitsandbytes remains required for NF4 base-model quantization; only the bitsandbytes 8-bit optimizer path is removed.
- Expanded `scripts/probe_bitsandbytes.py` from NF4 forward-only to NF4 forward, backward, and a standard PyTorch AdamW optimizer update.
- Added an in-process Qwen+LoRA train-step preflight in `scripts/train_qlora.py`. It checks completion-only batch construction, finite loss, LoRA gradients, and a PyTorch AdamW update, restores all trainable weights after the probe, and writes `training_step_preflight.json` before `Trainer.train()` begins.
- Added regression tests which forbid `paged_adamw_8bit` in the Windows QLoRA path and require the expanded probe plus model-level preflight.
- Preserved the existing `setuptools>=68,<82` guard required by the installed PyTorch 2.11 CUDA wheel.

## Executed validation in this release workspace

| Check | Result |
|---|---|
| Fresh editable installation and `pip check` | PASS |
| Test suite | 61 passed, 1 skipped |
| Branch coverage | 86.92% |
| Ruff lint and format checks | PASS |
| Mypy | PASS, 0 errors across 20 source files |
| `compileall` over `src` and `scripts` | PASS |
| QLoRA train and bitsandbytes probe CLI help | PASS |
| CI-smoke `run-all` | PASS; intentional release verdict `REVIEW` |
| Output artifact verification | PASS; 15 required artifacts |
| Incident replay | PASS; 12/12 |
| Multi-seed run and resume/reuse | PASS in the immediately preceding v0.6.11 working tree check; the QLoRA-only final edit does not affect deterministic pipeline source |
| Fresh wheel build, install, and `pip check` | PASS |
| Installed wheel CLI help | PASS |

## Remaining machine-specific gate

This environment cannot execute native Windows PowerShell or the RTX 4090 CUDA runtime. The only valid final confirmation for the remediation is a fresh Windows run using this v0.6.11 source:

```powershell
.\scripts\run_gpu_local.ps1 -Mode Verify
.\scripts\run_gpu_local.ps1 -Mode Full
```

The `Verify` probe now exercises the NF4 backward path. During `Full`, the model-level `training_step_preflight.json` must show `"status": "PASS"` before the 120-step Trainer loop begins. If the 4-bit backend itself fails, the runner now fails before training with a targeted Qwen+LoRA preflight message instead of failing ambiguously at step 0.

## Claim boundary

This report does not claim a successful native Windows Qwen3-4B QLoRA training run. It documents a targeted source remediation and reproducible Linux-side package/pipeline validation after the observed Windows failure.
