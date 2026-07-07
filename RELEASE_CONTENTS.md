# R-TRACE v0.6.11 delivery contents

This source-only delivery contains the Python package, deterministic SafeAssist-MCP benchmark, tests, PowerShell runners, GitHub Actions workflow, Docker configuration, and the current v0.6.11 pre-execution and verification reports.

It excludes virtual environments, test caches, Python bytecode, build output, wheel output, egg metadata, generated benchmark artifacts, and obsolete release reports.

## v0.6.11 change

A native Windows full run reached the first Qwen3-4B QLoRA training step after model download and failed inside the bitsandbytes native backend. The v0.6.10 runner used bitsandbytes for both NF4 base-model quantization and the 8-bit paged AdamW optimizer, while the prior Verify probe tested only a minimal NF4 forward operation.

This release keeps bitsandbytes only for NF4 base-model quantization, uses Transformers `adamw_torch` for the small set of trainable LoRA parameters, extends the GPU probe to test NF4 forward/backward plus a PyTorch AdamW update, and adds a restored Qwen+LoRA in-process train-step preflight before `Trainer.train()`. It also retains the v0.6.10 `setuptools>=68,<82` bootstrap constraint required by the installed PyTorch 2.11 CUDA wheel.

## Claim boundary

This delivery has local source, synthetic pipeline, multiseed, package-wheel, static PowerShell/CI, and targeted QLoRA regression validation. It does not claim an executed native Windows full QLoRA training pass, Docker daemon execution, hosted GitHub Actions, API-Bank adaptation, or external-benchmark evidence.
