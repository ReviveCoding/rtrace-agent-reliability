# v0.6.11 Windows QLoRA train-step hardening

## Trigger

A native Windows full run reached the first Qwen3-4B QLoRA training step and stopped with `Error initialization error ... pythonInterface.cpp`. Earlier preflight only validated a minimal bitsandbytes NF4 forward operation.

## Change

- Retired `paged_adamw_8bit` from the Windows QLoRA path. The project retains bitsandbytes for NF4 base-model quantization but uses Transformers `adamw_torch` for trainable LoRA parameters.
- Expanded the bitsandbytes probe to run NF4 forward, backward, and a standard PyTorch AdamW update.
- Added an in-process Qwen + LoRA forward/backward/AdamW preflight before `Trainer.train()`. It restores LoRA weights afterward and writes `training_step_preflight.json`.

## Boundary

The native Windows GPU must still execute the v0.6.11 Verify mode and then Full mode. This source package cannot claim that the real Qwen train step has passed until that machine-specific run completes.
