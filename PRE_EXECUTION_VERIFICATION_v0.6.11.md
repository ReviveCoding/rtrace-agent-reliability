# Pre-execution verification, v0.6.11

This release addresses the native Windows bitsandbytes failure reached during the first Qwen3-4B QLoRA train step. It replaces the bitsandbytes 8-bit optimizer with `adamw_torch`, strengthens the bitsandbytes runtime probe to cover NF4 forward/backward, and executes a restored in-process Qwen+LoRA train-step preflight before `Trainer.train()`.

Static and CPU-side validation cannot replace native Windows CUDA execution. Run `Verify` before `Full`.
