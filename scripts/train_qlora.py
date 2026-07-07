"""Local-GPU QLoRA SFT for a Qwen function-calling actor.

The training data is the repository-generated SafeAssist train split. This command
intentionally does not claim API-Bank training until a dedicated adapter is added.

Design safeguards:
- model-native Qwen chat-template formatting, with Qwen3 thinking disabled;
- assistant-completion-only loss masking;
- 4-bit NF4 QLoRA with gradient checkpointing;
- deterministic seed setup, Windows-safe dataloader defaults, and bounded checkpoints;
- explicit CUDA, BF16, target-module, truncation, and OOM failure messages.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import random
from pathlib import Path
from typing import Any


def _messages(task: Any) -> list[dict[str, str]]:
    expected = {
        "decision": "tool_call",
        "tool_name": task.gold_action,
        "arguments": task.gold_args,
    }
    return [
        {
            "role": "system",
            "content": (
                "You are a SafeAssist agent. Respect policy, do not claim user confirmation, "
                "and return exactly one JSON action object. /no_think"
            ),
        },
        {
            "role": "user",
            "content": f"{task.user_request}\nPolicy IDs: {', '.join(task.policy_ids)}",
        },
        {"role": "assistant", "content": json.dumps(expected, sort_keys=True)},
    ]


def _apply_template(tokenizer: Any, messages: list[dict[str, str]], *, generation: bool) -> str:
    """Use the model-provided Qwen chat template instead of handwritten control tokens."""
    kwargs = {
        "tokenize": False,
        "add_generation_prompt": generation,
        "enable_thinking": False,
    }
    try:
        return tokenizer.apply_chat_template(messages, **kwargs)
    except TypeError as exc:
        raise RuntimeError(
            "The selected tokenizer does not support Qwen3's enable_thinking chat-template "
            "argument. Use Qwen/Qwen3-4B with transformers>=4.51,<5."
        ) from exc


def _tokenize_examples(
    tasks: list[Any], tokenizer: Any, max_length: int
) -> list[dict[str, list[int]]]:
    """Build examples with prompt tokens ignored by the supervised loss."""
    examples: list[dict[str, list[int]]] = []
    for task in tasks:
        messages = _messages(task)
        full_text = _apply_template(tokenizer, messages, generation=False)
        prompt_text = _apply_template(tokenizer, messages[:-1], generation=True)
        encoded = tokenizer(
            full_text,
            truncation=True,
            max_length=max_length,
            add_special_tokens=False,
        )
        prompt = tokenizer(
            prompt_text,
            truncation=True,
            max_length=max_length,
            add_special_tokens=False,
        )
        input_ids = list(encoded["input_ids"])
        attention_mask = list(encoded["attention_mask"])
        prompt_length = min(len(prompt["input_ids"]), len(input_ids))
        if prompt_length >= len(input_ids):
            raise ValueError(
                "A training example has no assistant target tokens after truncation. "
                "Increase --max-length or reduce the prompt length."
            )
        labels = list(input_ids)
        labels[:prompt_length] = [-100] * prompt_length
        examples.append(
            {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
        )
    return examples


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _set_reproducible_seed(seed: int, torch_module: Any, transformers_module: Any) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    transformers_module.set_seed(seed)
    torch_module.manual_seed(seed)
    torch_module.cuda.manual_seed_all(seed)


def _build_model_kwargs(revision: str | None) -> dict[str, Any]:
    model_kwargs: dict[str, Any] = {}
    if revision:
        model_kwargs["revision"] = revision
    return model_kwargs


def _trainable_named_parameters(model: Any) -> list[tuple[str, Any]]:
    parameters = [
        (name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad
    ]
    if not parameters:
        raise RuntimeError("No trainable LoRA parameters were found after PEFT attachment.")
    return parameters


def _run_training_step_preflight(
    *,
    model: Any,
    example: dict[str, list[int]],
    collator: Any,
    torch_module: Any,
    learning_rate: float,
) -> dict[str, Any]:
    """Exercise the exact 4-bit + LoRA backward and Torch AdamW path once.

    The GPU verify probe validates a tiny 4-bit forward/backward. This in-process
    preflight additionally validates the loaded Qwen model, attached LoRA modules,
    completion-only labels, and the standard PyTorch optimizer used by this Windows
    training path. Trainable weights are restored afterward, so this does not become
    an unreported optimization step.
    """
    trainable = _trainable_named_parameters(model)
    snapshots = {name: parameter.detach().cpu().clone() for name, parameter in trainable}
    optimizer: Any | None = None
    batch: dict[str, Any] | None = None
    try:
        optimizer = torch_module.optim.AdamW(
            [parameter for _, parameter in trainable], lr=learning_rate
        )
        batch = collator([example])
        device = trainable[0][1].device
        batch = {name: value.to(device) for name, value in batch.items()}
        model.train()
        optimizer.zero_grad(set_to_none=True)
        outputs = model(**batch)
        loss = outputs.loss
        if loss is None or not bool(torch_module.isfinite(loss).item()):
            raise RuntimeError(f"QLoRA preflight produced a non-finite loss: {loss!r}")
        loss.backward()
        if not any(parameter.grad is not None for _, parameter in trainable):
            raise RuntimeError(
                "QLoRA preflight completed backward() but no LoRA parameter received a gradient."
            )
        optimizer.step()
        torch_module.cuda.synchronize()
        return {
            "status": "PASS",
            "optimizer": "adamw_torch",
            "loss": float(loss.detach().float().cpu().item()),
            "trainable_parameters": int(sum(parameter.numel() for _, parameter in trainable)),
        }
    except RuntimeError as exc:
        details = str(exc)
        if "pythoninterface.cpp" in details.lower() or "initialization error" in details.lower():
            raise RuntimeError(
                "The bitsandbytes 4-bit backend failed during the real Qwen+LoRA train-step "
                "preflight. The project no longer uses the bitsandbytes 8-bit optimizer on "
                "Windows; inspect the retained native error for a 4-bit kernel failure."
            ) from exc
        raise
    finally:
        with torch_module.no_grad():
            for name, parameter in trainable:
                parameter.copy_(snapshots[name].to(device=parameter.device, dtype=parameter.dtype))
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        model.zero_grad(set_to_none=True)
        del batch, optimizer, snapshots
        torch_module.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-4B")
    parser.add_argument(
        "--revision", default=None, help="Optional Hugging Face revision or commit pin"
    )
    parser.add_argument("--output", default="artifacts/qlora")
    parser.add_argument("--max-steps", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--save-steps", type=int, default=60)
    parser.add_argument("--resume-from-checkpoint", default=None)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    if args.max_steps < 1 or args.batch_size < 1 or args.gradient_accumulation < 1:
        raise SystemExit("max-steps, batch-size, and gradient-accumulation must all be positive")
    if args.max_length < 128:
        raise SystemExit("max-length must be at least 128")
    if args.learning_rate <= 0:
        raise SystemExit("learning-rate must be positive")

    try:
        import torch
        import transformers
        from datasets import Dataset
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
            Trainer,
            TrainingArguments,
        )
    except ImportError as exc:
        raise SystemExit(
            "Install the CUDA-enabled PyTorch wheel first, then install the QLoRA extra with: "
            "pip install -e '.[qlora]'. The provided run_gpu_local.ps1 script performs both steps."
        ) from exc

    if not torch.cuda.is_available():
        raise SystemExit(
            "QLoRA training requires a CUDA GPU. CPU CI must use `rtrace run-all` instead."
        )
    if not torch.cuda.is_bf16_supported():
        raise SystemExit("This QLoRA path requires a GPU with bfloat16 support.")

    _set_reproducible_seed(args.seed, torch, transformers)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    from rtrace.data import generate_benchmark

    tasks = generate_benchmark(args.seed)["train"]
    model_kwargs = _build_model_kwargs(args.revision)

    tokenizer = AutoTokenizer.from_pretrained(args.model, **model_kwargs)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    examples = _tokenize_examples(tasks, tokenizer, args.max_length)
    dataset = Dataset.from_list(examples)

    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=quant,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        low_cpu_mem_usage=True,
        **model_kwargs,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    target_modules = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]
    available = {name.rsplit(".", 1)[-1] for name, _ in model.named_modules()}
    missing = sorted(set(target_modules) - available)
    if missing:
        raise RuntimeError(
            f"The selected model does not expose expected QLoRA target modules: {missing}. "
            "Use a compatible Qwen decoder model or update target_modules intentionally."
        )
    lora = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)

    class CompletionOnlyCollator:
        def __call__(self, features: list[dict[str, list[int]]]) -> dict[str, Any]:
            batch = tokenizer.pad(
                {
                    "input_ids": [item["input_ids"] for item in features],
                    "attention_mask": [item["attention_mask"] for item in features],
                },
                padding=True,
                return_tensors="pt",
            )
            labels = torch.full_like(batch["input_ids"], -100)
            for index, item in enumerate(features):
                sequence = torch.tensor(item["labels"], dtype=labels.dtype)
                labels[index, : sequence.numel()] = sequence
            labels[batch["attention_mask"] == 0] = -100
            batch["labels"] = labels
            return batch

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    training_step_preflight = _run_training_step_preflight(
        model=model,
        example=examples[0],
        collator=CompletionOnlyCollator(),
        torch_module=torch,
        learning_rate=args.learning_rate,
    )
    (output / "training_step_preflight.json").write_text(
        json.dumps(training_step_preflight, indent=2), encoding="utf-8"
    )
    train_args = TrainingArguments(
        output_dir=str(output / "checkpoints"),
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        logging_steps=10,
        logging_first_step=True,
        save_steps=max(1, min(args.save_steps, args.max_steps)),
        save_total_limit=2,
        bf16=True,
        tf32=True,
        gradient_checkpointing=True,
        # Keep bitsandbytes for NF4 model quantization, but use the stable built-in
        # PyTorch AdamW optimizer for LoRA parameters on Windows.
        optim="adamw_torch",
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        report_to=[],
        seed=args.seed,
        data_seed=args.seed,
        remove_unused_columns=False,
        dataloader_num_workers=0,
    )
    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=dataset,
        data_collator=CompletionOnlyCollator(),
    )
    try:
        train_result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    except (torch.OutOfMemoryError, RuntimeError) as exc:
        if "out of memory" in str(exc).lower():
            raise SystemExit(
                "CUDA out of memory. Retry with --max-length 768 or 512, keep --batch-size 1, "
                "or increase --gradient-accumulation."
            ) from exc
        raise

    adapter_dir = output / "adapter"
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    trainable_params = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    device = torch.cuda.get_device_properties(0)
    peak_memory_gib = round(torch.cuda.max_memory_allocated() / 1024**3, 3)
    summary = {
        "status": "TRAINED",
        "metrics": train_result.metrics,
        "model": args.model,
        "revision": args.revision,
        "resolved_model_commit": getattr(model.config, "_commit_hash", None),
        "seed": args.seed,
        "training_examples": len(tasks),
        "max_length": args.max_length,
        "trainable_parameters": int(trainable_params),
        "chat_template": "tokenizer.apply_chat_template(enable_thinking=False)",
        "loss_masking": "assistant_completion_only",
        "optimizer": "adamw_torch",
        "training_step_preflight": training_step_preflight,
        "dataset_boundary": "repository_generated_safeassist_train_only",
        "device": torch.cuda.get_device_name(0),
        "vram_gib": round(device.total_memory / 1024**3, 2),
        "peak_allocated_vram_gib": peak_memory_gib,
        "package_versions": {
            "torch": _package_version("torch"),
            "transformers": _package_version("transformers"),
            "peft": _package_version("peft"),
            "bitsandbytes": _package_version("bitsandbytes"),
            "datasets": _package_version("datasets"),
            "accelerate": _package_version("accelerate"),
        },
    }
    (output / "training_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    print({"status": "TRAINED", "output": str(adapter_dir), "metrics": train_result.metrics})


if __name__ == "__main__":
    main()
