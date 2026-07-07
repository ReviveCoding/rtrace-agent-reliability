"""Run a minimal, non-claiming generation smoke after local QLoRA adapter training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _model_kwargs(revision: str | None) -> dict[str, Any]:
    return {"revision": revision} if revision else {}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    args = parser.parse_args()

    if args.max_new_tokens < 1:
        raise SystemExit("max-new-tokens must be positive")

    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    except ImportError as exc:
        raise SystemExit(
            "Install the QLoRA optional dependencies before adapter inference."
        ) from exc

    if not torch.cuda.is_available():
        raise SystemExit("Adapter inference requires CUDA in this 4-bit smoke path.")

    adapter = Path(args.adapter)
    if not (adapter / "adapter_config.json").is_file():
        raise SystemExit(f"QLoRA adapter_config.json was not found under: {adapter}")

    model_kwargs = _model_kwargs(args.revision)
    tokenizer = AutoTokenizer.from_pretrained(adapter)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    messages = [
        {
            "role": "system",
            "content": "You are a SafeAssist agent. Return one JSON action object. /no_think",
        },
        {
            "role": "user",
            "content": (
                "Create a calendar event titled Project review tomorrow at 10 AM. "
                "Policy IDs: calendar_write"
            ),
        },
    ]
    try:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError as exc:
        raise SystemExit(
            "The saved tokenizer does not support Qwen3 enable_thinking=False formatting."
        ) from exc

    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    base = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=quant,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        low_cpu_mem_usage=True,
        **model_kwargs,
    )
    model = PeftModel.from_pretrained(base, adapter).eval()
    device = next(model.parameters()).device
    inputs = tokenizer([text], return_tensors="pt").to(device)
    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    completion = tokenizer.decode(
        generated[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True
    ).strip()
    parsed_action: dict[str, Any] | None = None
    json_error: str | None = None
    if completion:
        try:
            parsed = json.loads(completion)
            if isinstance(parsed, dict):
                parsed_action = parsed
            else:
                json_error = "completion_parsed_but_is_not_an_object"
        except json.JSONDecodeError as exc:
            json_error = str(exc)
    else:
        json_error = "empty_completion"

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": "RAW_ADAPTER_INFERENCE_COMPLETED",
        "model": args.model,
        "revision": args.revision,
        "adapter": str(adapter),
        "completion": completion,
        "completion_nonempty": bool(completion),
        "json_object_parsed": parsed_action is not None,
        "parsed_action": parsed_action,
        "json_parse_error": json_error,
        "claim_boundary": (
            "Generation smoke only. This does not establish tool-use correctness, safe routing, "
            "or R-TRACE release-gate improvement."
        ),
    }
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print({"status": payload["status"], "output": str(output), "nonempty": bool(completion)})


if __name__ == "__main__":
    main()
