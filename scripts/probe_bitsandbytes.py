"""Emit a non-failing JSON probe for the local bitsandbytes 4-bit CUDA path.

The PowerShell runner must not call ``python -m bitsandbytes`` directly: PyTorch or
bitsandbytes may write non-fatal diagnostics to stderr, and Windows PowerShell 5.1
can promote native stderr output to an error record when ``ErrorActionPreference``
is ``Stop``. This helper runs the actual import and a tiny NF4 Linear4bit operation
inside a child Python process, captures that child's stderr, and emits one JSON
result on stdout. A warning such as a missing Triton flop-counter extension is
recorded, not treated as an operational failure, if the NF4 operation succeeds.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
from typing import Any


def _child_program() -> str:
    """Return the isolated CUDA/Linear4bit validation program."""
    return r"""
import json
import traceback

payload = {
    "status": "OPERATION_ERROR",
    "ready": False,
    "bitsandbytes_version": None,
    "torch_version": None,
    "torch_cuda": None,
    "device": None,
    "output_shape": None,
    "error": None,
}
try:
    import torch
    import torch.nn as nn
    import bitsandbytes as bnb
    from bitsandbytes.nn import Linear4bit
    from transformers import BitsAndBytesConfig

    if not torch.cuda.is_available():
        raise RuntimeError("torch.cuda.is_available() is False")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("torch.cuda.is_bf16_supported() is False")

    config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    if not config.load_in_4bit or config.bnb_4bit_quant_type != "nf4":
        raise RuntimeError("BitsAndBytesConfig did not retain the requested NF4 configuration")

    source = nn.Linear(8, 8, bias=False, dtype=torch.bfloat16)
    quantized = Linear4bit(
        8,
        8,
        bias=False,
        compute_dtype=torch.bfloat16,
        compress_statistics=True,
        quant_type="nf4",
    )
    quantized.load_state_dict(source.state_dict())
    quantized = quantized.to("cuda")
    adapter = nn.Linear(8, 8, bias=False, dtype=torch.bfloat16).to("cuda")
    sample = torch.randn(2, 8, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    output = quantized(sample) + adapter(sample)
    loss = output.float().square().mean()
    loss.backward()
    if sample.grad is None or adapter.weight.grad is None:
        raise RuntimeError("NF4 backward path did not populate required gradients")
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=1e-4)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    if tuple(output.shape) != (2, 8):
        raise RuntimeError(f"unexpected Linear4bit output shape: {tuple(output.shape)}")

    payload.update(
        {
            "status": "READY",
            "ready": True,
            "bitsandbytes_version": getattr(bnb, "__version__", None),
            "torch_version": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(0),
            "output_shape": list(output.shape),
            "operation": "nf4_forward_backward_torch_adamw_step",
        }
    )
except BaseException as exc:
    payload["error"] = f"{type(exc).__name__}: {exc}"
    payload["traceback"] = traceback.format_exc(limit=6)
print(json.dumps(payload, sort_keys=True))
"""


def _last_json_line(text: str) -> dict[str, Any] | None:
    for line in reversed(text.splitlines()):
        candidate = line.strip()
        if not candidate.startswith("{"):
            continue
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _trim(text: str, limit: int = 6000) -> str:
    text = text.strip()
    return text if len(text) <= limit else text[-limit:]


def _payload(expected_cuda: str) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "status": "MISSING",
        "ready": False,
        "expected_cuda": expected_cuda,
        "bitsandbytes_installed": False,
        "torch_installed": False,
        "child_exit_code": None,
        "child_stderr": "",
        "error": None,
    }
    try:
        payload["bitsandbytes_installed"] = importlib.util.find_spec("bitsandbytes") is not None
        payload["torch_installed"] = importlib.util.find_spec("torch") is not None
    except BaseException as exc:
        payload.update({"status": "DISCOVERY_ERROR", "error": f"{type(exc).__name__}: {exc}"})
        return payload

    if not payload["bitsandbytes_installed"] or not payload["torch_installed"]:
        return payload

    try:
        completed = subprocess.run(
            [sys.executable, "-c", _child_program()],
            check=False,
            capture_output=True,
            text=True,
            errors="replace",
            timeout=90,
        )
    except BaseException as exc:
        payload.update({"status": "PROBE_ERROR", "error": f"{type(exc).__name__}: {exc}"})
        return payload

    payload["child_exit_code"] = completed.returncode
    payload["child_stderr"] = _trim(completed.stderr)
    child_payload = _last_json_line(completed.stdout)
    if child_payload is None:
        payload.update(
            {
                "status": "CHILD_PROTOCOL_ERROR",
                "error": f"child did not emit a JSON payload; stdout={_trim(completed.stdout)!r}",
            }
        )
        return payload
    if completed.returncode != 0:
        payload.update(
            {
                "status": "CHILD_EXIT_NONZERO",
                "error": f"child exited with code {completed.returncode}: {child_payload.get('error')}",
                "child_payload": child_payload,
            }
        )
        return payload

    payload.update(child_payload)
    payload["expected_cuda"] = expected_cuda
    payload["cuda_matches_expected"] = str(payload.get("torch_cuda") or "").startswith(
        expected_cuda
    )
    payload["ready"] = bool(payload.get("ready")) and bool(payload["cuda_matches_expected"])
    if payload["status"] == "READY" and not payload["cuda_matches_expected"]:
        payload["status"] = "CUDA_VERSION_MISMATCH"
        payload["error"] = f"expected CUDA {expected_cuda}, received {payload.get('torch_cuda')}"
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-cuda", required=True)
    args = parser.parse_args()
    print(json.dumps(_payload(args.expected_cuda), sort_keys=True))


if __name__ == "__main__":
    main()
