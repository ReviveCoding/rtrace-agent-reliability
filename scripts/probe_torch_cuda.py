"""Emit a non-failing JSON probe for the optional local PyTorch CUDA runtime.

This helper intentionally returns exit code 0 when torch is missing, CPU-only,
or otherwise not importable. The PowerShell runner uses that state to install the
requested CUDA wheel instead of treating a first-run absence of torch as an error.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from typing import Any


def _payload(expected_cuda: str) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "status": "MISSING",
        "expected_cuda": expected_cuda,
        "torch_installed": False,
        "torch_version": None,
        "torch_cuda": None,
        "cuda_available": False,
        "cuda_ready": False,
        # `ready` is the normalized field used by all optional GPU probes.
        # `cuda_ready` remains for backward-compatible machine diagnostics.
        "ready": False,
        "error": None,
    }
    if importlib.util.find_spec("torch") is None:
        return payload

    payload["torch_installed"] = True
    try:
        import torch

        torch_cuda = torch.version.cuda
        cuda_available = bool(torch.cuda.is_available())
        payload.update(
            {
                "status": "READY"
                if cuda_available and str(torch_cuda).startswith(expected_cuda)
                else "NOT_READY",
                "torch_version": torch.__version__,
                "torch_cuda": torch_cuda,
                "cuda_available": cuda_available,
                "cuda_ready": cuda_available and str(torch_cuda).startswith(expected_cuda),
                "ready": cuda_available and str(torch_cuda).startswith(expected_cuda),
            }
        )
    except BaseException as exc:  # probe only: retain a structured install-needed state
        payload.update({"status": "IMPORT_ERROR", "error": f"{type(exc).__name__}: {exc}"})
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-cuda", required=True)
    args = parser.parse_args()
    print(json.dumps(_payload(args.expected_cuda), sort_keys=True))


if __name__ == "__main__":
    main()
