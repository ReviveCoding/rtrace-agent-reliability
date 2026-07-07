from pathlib import Path


def test_qlora_path_uses_chat_template_completion_only_loss_and_single_gpu_mapping() -> None:
    source = Path("scripts/train_qlora.py").read_text(encoding="utf-8")
    assert "apply_chat_template" in source
    assert "enable_thinking=False" in source
    assert "assistant_completion_only" in source
    assert 'optim="adamw_torch"' in source
    assert "paged_adamw_8bit" not in source
    assert "_run_training_step_preflight" in source
    assert "training_step_preflight.json" in source
    assert "torch_module.optim.AdamW" in source
    assert "gradient_checkpointing=True" in source
    assert 'device_map={"": 0}' in source
    assert "save_total_limit=2" in source
    assert "trust_remote_code" not in source


def test_adapter_inference_reuses_optional_model_revision_and_avoids_model_device_property() -> (
    None
):
    source = Path("scripts/raw_adapter_inference.py").read_text(encoding="utf-8")
    assert 'parser.add_argument("--revision", default=None)' in source
    assert "next(model.parameters()).device" in source
    assert "completion_nonempty" in source
    assert "enable_thinking=False" in source
    assert "model.device" not in source


def test_gpu_powershell_runner_has_safe_verify_mode_and_runtime_preflight() -> None:
    source = Path("scripts/run_gpu_local.ps1").read_text(encoding="utf-8")
    assert 'ValidateSet("Verify", "Full")' in source
    assert "Test-PowerShellParse" in source
    assert "Resolve-Python311" in source
    assert "nvidia-smi failed with exit code" in source
    assert 'ValidateSet("cu126", "cu128")' in source
    assert "https://download.pytorch.org/whl/$TorchCudaWheel" in source
    assert "-IncludeOptionalMcp" in source
    assert "Optional FastMCP smoke skipped" in source
    assert "bitsandbytes>=0.49.2,<0.50" in Path("pyproject.toml").read_text(encoding="utf-8")
    assert "raw_adapter_inference.py" in source
    assert "probe_torch_cuda.py" in source
    assert "probe_bitsandbytes.py" in source
    assert "nf4_forward_backward_torch_adamw_step" in Path(
        "scripts/probe_bitsandbytes.py"
    ).read_text(encoding="utf-8")
    assert "Get-TorchCudaProbe" in source
    assert "Get-BitsAndBytesProbe" in source
    assert "-m bitsandbytes" not in source
    assert "bitsandbytes_probe.json" in source
    assert "Test-TorchCuda" not in source
    assert "torch_probe_before_install.json" in source
    assert "torch_probe_after_install.json" in source
    assert "TorchProbeBefore.ready" in source
    assert "TorchProbeAfter.ready" in source
    assert "failure_manifest.json" in source


def test_gpu_powershell_runner_pins_setuptools_below_the_torch_211_ceiling() -> None:
    source = Path("scripts/run_gpu_local.ps1").read_text(encoding="utf-8")
    pyproject = Path("pyproject.toml").read_text(encoding="utf-8")
    assert '"setuptools>=68,<82"' in source
    assert '"verify-setuptools-compatible"' in source
    assert source.index('"setuptools>=68,<82"') < source.index('"pip-check-base"')
    assert 'requires = ["setuptools>=68,<82", "wheel"]' in pyproject
