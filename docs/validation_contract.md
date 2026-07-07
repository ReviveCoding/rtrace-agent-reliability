# Validation contract

## Supported local path

1. Create a Python 3.11-3.13 virtual environment.
2. Install `.[dev]`.
3. Run `make verify` for the small deterministic topology or `make run` for the default benchmark.
4. Use `python scripts/verify_output.py --output <run-directory>` to validate required files, data-quality status, incident replay status, and SHA-256 integrity for core artifacts.

## Release evidence boundary

The repository produces local synthetic SafeAssist-MCP evidence only. It does not establish production safety, real-user utility, QLoRA training improvement, external MCP interoperability, hosted GitHub Actions execution, native Windows execution, or Docker execution unless each has separately materialized run evidence.

## CI contract

The workflow validates source tests on Python 3.11 and 3.13, quality checks, clean-wheel install, deterministic smoke topology, optional FastMCP in-process registration, Windows PowerShell wrapper, and Docker smoke. A workflow file is not equivalent to hosted-run evidence; GitHub Actions must be run after push and its artifacts retained.
