# Operations and Lean-Source Policy

## Source-tree policy

The repository keeps only executable source, configuration, tests, packaging metadata, CI, and current operational documentation. Generated benchmark outputs, package metadata, build directories, caches, coverage files, local virtual environments, SQLite state, and historical release evidence are excluded through `.gitignore`, `.dockerignore`, and `make clean`. This prevents stale synthetic metrics from being mistaken for current-source evidence and keeps clone and Docker build contexts small.

## Idempotent commands

- `make smoke` and `make run` intentionally use `--overwrite` because their paths are fixed generated-artifact locations.
- Direct CLI use stays fail-closed by default: a non-empty output directory requires an explicit `--overwrite`.
- `rtrace run-multiseed` resumes only compatible seed directories. It now rejects filesystem roots, project roots, current working directories, ancestors, symlinks, and non-directories before writing resumable artifacts.
- `make clean` removes build output, virtual wheel environments, caches, coverage data, generated artifacts, and editable package metadata.

## Small-data verification record

The CI smoke configuration preserves the complete pipeline graph while reducing sample counts and estimator count. It exercises:

```text
benchmark generation
→ schema / leakage validation
→ C0-C5 evaluation
→ calibration and development-only routing selection
→ SQLite execution and incident replay
→ reports and artifact manifest
→ fail-closed verification
```

`run_manifest.json` records the effective configuration, source fingerprint, hardware metadata, and exact installed versions for core runtime packages. The smoke benchmark may return `REVIEW` under the protected C3/C5 release criteria. That is a model-release decision, not a pipeline failure.

## Evidence not yet established

A source-level workflow, Dockerfile, and PowerShell wrapper do not substitute for hosted or native runtime evidence. Before claiming those environments passed, run and retain artifacts for GitHub-hosted Actions, native Windows PowerShell, Docker daemon execution, GPU QLoRA training, human adjudication, and external transfer evaluation.
