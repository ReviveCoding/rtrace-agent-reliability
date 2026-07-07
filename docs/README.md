# R-TRACE operational documentation

- `validation_contract.md`: reproducibility, output, CI, and claim boundaries.
- `monitoring_spec.md`: monitoring signals and requalification triggers.
- `operations.md`: clean-source policy, idempotent developer commands, small-data verification record, and unresolved runtime evidence.

Generated benchmark outputs are deliberately excluded from source control. Run `make smoke` or `make verify` to regenerate them, then archive immutable evidence separately with the generated `core_artifact_manifest.json`.
