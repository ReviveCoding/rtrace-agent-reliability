from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

from .utils import ensure_dir


def _save(fig, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def write_figures(
    output: Path,
    comparison: pd.DataFrame,
    records: pd.DataFrame,
    frontier: pd.DataFrame,
    calibration: pd.DataFrame,
    reference_ablation: pd.DataFrame,
) -> None:
    figures = ensure_dir(output / "figures")

    fig = plt.figure()
    plt.bar(comparison["candidate_id"], comparison["safe_completion_at_1"])
    plt.ylim(0, 1)
    plt.title("Human-Handoff Oracle Completion by Candidate")
    plt.ylabel("Outcome with evaluator-side human handoff")
    _save(fig, figures / "candidate_handoff_oracle_completion.png")

    fig = plt.figure()
    plt.bar(comparison["candidate_id"], comparison["autonomous_safe_completion_at_1"])
    plt.ylim(0, 1)
    plt.title("Autonomous Safe Completion by Candidate")
    plt.ylabel("No human assistance")
    _save(fig, figures / "candidate_autonomous_completion.png")

    fig = plt.figure()
    plt.bar(comparison["candidate_id"], comparison["critical_false_greenlight_rate"])
    plt.title("Critical False-Greenlight Rate")
    plt.ylabel("Rate")
    _save(fig, figures / "critical_false_greenlight.png")

    fig = plt.figure()
    plt.plot(frontier["review_rate"], frontier["false_greenlight_rate"], marker="o")
    plt.title("Review Capacity vs Critical False-Greenlight")
    plt.xlabel("Review / confirmation rate")
    plt.ylabel("False-greenlight rate")
    _save(fig, figures / "threshold_frontier.png")

    fig = plt.figure()
    plt.plot([0, 1], [0, 1], linestyle="--", label="perfect calibration")
    plt.plot(
        calibration["mean_predicted_risk"],
        calibration["observed_critical_rate"],
        marker="o",
        label="observed",
    )
    plt.title("Risk Calibration")
    plt.xlabel("Predicted critical-action risk")
    plt.ylabel("Observed critical-action rate")
    plt.legend()
    _save(fig, figures / "calibration_curve.png")

    taxonomy = (
        records.groupby("failure_family")["false_greenlight"].mean().sort_values(ascending=False)
    )
    fig = plt.figure()
    plt.bar(taxonomy.index.astype(str), taxonomy.values)
    plt.title("Failure Family: Critical False-Greenlight")
    plt.xticks(rotation=30, ha="right")
    _save(fig, figures / "failure_taxonomy.png")

    fig = plt.figure()
    plt.bar(reference_ablation["reference_strategy"], reference_ablation["risk_pr_auc"])
    plt.ylim(0, 1)
    plt.title("R-TRACE Reference Strategy Sensitivity")
    plt.ylabel("Risk PR-AUC")
    _save(fig, figures / "reference_ablation.png")


def write_reports(
    output: Path,
    comparison: pd.DataFrame,
    release: dict[str, Any],
    manifest: dict[str, Any],
    paired: dict[str, float | str],
    reference_ablation: pd.DataFrame,
) -> None:
    reports = ensure_dir(output / "reports")
    best_autonomous = comparison.sort_values(
        "autonomous_safe_completion_at_1", ascending=False
    ).iloc[0]
    lines = [
        "# Evaluation Report",
        "",
        "## Claim boundary",
        "This run is local, synthetic, deterministic evaluation evidence. It does not certify production deployment, real-user safety, externally audited MCP security, or QLoRA performance.",
        "",
        "## Human-handoff boundary",
        "`safe_completion_at_1` includes an evaluator-side oracle handoff that resolves routed tasks. It is a workflow upper-bound metric, not autonomous-agent performance. The production-facing release comparison uses autonomous safe completion, false-greenlight, scope, calibration, incident replay, and the paired autonomous interval.",
        "",
        "## Oracle-leakage boundary",
        "Gold action and scope fields are used only for final labels and post-hoc outcome measurement. Runtime critics, routers, reference features, and SASC use candidate outputs, declared policy, runtime context, and state observations only.",
        "",
        "## Candidate comparison",
        comparison.to_markdown(index=False),
        "",
        "## Selected autonomous result",
        f"Highest autonomous safe completion in this run: `{best_autonomous['candidate_id']}` at `{best_autonomous['autonomous_safe_completion_at_1']:.3f}`.",
        "",
        "## Paired autonomous comparison C5 minus C3",
        pd.DataFrame([paired]).to_markdown(index=False),
        "",
        "## Reference sensitivity",
        reference_ablation.to_markdown(index=False),
        "",
        "## Release decision",
        f"Verdict: **{release['verdict']}**",
        "",
        "### Reasons",
        *[f"- {reason}" for reason in release["reasons"]],
        "",
        "## Artifact lineage",
        f"Run ID: `{manifest['run_id']}`",
        f"Dataset fingerprint: `{manifest['dataset_fingerprint']}`",
        f"Source fingerprint: `{manifest['source_fingerprint']}`",
        f"Config fingerprint: `{manifest['config_fingerprint']}`",
    ]
    (reports / "evaluation_report.md").write_text("\n".join(lines), encoding="utf-8")
    card = [
        "# Release Card",
        f"Verdict: **{release['verdict']}**",
        "",
        "This release card applies only to the run manifest and synthetic local SafeAssist-MCP benchmark represented here.",
        "",
        "## Gates",
        *[f"- {key}: {'PASS' if value else 'FAIL'}" for key, value in release["gates"].items()],
    ]
    (reports / "release_card.md").write_text("\n".join(card), encoding="utf-8")


def write_multiseed_reports(output: Path, per_seed: pd.DataFrame) -> None:
    """Write deterministic multi-seed evidence tables, figures, and a concise report.

    This is deliberately post-hoc aggregation only: each seed's C5 operating point is
    selected before the corresponding final split is evaluated by ``run_all``.
    """
    import numpy as np

    reports = ensure_dir(output / "reports")
    figures = ensure_dir(output / "figures")
    c3 = per_seed.loc[per_seed["candidate_id"] == "C3"].set_index("seed")
    c5 = per_seed.loc[per_seed["candidate_id"] == "C5"].set_index("seed")
    shared = sorted(set(c3.index).intersection(c5.index))
    if not shared:
        raise ValueError("multi-seed report requires C3 and C5 rows for at least one shared seed")

    metrics = [
        "safe_completion_at_1",
        "autonomous_safe_completion_at_1",
        "critical_false_greenlight_rate",
        "unauthorized_scope_rate",
        "scope_f1",
        "risk_pr_auc",
        "brier",
        "harm_weighted_cost",
        "confirmation_burden",
        "mean_latency_ms",
    ]
    rng = np.random.default_rng(20260626)
    rows: list[dict[str, float | str | int]] = []
    for metric in metrics:
        delta = (c5.loc[shared, metric] - c3.loc[shared, metric]).to_numpy(dtype=float)
        bootstrap = np.array(
            [rng.choice(delta, size=len(delta), replace=True).mean() for _ in range(10_000)]
        )
        rows.append(
            {
                "metric": metric,
                "n_seeds": len(delta),
                "c3_mean": float(c3.loc[shared, metric].mean()),
                "c5_mean": float(c5.loc[shared, metric].mean()),
                "c5_minus_c3_mean": float(delta.mean()),
                "c5_minus_c3_std": float(delta.std(ddof=1)) if len(delta) > 1 else 0.0,
                "bootstrap_95_low": float(np.quantile(bootstrap, 0.025)),
                "bootstrap_95_high": float(np.quantile(bootstrap, 0.975)),
            }
        )
    comparison = pd.DataFrame(rows)
    comparison.to_csv(output / "c5_vs_c3_seed_level_summary.csv", index=False)

    fig = plt.figure()
    safe = comparison.loc[comparison["metric"] == "safe_completion_at_1"].iloc[0]
    auto = comparison.loc[comparison["metric"] == "autonomous_safe_completion_at_1"].iloc[0]
    # Use points rather than stacked bars to avoid visually implying additive outcomes.
    plt.plot(
        ["Safe completion", "Autonomous safe completion"],
        [safe["c3_mean"], auto["c3_mean"]],
        marker="o",
        label="C3",
    )
    plt.plot(
        ["Safe completion", "Autonomous safe completion"],
        [safe["c5_mean"], auto["c5_mean"]],
        marker="o",
        label="C5",
    )
    plt.ylim(0, 1)
    plt.ylabel("Mean across independent seeds")
    plt.title("C3 vs C5 Completion Metrics")
    plt.legend()
    _save(fig, figures / "c3_c5_completion_multiseed.png")

    risk_metrics = [
        "critical_false_greenlight_rate",
        "unauthorized_scope_rate",
        "brier",
        "harm_weighted_cost",
    ]
    risk_rows = comparison.set_index("metric").loc[risk_metrics].reset_index()
    fig = plt.figure()
    plt.bar(risk_rows["metric"], risk_rows["c5_minus_c3_mean"])
    plt.xticks(rotation=25, ha="right")
    plt.axhline(0, linewidth=1)
    plt.ylabel("C5 minus C3")
    plt.title("C5 vs C3 Risk and Cost Deltas")
    _save(fig, figures / "c3_c5_risk_delta_multiseed.png")

    lines = [
        "# Multi-Seed Evidence Report",
        "",
        "## Boundary",
        "This aggregation summarizes five independent runs of the local synthetic SafeAssist-MCP benchmark. Each run selected the C5 operating point on development data before final evaluation. These intervals summarize only seed variation in this simulator; they are not production or external-validity confidence intervals.",
        "",
        "## C5 versus C3",
        comparison.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Interpretation rules",
        "- Higher is better for safe completion, autonomous safe completion, Scope F1, and PR-AUC.",
        "- Lower is better for false-greenlight rate, unauthorized-scope rate, Brier score, harm-weighted cost, and latency.",
        "- Confirmation burden is a safety-utility tradeoff and is not automatically treated as an improvement.",
    ]
    (reports / "multiseed_evidence.md").write_text("\n".join(lines), encoding="utf-8")
