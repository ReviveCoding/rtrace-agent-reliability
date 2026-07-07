from rtrace.actors import GenericSFTActor
from rtrace.data import generate_benchmark, validate_benchmark
from rtrace.features import feature_row, preflight_context
from rtrace.models import BASE_FEATURES
from rtrace.policy import PolicyEngine


def test_runtime_features_exclude_oracle_scope_and_action_match():
    task = generate_benchmark(
        17,
        {
            "train": 1,
            "development": 0,
            "calibration": 0,
            "final_clean": 0,
            "final_hard": 0,
            "final_compositional": 0,
        },
    )["train"][0]
    actor = GenericSFTActor()
    policy = PolicyEngine()
    candidate = actor.propose(task, 17)
    row = feature_row(task, candidate, policy, preflight_context(task, actor, 17))
    assert "action_match" not in row
    assert "scope_recall" not in row
    assert "scope_precision" not in row
    assert set(BASE_FEATURES).isdisjoint(
        {"action_match", "scope_recall", "scope_precision", "hard_deny"}
    )


def test_split_contracts_include_surface_request_and_compositional_holdout():
    result = validate_benchmark(generate_benchmark(17))
    assert result["status"] == "PASS"
    assert not any("leakage" in x for x in result["errors"])
    assert result["final_compositional_flags"]["requires_confirmation"] > 0
    assert result["final_compositional_flags"]["ambiguity_required"] > 0
