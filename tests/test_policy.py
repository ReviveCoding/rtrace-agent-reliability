from rtrace.actors import GenericSFTActor
from rtrace.data import generate_benchmark
from rtrace.policy import PolicyEngine


def test_policy_detects_forbidden_or_risk():
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
    candidate = GenericSFTActor().propose(task, 999)
    result = PolicyEngine().assess(task, candidate)
    assert 0.0 <= result.policy_prior <= 1.0


def test_high_impact_confirmation_requires_host_verified_context_not_actor_claim():
    tasks = generate_benchmark(
        17,
        {
            "train": 12,
            "development": 0,
            "calibration": 0,
            "final_clean": 0,
            "final_hard": 0,
            "final_compositional": 0,
        },
    )["train"]
    task = next(t for t in tasks if t.requires_confirmation)
    candidate = GenericSFTActor().propose(task, 17)
    candidate.args["confirmed"] = True
    runtime = PolicyEngine().runtime_assess(task, candidate)
    assert "confirmation_required" in runtime.reasons
