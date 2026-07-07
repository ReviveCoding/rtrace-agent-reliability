from rtrace.actors import GenericSFTActor
from rtrace.data import generate_benchmark
from rtrace.policy import PolicyEngine
from rtrace.sasc import StructuredActionScopeCorrection
from rtrace.schemas import ActionCandidate


def test_sasc_recovers_from_candidate_pool_and_prunes_excess():
    task = generate_benchmark(
        17,
        {
            "train": 3,
            "development": 0,
            "calibration": 0,
            "final_clean": 0,
            "final_hard": 0,
            "final_compositional": 0,
        },
    )["train"][0]
    c = ActionCandidate(
        task_id=task.task_id,
        domain=task.domain,
        action=task.gold_action,
        args={
            "title": task.required_scope.get("title", "x"),
            "attendees": ["external@example.test"],
            "extra_permission": "admin",
        },
        idempotency_key="x",
        source="test",
    )
    pool = GenericSFTActor().propose_pool(task, 17)
    fixed = StructuredActionScopeCorrection().correct(task, c, pool, PolicyEngine())
    assert "extra_permission" not in fixed.args
    assert fixed.args["attendees"] == ["internal@example.test"]
    for key in PolicyEngine().required_fields(task.gold_action):
        assert key in fixed.args


def test_sasc_does_not_read_evaluator_gold_values_when_pool_has_no_value():
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
    c = ActionCandidate(
        task_id=task.task_id,
        domain=task.domain,
        action=task.gold_action,
        args={"title": "Only title"},
        idempotency_key="x",
        source="test",
    )
    fixed = StructuredActionScopeCorrection().correct(task, c, [c], PolicyEngine())
    assert "start" not in fixed.args
