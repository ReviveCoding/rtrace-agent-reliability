from rtrace.data import generate_benchmark
from rtrace.schemas import ActionCandidate, Route
from rtrace.state import StateStore


def _tasks():
    return generate_benchmark(
        17,
        {
            "train": 24,
            "development": 1,
            "calibration": 1,
            "final_clean": 1,
            "final_hard": 1,
            "final_compositional": 1,
        },
    )["train"]


def test_executor_enforces_policy_even_when_router_is_bypassed():
    task = next(task for task in _tasks() if task.gold_action == "calendar.create")
    candidate = ActionCandidate(
        task_id=task.task_id,
        domain=task.domain,
        action=task.gold_action,
        args={**task.gold_args, "extra_permission": "admin"},
        idempotency_key="bypass-policy",
        source="test",
    )
    with StateStore(":memory:", state_seed=task.state_seed) as store:
        before = store.digest()
        result = store.execute(task, candidate, Route.ALLOW)
        assert not result.success
        assert result.message.startswith("policy_denied")
        assert before == store.digest()


def test_executor_requires_host_confirmation_before_irreversible_mutation():
    task = next(task for task in _tasks() if task.gold_action == "payments.execute")
    candidate = ActionCandidate(
        task_id=task.task_id,
        domain=task.domain,
        action=task.gold_action,
        args=dict(task.gold_args),
        idempotency_key="bypass-confirmation",
        source="test",
    )
    with StateStore(":memory:", state_seed=task.state_seed) as store:
        before = store.digest()
        result = store.execute(
            task,
            candidate,
            Route.ALLOW,
            preflight={"executed": True, "success": True, "target_exists": True},
        )
        assert not result.success
        assert result.message == "confirmation_required"
        assert before == store.digest()
