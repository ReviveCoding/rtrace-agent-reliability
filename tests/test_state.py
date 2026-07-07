from rtrace.data import generate_benchmark
from rtrace.schemas import ActionCandidate, Route
from rtrace.state import StateStore


def test_block_does_not_mutate_state():
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
    candidate = ActionCandidate(
        task_id=task.task_id,
        domain=task.domain,
        action=task.gold_action,
        args=task.gold_args,
        idempotency_key="x",
        source="test",
    )
    with StateStore(":memory:", state_seed=task.state_seed) as store:
        before = store.digest()
        result = store.execute(task, candidate, Route.BLOCK)
        assert before == store.digest()
        assert not result.success


def test_idempotency():
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
    candidate = ActionCandidate(
        task_id=task.task_id,
        domain=task.domain,
        action=task.gold_action,
        args=task.gold_args,
        idempotency_key="same",
        source="test",
    )
    with StateStore(":memory:", state_seed=task.state_seed) as store:
        first = store.execute(task, candidate, Route.ALLOW)
        second = store.execute(task, candidate, Route.ALLOW)
        assert first.success and second.message == "idempotent_replay"
