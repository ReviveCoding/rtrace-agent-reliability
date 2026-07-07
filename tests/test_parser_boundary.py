from __future__ import annotations

from rtrace.actors import GenericSFTActor
from rtrace.data import generate_benchmark, validate_benchmark
from rtrace.parser import parse_request


def _data():
    return generate_benchmark(
        17,
        {
            "train": 24,
            "development": 12,
            "calibration": 12,
            "final_clean": 12,
            "final_hard": 12,
            "final_compositional": 24,
        },
    )


def test_actor_uses_text_parser_not_gold_or_agent_payload():
    task = _data()["train"][0]
    actor = GenericSFTActor()
    baseline = actor.propose(task, 123)
    changed_gold = task.model_copy(
        update={
            "gold_action": "payments.execute",
            "gold_args": {"intent_id": "intent_999"},
            "required_scope": {"intent_id": "intent_999"},
        }
    )
    changed = actor.propose(changed_gold, 123)
    assert baseline.action == changed.action
    assert baseline.args == changed.args
    assert "requested_action" not in task.agent_context
    assert "requested_args" not in task.agent_context


def test_text_parser_covers_all_local_action_families():
    data = _data()
    for split_tasks in data.values():
        for task in split_tasks:
            parsed = parse_request(task.user_request)
            assert parsed.action == task.gold_action
            for key, value in task.required_scope.items():
                assert parsed.args.get(key) == value


def test_compositional_split_contains_actual_new_control_combinations():
    data = _data()
    result = validate_benchmark(data)
    assert result["status"] == "PASS"
    final = data["final_compositional"]
    assert any(task.requires_confirmation for task in final)
    assert any(task.ambiguity_required for task in final)
    train_compositions = {task.composition_id for task in data["train"]}
    assert not (train_compositions & {task.composition_id for task in final})
