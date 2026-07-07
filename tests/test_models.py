from rtrace.actors import GenericSFTActor
from rtrace.data import generate_benchmark
from rtrace.features import frame_for
from rtrace.models import ReferenceBank, RiskCritic
from rtrace.policy import PolicyEngine


def test_reference_bank_and_critic_score():
    data = generate_benchmark(
        17,
        {
            "train": 60,
            "development": 0,
            "calibration": 30,
            "final_clean": 0,
            "final_hard": 0,
            "final_compositional": 0,
        },
    )
    policy = PolicyEngine()
    actor = GenericSFTActor()
    refs = ReferenceBank().fit(data["train"], actor, policy, 17)
    train, _ = frame_for(data["train"], actor, policy, 17, refs)
    cal, _ = frame_for(data["calibration"], actor, policy, 19, refs)
    critic = RiskCritic(use_reference=True).fit(train).calibrate(cal)
    scores = critic.score(cal)
    assert len(scores) == len(cal)
    assert ((scores >= 0) & (scores <= 1)).all()
