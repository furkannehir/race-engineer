import asyncio
from pathlib import Path

from race_engineer.core.enums import PolicyDecisionReason
from race_engineer.fixtures import load_fixture
from race_engineer.policy import DefaultRaceContextBuilder, StrictRulePolicy
from race_engineer.telemetry.iracing import IracingEventDeriver

ROOT = Path(__file__).parents[1]


def test_m2_policy_replay_reproduces_events_decisions_and_intents() -> None:
    fixture = load_fixture(ROOT / "fixtures" / "synthetic" / "m2_green_flag")
    deriver = IracingEventDeriver()
    builder = DefaultRaceContextBuilder()
    decisions = []
    policy = StrictRulePolicy(decision_sink=decisions.append)
    previous = None
    events = []
    intents = []

    for frame in fixture.frames:
        current_events = tuple(deriver.derive(previous, frame))
        events.extend(current_events)
        context = builder.update(frame, current_events)
        intents.extend(asyncio.run(policy.decide(context)))
        previous = frame

    assert tuple(events) == fixture.expected_events
    assert tuple(intents) == fixture.expected_intents
    assert [decision.reason for decision in decisions] == [
        PolicyDecisionReason.APPROVED,
        PolicyDecisionReason.DUPLICATE,
    ]
