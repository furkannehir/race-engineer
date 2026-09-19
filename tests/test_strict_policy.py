import asyncio
from datetime import UTC, datetime, timedelta

from race_engineer.core.contracts import PlayerState, RaceContext, RaceEvent, TelemetryFrame
from race_engineer.core.enums import (
    EventType,
    InterruptionPolicy,
    PolicyDecisionOutcome,
    PolicyDecisionReason,
    Urgency,
)
from race_engineer.policy import StrictRulePolicy

START = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)


def make_frame(sequence: int, seconds: float = 0, session_id: str = "session-1") -> TelemetryFrame:
    return TelemetryFrame(
        source="test",
        session_id=session_id,
        sequence=sequence,
        observed_at=START + timedelta(seconds=seconds),
        session_time_s=seconds,
        player=PlayerState(driver_id="player", position=4),
    )


def make_event(
    frame: TelemetryFrame,
    event_type: EventType,
    facts: dict[str, object],
    *,
    event_id: str,
    urgency: Urgency = Urgency.ROUTINE,
    expires_in_s: float = 10,
) -> RaceEvent:
    return RaceEvent(
        event_id=event_id,
        session_id=frame.session_id,
        source_sequence=frame.sequence,
        occurred_at=frame.observed_at,
        event_type=event_type,
        facts=facts,  # type: ignore[arg-type]
        urgency=urgency,
        expires_at=frame.observed_at + timedelta(seconds=expires_in_s),
    )


def decide(policy: StrictRulePolicy, frame: TelemetryFrame, *events: RaceEvent):
    return asyncio.run(policy.decide(RaceContext(frame=frame, recent_events=events)))


def test_policy_deduplicates_equivalent_phase_and_flag_calls() -> None:
    frame = make_frame(1)
    phase = make_event(
        frame,
        EventType.SESSION_PHASE_CHANGED,
        {"from": "formation", "to": "green"},
        event_id="phase-green",
        urgency=Urgency.IMPORTANT,
    )
    flag = make_event(
        frame,
        EventType.FLAG_CHANGED,
        {"added": ["green"], "removed": []},
        event_id="flag-green",
        urgency=Urgency.IMPORTANT,
    )
    decisions = []
    policy = StrictRulePolicy(decision_sink=decisions.append)

    intents = decide(policy, frame, phase, flag)

    assert len(intents) == 1
    assert intents[0].facts == {"phase": "green", "previous_phase": "formation"}
    assert [decision.outcome for decision in decisions] == [
        PolicyDecisionOutcome.APPROVED,
        PolicyDecisionOutcome.SUPPRESSED,
    ]
    assert decisions[1].reason is PolicyDecisionReason.DUPLICATE


def test_policy_applies_routine_cooldown() -> None:
    decisions = []
    policy = StrictRulePolicy(decision_sink=decisions.append)
    first_frame = make_frame(1)
    first = make_event(
        first_frame,
        EventType.POSITION_CHANGED,
        {"from": 5, "to": 4},
        event_id="position-4",
    )
    second_frame = make_frame(2, seconds=1)
    second = make_event(
        second_frame,
        EventType.POSITION_CHANGED,
        {"from": 4, "to": 3},
        event_id="position-3",
    )
    third_frame = make_frame(3, seconds=16)
    third = make_event(
        third_frame,
        EventType.POSITION_CHANGED,
        {"from": 3, "to": 2},
        event_id="position-2",
    )

    assert len(decide(policy, first_frame, first)) == 1
    assert decide(policy, second_frame, second) == []
    assert len(decide(policy, third_frame, third)) == 1
    assert [decision.reason for decision in decisions] == [
        PolicyDecisionReason.APPROVED,
        PolicyDecisionReason.COOLDOWN,
        PolicyDecisionReason.APPROVED,
    ]


def test_critical_call_supersedes_routine_call_and_has_fixed_template() -> None:
    frame = make_frame(1)
    position = make_event(
        frame,
        EventType.POSITION_CHANGED,
        {"from": 5, "to": 4},
        event_id="position-4",
    )
    yellow = make_event(
        frame,
        EventType.FLAG_CHANGED,
        {"added": ["yellow"], "removed": []},
        event_id="yellow",
        urgency=Urgency.CRITICAL,
    )
    decisions = []
    policy = StrictRulePolicy(decision_sink=decisions.append)

    intents = decide(policy, frame, position, yellow)

    assert len(intents) == 1
    assert intents[0].facts == {"flag": "yellow", "state": "shown"}
    assert intents[0].priority == 95
    assert intents[0].critical_template == "Yellow flag"
    assert intents[0].interruption_policy is InterruptionPolicy.INTERRUPT_ANY
    assert [decision.reason for decision in decisions] == [
        PolicyDecisionReason.APPROVED,
        PolicyDecisionReason.SUPERSEDED,
    ]


def test_disabled_pit_rule_records_suppression_reason() -> None:
    frame = make_frame(1)
    pit = make_event(
        frame,
        EventType.PIT_STATE_CHANGED,
        {"in_pit_lane": True},
        event_id="pit-entry",
    )
    decisions = []
    policy = StrictRulePolicy(decision_sink=decisions.append)

    assert decide(policy, frame, pit) == []
    assert len(decisions) == 1
    assert decisions[0].reason is PolicyDecisionReason.DISABLED


def test_expired_candidate_is_suppressed() -> None:
    frame = make_frame(2, seconds=2)
    event = RaceEvent(
        event_id="expired-position",
        session_id=frame.session_id,
        source_sequence=frame.sequence,
        occurred_at=START,
        event_type=EventType.POSITION_CHANGED,
        facts={"from": 5, "to": 4},
        expires_at=START + timedelta(seconds=1),
    )
    decisions = []
    policy = StrictRulePolicy(decision_sink=decisions.append)

    assert decide(policy, frame, event) == []
    assert decisions[0].reason is PolicyDecisionReason.EXPIRED


def test_critical_calls_bypass_cooldown() -> None:
    decisions = []
    policy = StrictRulePolicy(decision_sink=decisions.append)
    first_frame = make_frame(1)
    second_frame = make_frame(2, seconds=1)
    first = make_event(
        first_frame,
        EventType.FLAG_CHANGED,
        {"added": ["yellow"], "removed": []},
        event_id="yellow-1",
        urgency=Urgency.CRITICAL,
    )
    second = make_event(
        second_frame,
        EventType.FLAG_CHANGED,
        {"added": ["yellow"], "removed": []},
        event_id="yellow-2",
        urgency=Urgency.CRITICAL,
    )

    assert len(decide(policy, first_frame, first)) == 1
    assert len(decide(policy, second_frame, second)) == 1
    assert all(decision.reason is PolicyDecisionReason.APPROVED for decision in decisions)
