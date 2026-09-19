from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from race_engineer.core.contracts import (
    PlayerState,
    RaceContext,
    RaceEvent,
    TelemetryFrame,
    canonical_json,
)
from race_engineer.core.enums import EventType


def make_frame(*, session_id: str = "session-1") -> TelemetryFrame:
    return TelemetryFrame(
        source="test",
        session_id=session_id,
        sequence=1,
        observed_at=datetime(2026, 1, 1, tzinfo=UTC),
        session_time_s=1.0,
        player=PlayerState(driver_id="driver-1", lap_number=1),
        capabilities=("fuel", "lap"),
    )


def test_contract_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        TelemetryFrame.model_validate(
            {
                **make_frame().model_dump(),
                "simulator_private_value": 42,
            }
        )


def test_contract_requires_aware_timestamp() -> None:
    with pytest.raises(ValidationError):
        TelemetryFrame(
            source="test",
            session_id="session-1",
            sequence=1,
            observed_at=datetime(2026, 1, 1),
            session_time_s=1.0,
            player=PlayerState(driver_id="driver-1", lap_number=1),
        )


def test_capabilities_are_sorted_and_unique() -> None:
    data = make_frame().model_dump()
    data["capabilities"] = ("lap", "fuel")
    with pytest.raises(ValidationError):
        TelemetryFrame.model_validate(data)


def test_context_rejects_events_from_another_session() -> None:
    frame = make_frame()
    event = RaceEvent(
        event_id="event-1",
        session_id="other-session",
        source_sequence=1,
        occurred_at=frame.observed_at,
        event_type=EventType.CUSTOM,
        expires_at=frame.observed_at + timedelta(seconds=1),
    )
    with pytest.raises(ValidationError):
        RaceContext(frame=frame, recent_events=(event,))


def test_canonical_json_is_stable_and_sorted() -> None:
    frame = make_frame()
    first = canonical_json(frame)
    second = canonical_json(TelemetryFrame.model_validate_json(first))
    assert first == second
    assert first.index('"capabilities"') < first.index('"flags"')
