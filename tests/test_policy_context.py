from datetime import UTC, datetime, timedelta

import pytest

from race_engineer.core.contracts import OpponentState, PlayerState, RaceEvent, TelemetryFrame
from race_engineer.core.enums import EventType
from race_engineer.policy import DefaultRaceContextBuilder

START = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)


def make_frame(
    sequence: int,
    *,
    lap: int,
    fuel_l: float,
    seconds: float = 0,
    in_pit_lane: bool = False,
    session_id: str = "session-1",
    opponents: tuple[OpponentState, ...] = (),
) -> TelemetryFrame:
    return TelemetryFrame(
        source="test",
        session_id=session_id,
        sequence=sequence,
        observed_at=START + timedelta(seconds=seconds),
        session_time_s=seconds,
        player=PlayerState(
            driver_id="player",
            lap_number=lap,
            fuel_l=fuel_l,
            in_pit_lane=in_pit_lane,
        ),
        opponents=opponents,
    )


def test_context_builder_derives_stint_fuel_and_battle_features() -> None:
    builder = DefaultRaceContextBuilder()
    opponents = (
        OpponentState(driver_id="ahead", gap_to_player_s=-1.2),
        OpponentState(driver_id="behind", gap_to_player_s=0.8),
    )

    first = builder.update(make_frame(1, lap=4, fuel_l=50, opponents=opponents), ())
    second = builder.update(make_frame(2, lap=5, fuel_l=47.5, seconds=60), ())
    third = builder.update(make_frame(3, lap=6, fuel_l=44, seconds=120), ())

    assert first.stint_lap == 1
    assert first.gap_ahead_s == pytest.approx(1.2)
    assert first.gap_behind_s == pytest.approx(0.8)
    assert first.battle_state == "sandwiched"
    assert first.fuel_trend_l_per_lap is None
    assert second.stint_lap == 2
    assert second.fuel_trend_l_per_lap == pytest.approx(2.5)
    assert third.stint_lap == 3
    assert third.fuel_trend_l_per_lap == pytest.approx(3.0)


def test_context_builder_resets_fuel_trend_after_refueling_and_stint_after_pit() -> None:
    builder = DefaultRaceContextBuilder()
    builder.update(make_frame(1, lap=4, fuel_l=50), ())
    builder.update(make_frame(2, lap=5, fuel_l=47, seconds=60), ())
    in_pit = builder.update(make_frame(3, lap=5, fuel_l=55, seconds=70, in_pit_lane=True), ())
    left_pit = builder.update(make_frame(4, lap=5, fuel_l=55, seconds=80), ())

    assert in_pit.fuel_trend_l_per_lap is None
    assert left_pit.stint_lap == 1


def test_context_builder_expires_history_and_rejects_cross_session_events() -> None:
    builder = DefaultRaceContextBuilder()
    frame = make_frame(1, lap=1, fuel_l=20)
    event = RaceEvent(
        event_id="event-1",
        session_id=frame.session_id,
        source_sequence=frame.sequence,
        occurred_at=frame.observed_at,
        event_type=EventType.CUSTOM,
        expires_at=frame.observed_at + timedelta(seconds=1),
    )
    assert builder.update(frame, (event,)).recent_events == (event,)
    expired = builder.update(make_frame(2, lap=1, fuel_l=20, seconds=2), ())
    assert expired.recent_events == ()

    other_event = event.model_copy(update={"session_id": "other-session"})
    with pytest.raises(ValueError, match="must match"):
        builder.update(make_frame(3, lap=1, fuel_l=20, seconds=3), (other_event,))
