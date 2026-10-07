from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from race_engineer.core.contracts import (
    OpponentState,
    PlayerState,
    RaceContext,
    TelemetryFrame,
)
from race_engineer.core.intelligence import CapabilityRequest
from race_engineer.fixtures import load_fixture
from race_engineer.intelligence.capabilities import (
    DeterministicRaceCapabilities,
    RaceCapabilityError,
)
from race_engineer.intelligence.telemetry_memory import BoundedTelemetryMemory
from race_engineer.policy import DefaultRaceContextBuilder

NOW = datetime(2026, 10, 1, 12, tzinfo=UTC)
ROOT = Path(__file__).parents[1]


def context(
    *,
    sequence: int = 42,
    session_time_s: float = 120,
    position: int | None = 5,
    opponent_positions: tuple[int | None, ...] = (1, 6, None),
    fuel_l: float | None = 20,
    fuel_burn_l_per_lap: float | None = 2,
    gap_ahead_s: float | None = None,
    gap_behind_s: float | None = None,
    opponent_gaps: tuple[float | None, ...] | None = None,
) -> RaceContext:
    return RaceContext(
        frame=TelemetryFrame(
            source="test",
            session_id="iracing:capabilities",
            sequence=sequence,
            observed_at=NOW + timedelta(seconds=session_time_s),
            session_time_s=session_time_s,
            player=PlayerState(
                driver_id="player",
                position=position,
                fuel_l=fuel_l,
            ),
            opponents=tuple(
                OpponentState(
                    driver_id=f"opponent-{index}",
                    position=value,
                    gap_to_player_s=(
                        opponent_gaps[index - 1]
                        if opponent_gaps is not None
                        else -gap_ahead_s
                        if index == 1 and gap_ahead_s is not None
                        else gap_behind_s
                        if index == 2 and gap_behind_s is not None
                        else None
                    ),
                )
                for index, value in enumerate(opponent_positions, start=1)
            ),
        ),
        fuel_trend_l_per_lap=fuel_burn_l_per_lap,
        gap_ahead_s=gap_ahead_s,
        gap_behind_s=gap_behind_s,
    )


def registry(*contexts: RaceContext) -> DeterministicRaceCapabilities:
    memory = BoundedTelemetryMemory()
    for current in contexts:
        memory.update(current)
    return DeterministicRaceCapabilities(memory)


def request(capability_id: str, request_id: str = "c1") -> CapabilityRequest:
    return CapabilityRequest(request_id=request_id, capability_id=capability_id)


def test_classification_capability_calculates_authoritative_relationship():
    capabilities = registry(context())

    result = capabilities.execute(request("current_classification"))
    values = {item.evidence_id: item.value for item in result.evidence}

    assert result.status == "available"
    assert values == {
        "c1:player_position": 5,
        "c1:last_position": 6,
        "c1:classified_cars": 3,
        "c1:is_last": False,
    }
    assert capabilities.evidence_ids(request("current_classification")) == tuple(values)


def test_classification_capability_is_explicitly_unavailable_without_position():
    capabilities = registry(context(position=None))
    descriptor = capabilities.catalog()[0]

    result = capabilities.execute(request("current_classification"))

    assert descriptor.available is False
    assert descriptor.unavailable_reason == "missing_player_position"
    assert result.status == "unavailable"
    assert result.unavailable_reason == "missing_player_position"
    assert all(item.kind == "unknown" for item in result.evidence)


def test_fuel_range_capability_uses_observed_burn_deterministically():
    capabilities = registry(context(fuel_l=18, fuel_burn_l_per_lap=2.4))

    result = capabilities.execute(request("fuel_range"))

    assert result.status == "available"
    assert result.evidence[0].metric == "fuel_laps_remaining"
    assert result.evidence[0].value == pytest.approx(7.5)
    assert result.evidence[0].unit == "laps"


def test_fuel_range_capability_reports_missing_burn_instead_of_guessing():
    capabilities = registry(context(fuel_burn_l_per_lap=None))
    descriptor = capabilities.catalog()[1]

    result = capabilities.execute(request("fuel_range"))

    assert descriptor.available is False
    assert descriptor.unavailable_reason == "missing_fuel_burn_per_lap"
    assert result.status == "unavailable"
    assert result.evidence[0].kind == "unknown"


def test_position_change_reports_deterministic_direction_and_magnitude():
    capabilities = registry(
        context(sequence=1, session_time_s=0, position=8),
        context(sequence=2, session_time_s=30, position=5),
    )

    result = capabilities.execute(request("position_change"))

    assert result.status == "available"
    assert [item.value for item in result.evidence] == ["gained", 3]
    assert result.evidence[1].unit == "positions"


def test_position_change_requires_a_complete_thirty_second_window():
    capabilities = registry(context(sequence=1, session_time_s=20, position=5))

    result = capabilities.execute(request("position_change"))

    assert result.status == "unavailable"
    assert result.unavailable_reason == "insufficient_position_history_30s"


def test_current_gap_capabilities_keep_ahead_and_behind_separate():
    capabilities = registry(context(gap_ahead_s=1.2, gap_behind_s=0.8))

    ahead = capabilities.execute(request("gap_ahead", "ahead"))
    behind = capabilities.execute(request("gap_behind", "behind"))

    assert ahead.evidence[0].value == pytest.approx(1.2)
    assert ahead.evidence[0].subject == "car_ahead"
    assert behind.evidence[0].value == pytest.approx(0.8)
    assert behind.evidence[0].subject == "car_behind"


def test_relative_pace_and_catch_time_use_bounded_gap_trends():
    capabilities = registry(
        context(
            sequence=1,
            session_time_s=0,
            gap_ahead_s=2.0,
            gap_behind_s=1.0,
        ),
        context(
            sequence=2,
            session_time_s=10,
            gap_ahead_s=1.0,
            gap_behind_s=0.5,
        ),
    )

    ahead_pace = capabilities.execute(request("relative_pace_ahead", "pace-ahead"))
    behind_pace = capabilities.execute(request("relative_pace_behind", "pace-behind"))
    ahead_catch = capabilities.execute(request("catch_time_ahead", "catch-ahead"))
    behind_catch = capabilities.execute(request("catch_time_behind", "catch-behind"))

    assert [item.value for item in ahead_pace.evidence] == [
        "closing",
        pytest.approx(0.1),
    ]
    assert [item.value for item in behind_pace.evidence] == [
        "closing",
        pytest.approx(0.05),
    ]
    assert ahead_catch.evidence[0].value == pytest.approx(10)
    assert behind_catch.evidence[0].value == pytest.approx(10)


def test_relative_pace_tracks_the_current_opponent_identity_across_the_window():
    capabilities = registry(
        context(
            sequence=1,
            session_time_s=0,
            opponent_positions=(4, 3),
            opponent_gaps=(-1.0, -3.0),
        ),
        context(
            sequence=2,
            session_time_s=10,
            opponent_positions=(3, 4),
            opponent_gaps=(-4.0, -2.0),
        ),
    )

    result = capabilities.execute(request("relative_pace_ahead"))

    assert result.status == "available"
    assert result.evidence[0].value == "closing"
    assert result.evidence[1].value == pytest.approx(0.1)
    assert result.evidence[1].source_fields == (
        "opponent:opponent-2.gap_to_player_s[10s]",
    )


def test_catch_time_is_unavailable_when_gap_is_not_closing():
    capabilities = registry(
        context(sequence=1, session_time_s=0, gap_ahead_s=1.0),
        context(sequence=2, session_time_s=10, gap_ahead_s=2.0),
    )

    result = capabilities.execute(request("catch_time_ahead"))

    assert result.status == "unavailable"
    assert result.unavailable_reason == "player_not_closing_car_ahead"


@pytest.mark.parametrize(
    ("capability_id", "reason"),
    (
        ("pit_loss_projection", "missing_pit_loss_model"),
        ("pit_stop_duration_projection", "missing_pit_service_model"),
        ("projected_rejoin_position", "missing_rejoin_projection_model"),
    ),
)
def test_strategy_projections_are_typed_but_never_fabricated(
    capability_id: str,
    reason: str,
):
    capabilities = registry(context())

    result = capabilities.execute(request(capability_id))

    assert result.status == "unavailable"
    assert result.unavailable_reason == reason
    assert all(item.kind == "unknown" for item in result.evidence)


def test_capability_catalog_contains_the_complete_int06_workload():
    capabilities = registry(context())

    assert [item.capability_id for item in capabilities.catalog()] == [
        "current_classification",
        "fuel_range",
        "position_change",
        "gap_ahead",
        "gap_behind",
        "relative_pace_ahead",
        "relative_pace_behind",
        "catch_time_ahead",
        "catch_time_behind",
        "pit_loss_projection",
        "pit_stop_duration_projection",
        "projected_rejoin_position",
    ]


def test_int06_replay_fixture_exercises_calculations_together():
    fixture = load_fixture(ROOT / "fixtures/synthetic/intelligence-capabilities")
    builder = DefaultRaceContextBuilder()
    memory = BoundedTelemetryMemory()
    for frame in fixture.frames:
        memory.update(builder.update(frame, ()))
    capabilities = DeterministicRaceCapabilities(memory)

    position = capabilities.execute(request("position_change", "position"))
    fuel = capabilities.execute(request("fuel_range", "fuel"))
    ahead_gap = capabilities.execute(request("gap_ahead", "ahead-gap"))
    ahead_pace = capabilities.execute(
        request("relative_pace_ahead", "ahead-pace")
    )
    catch = capabilities.execute(request("catch_time_ahead", "catch"))

    assert [item.value for item in position.evidence] == ["gained", 3]
    assert fuel.evidence[0].value == pytest.approx(12)
    assert ahead_gap.evidence[0].value == pytest.approx(1)
    assert [item.value for item in ahead_pace.evidence] == [
        "closing",
        pytest.approx(0.05),
    ]
    assert catch.evidence[0].value == pytest.approx(20)


def test_capability_registry_rejects_unknown_capabilities_and_arguments():
    capabilities = registry(context())

    with pytest.raises(RaceCapabilityError, match="unknown"):
        capabilities.execute(request("made_up"))
    with pytest.raises(RaceCapabilityError, match="arguments"):
        capabilities.execute(
            CapabilityRequest(
                request_id="c1",
                capability_id="fuel_range",
                arguments={"reserve_laps": 1},
            )
        )
