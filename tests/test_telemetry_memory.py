from datetime import UTC, datetime, timedelta

import pytest

from race_engineer.core.contracts import (
    OpponentState,
    PlayerState,
    RaceContext,
    TelemetryFrame,
)
from race_engineer.core.intelligence import EvidenceQuery, SignalSelector
from race_engineer.intelligence.telemetry_memory import (
    BoundedTelemetryMemory,
    TelemetryMemoryError,
)

NOW = datetime(2026, 9, 28, 12, tzinfo=UTC)


def context(
    sequence: int,
    session_time_s: float,
    *,
    session_id: str = "iracing:1",
    position: int | None = 6,
    speed_mps: float | None = 20.0,
    fuel_l: float | None = 50.0,
    opponent_gap_s: float | None = -1.2,
) -> RaceContext:
    return RaceContext(
        frame=TelemetryFrame(
            source="test",
            session_id=session_id,
            sequence=sequence,
            observed_at=NOW + timedelta(seconds=session_time_s),
            session_time_s=session_time_s,
            player=PlayerState(
                driver_id="player",
                position=position,
                speed_mps=speed_mps,
                fuel_l=fuel_l,
            ),
            opponents=(
                OpponentState(
                    driver_id="car-ahead",
                    position=5,
                    gap_to_player_s=opponent_gap_s,
                ),
            ),
            capabilities=("fuel", "opponents", "position", "speed"),
        ),
        gap_ahead_s=abs(opponent_gap_s) if opponent_gap_s is not None else None,
    )


def query(
    query_id: str,
    signal: str,
    operation: str,
    *,
    source: str = "player",
    subject_id: str | None = None,
    window_s: float | None = None,
) -> EvidenceQuery:
    return EvidenceQuery.model_validate(
        {
            "query_id": query_id,
            "selector": {
                "source": source,
                "signal": signal,
                "subject_id": subject_id,
            },
            "operation": operation,
            "window_s": window_s,
        }
    )


def populated_memory() -> BoundedTelemetryMemory:
    memory = BoundedTelemetryMemory(history_s=10)
    memory.update(context(1, 0, speed_mps=10, fuel_l=50))
    memory.update(context(2, 1, speed_mps=20, fuel_l=49))
    memory.update(context(3, 2, speed_mps=30, fuel_l=48))
    return memory


def test_latest_query_reads_any_normalized_numeric_signal():
    memory = populated_memory()

    position = memory.query(query("position-now", "position", "latest"))
    gap = memory.query(query("gap-now", "gap_ahead_s", "latest", source="context"))
    opponent = memory.query(
        query(
            "opponent-gap",
            "gap_to_player_s",
            "latest",
            source="opponent",
            subject_id="car-ahead",
        )
    )

    assert (position.value, position.unit, position.kind) == (6, "position", "measurement")
    assert (gap.value, gap.unit) == (1.2, "s")
    assert opponent.value == -1.2
    assert opponent.source_fields == ("opponent:car-ahead.gap_to_player_s",)


def test_window_operations_are_generic_and_replay_deterministic():
    memory = populated_memory()

    delta = memory.query(query("fuel-change", "fuel_l", "delta", window_s=2))
    mean = memory.query(query("mean-speed", "speed_mps", "mean", window_s=2))
    trend = memory.query(query("speed-trend", "speed_mps", "trend", window_s=2))

    assert delta.value == -2
    assert mean.value == 20
    assert trend.value == 10
    assert all(item.kind == "derived" for item in (delta, mean, trend))
    assert all(item.source_sequence == 3 for item in (delta, mean, trend))
    assert all(item.confidence == 1 for item in (delta, mean, trend))


def test_missing_signal_or_incomplete_window_returns_explicit_unknown():
    memory = BoundedTelemetryMemory(history_s=10)
    memory.update(context(1, 10, position=None))

    missing = memory.query(query("position-now", "position", "latest"))
    incomplete = memory.query(query("speed-change", "speed_mps", "delta", window_s=5))

    assert missing.kind == "unknown" and missing.value is None
    assert incomplete.kind == "unknown" and incomplete.confidence == 0


def test_memory_resets_on_session_change_or_session_time_rewind():
    memory = populated_memory()
    memory.update(context(4, 3, session_id="iracing:2"))
    assert memory.session_id == "iracing:2" and memory.sample_count == 1

    memory.update(context(5, 1, session_id="iracing:2"))
    assert memory.session_id == "iracing:2" and memory.sample_count == 1


def test_memory_rejects_duplicate_sequences_and_query_ids():
    memory = BoundedTelemetryMemory()
    memory.update(context(1, 0))
    with pytest.raises(TelemetryMemoryError, match="sequence"):
        memory.update(context(1, 1))
    with pytest.raises(ValueError, match="query IDs"):
        memory.query_many(
            (
                query("same", "position", "latest"),
                query("same", "speed_mps", "latest"),
            )
        )


def test_selector_requires_an_opponent_identity_only_for_opponents():
    with pytest.raises(ValueError, match="opponent selectors"):
        SignalSelector(source="opponent", signal="position")
    with pytest.raises(ValueError, match="opponent selectors"):
        SignalSelector(source="player", signal="position", subject_id="someone")
