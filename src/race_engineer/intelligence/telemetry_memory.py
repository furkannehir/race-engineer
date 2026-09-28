"""Bounded, replay-deterministic memory for generic normalized telemetry queries."""

from collections import deque
from collections.abc import Iterable
from datetime import timedelta
from statistics import fmean

from race_engineer.core.contracts import RaceContext
from race_engineer.core.intelligence import EvidenceItem, EvidenceQuery, SignalSelector


class TelemetryMemoryError(Exception):
    """Telemetry history is unavailable or violates its ordering boundary."""


def _unit(signal: str) -> str | None:
    if signal == "position":
        return "position"
    if signal in {"lap_number", "stint_lap"}:
        return "lap"
    suffixes = {
        "_mps": "m/s",
        "_l": "l",
        "_s": "s",
        "_m": "m",
    }
    return next((unit for suffix, unit in suffixes.items() if signal.endswith(suffix)), None)


def _signal(context: RaceContext, selector: SignalSelector) -> int | float | None:
    if selector.source == "player":
        value = getattr(context.frame.player, selector.signal, None)
    elif selector.source == "context":
        value = getattr(context, selector.signal, None)
    else:
        opponent = next(
            (
                opponent
                for opponent in context.frame.opponents
                if opponent.driver_id == selector.subject_id
            ),
            None,
        )
        if opponent is None:
            return None
        value = getattr(opponent, selector.signal, None)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return value


def _source_path(selector: SignalSelector) -> str:
    if selector.source != "opponent":
        return f"{selector.source}.{selector.signal}"
    return f"opponent:{selector.subject_id}.{selector.signal}"


class BoundedTelemetryMemory:
    """Store a bounded session window and answer generic numerical operations."""

    def __init__(
        self,
        *,
        history_s: float = 120.0,
        max_samples: int = 2400,
        evidence_freshness_s: float = 3.0,
    ) -> None:
        if history_s <= 0 or max_samples < 2 or evidence_freshness_s <= 0:
            raise ValueError("telemetry memory bounds must be positive")
        self._history_s = history_s
        self._evidence_freshness_s = evidence_freshness_s
        self._frames: deque[RaceContext] = deque(maxlen=max_samples)

    @property
    def sample_count(self) -> int:
        return len(self._frames)

    @property
    def session_id(self) -> str | None:
        return self._frames[-1].frame.session_id if self._frames else None

    def clear(self) -> None:
        self._frames.clear()

    def update(self, context: RaceContext) -> None:
        frame = context.frame
        if self._frames:
            previous = self._frames[-1].frame
            if (
                frame.session_id != previous.session_id
                or frame.session_time_s < previous.session_time_s
            ):
                self.clear()
            elif frame.sequence <= previous.sequence:
                raise TelemetryMemoryError("telemetry_sequence_not_increasing")
            elif frame.observed_at < previous.observed_at:
                raise TelemetryMemoryError("telemetry_observation_time_moved_backwards")
        self._frames.append(context)
        cutoff = frame.session_time_s - self._history_s
        while len(self._frames) > 1 and self._frames[0].frame.session_time_s < cutoff:
            self._frames.popleft()

    def _unknown(self, request: EvidenceQuery, current: RaceContext) -> EvidenceItem:
        frame = current.frame
        return EvidenceItem(
            evidence_id=request.query_id,
            session_id=frame.session_id,
            source_sequence=frame.sequence,
            observed_at=frame.observed_at,
            kind="unknown",
            subject=request.selector.subject_id or request.selector.source,
            metric=request.selector.signal,
            confidence=0.0,
            source_fields=(_source_path(request.selector),),
            valid_until=frame.observed_at + self._freshness_delta,
        )

    @property
    def _freshness_delta(self) -> timedelta:
        return timedelta(seconds=self._evidence_freshness_s)

    def _known(
        self,
        request: EvidenceQuery,
        current: RaceContext,
        value: int | float,
        *,
        coverage: float,
    ) -> EvidenceItem:
        frame = current.frame
        unit = _unit(request.selector.signal)
        if request.operation == "trend" and unit is not None:
            unit = f"{unit}/s"
        return EvidenceItem(
            evidence_id=request.query_id,
            session_id=frame.session_id,
            source_sequence=frame.sequence,
            observed_at=frame.observed_at,
            kind="measurement" if request.operation == "latest" else "derived",
            subject=request.selector.subject_id or request.selector.source,
            metric=request.selector.signal,
            value=value,
            unit=unit,
            confidence=coverage,
            source_fields=(_source_path(request.selector),),
            valid_until=frame.observed_at + self._freshness_delta,
        )

    def query(self, request: EvidenceQuery) -> EvidenceItem:
        if not self._frames:
            raise TelemetryMemoryError("telemetry_memory_empty")
        current = self._frames[-1]
        if request.operation == "latest":
            value = _signal(current, request.selector)
            return (
                self._unknown(request, current)
                if value is None
                else self._known(request, current, value, coverage=1.0)
            )

        assert request.window_s is not None
        cutoff = current.frame.session_time_s - request.window_s
        selected = tuple(
            context for context in self._frames if context.frame.session_time_s >= cutoff
        )
        if not selected or self._frames[0].frame.session_time_s > cutoff:
            return self._unknown(request, current)
        observed = tuple(
            (context.frame.session_time_s, value)
            for context in selected
            if (value := _signal(context, request.selector)) is not None
        )
        required = 2 if request.operation in {"delta", "trend"} else 1
        if len(observed) < required:
            return self._unknown(request, current)
        coverage = len(observed) / len(selected)
        values = tuple(value for _, value in observed)
        result: int | float | None
        if request.operation == "delta":
            result = values[-1] - values[0]
        elif request.operation == "mean":
            result = fmean(values)
        elif request.operation == "minimum":
            result = min(values)
        elif request.operation == "maximum":
            result = max(values)
        else:
            result = self._trend(observed)
        if result is None:
            return self._unknown(request, current)
        return self._known(request, current, result, coverage=coverage)

    @staticmethod
    def _trend(observed: Iterable[tuple[float, int | float]]) -> float | None:
        points = tuple(observed)
        times = tuple(point[0] for point in points)
        values = tuple(float(point[1]) for point in points)
        mean_time = fmean(times)
        mean_value = fmean(values)
        denominator = sum((time - mean_time) ** 2 for time in times)
        if denominator == 0:
            return None
        return (
            sum(
                (time - mean_time) * (value - mean_value)
                for time, value in zip(times, values, strict=True)
            )
            / denominator
        )

    def query_many(self, requests: Iterable[EvidenceQuery]) -> tuple[EvidenceItem, ...]:
        items = tuple(requests)
        if len({request.query_id for request in items}) != len(items):
            raise ValueError("evidence query IDs must be unique")
        return tuple(self.query(request) for request in items)
