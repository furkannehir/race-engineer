import asyncio
import logging
from collections.abc import Awaitable, Callable

import pytest

from race_engineer.config import IracingTelemetryConfig
from race_engineer.core.contracts import TelemetryFrame
from race_engineer.telemetry.iracing import IracingRawSample, IracingTelemetryAdapter
from race_engineer.telemetry.iracing.source import IracingReadMetrics, IracingReadResult

ZERO_METRICS = IracingReadMetrics(
    buffer_wait_ms=0.0,
    variable_read_ms=0.0,
    metadata_refresh_ms=0.0,
    total_ms=0.0,
    metadata_refreshed=False,
)


class ManualClock:
    def __init__(self) -> None:
        self.value = 0.0
        self.sleep_delays: list[float] = []

    def __call__(self) -> float:
        return self.value

    def advance(self, delay: float) -> None:
        self.value += delay

    async def sleep(self, delay: float) -> None:
        self.sleep_delays.append(delay)
        self.advance(delay)


class FakeSource:
    def __init__(
        self,
        samples: list[IracingRawSample],
        connects: bool = True,
        *,
        clock: ManualClock | None = None,
        read_latency_s: float = 0.0,
        metrics: IracingReadMetrics = ZERO_METRICS,
    ) -> None:
        self._samples = samples
        self._connects = connects
        self._clock = clock
        self._read_latency_s = read_latency_s
        self._metrics = metrics
        self.read_started_at: list[float] = []
        self.closed = False

    def connect(self) -> bool:
        return self._connects

    @property
    def connected(self) -> bool:
        return self._connects and bool(self._samples)

    def read(self) -> IracingReadResult:
        if self._clock is not None:
            self.read_started_at.append(self._clock())
            self._clock.advance(self._read_latency_s)
        return IracingReadResult(sample=self._samples.pop(0), metrics=self._metrics)

    def close(self) -> None:
        self.closed = True


async def no_sleep(delay: float) -> None:
    del delay


def collect_frames(
    adapter: IracingTelemetryAdapter,
    count: int,
) -> tuple[TelemetryFrame, ...]:
    async def collect() -> tuple[TelemetryFrame, ...]:
        stream = adapter.stream()
        frames: list[TelemetryFrame] = []
        try:
            async for frame in stream:
                frames.append(frame)
                if len(frames) == count:
                    break
        finally:
            await stream.aclose()  # type: ignore[attr-defined]
        return tuple(frames)

    return asyncio.run(collect())


def adapter_for(
    provider: Callable[[], FakeSource],
    *,
    include_replay: bool = False,
    sleeper: Callable[[float], Awaitable[None]] = no_sleep,
    monotonic: Callable[[], float] | None = None,
    slow_read_warning_s: float = 0.05,
    sample_gap_warning_s: float = 0.25,
) -> IracingTelemetryAdapter:
    config = IracingTelemetryConfig(
        include_replay=include_replay,
        slow_read_warning_s=slow_read_warning_s,
        sample_gap_warning_s=sample_gap_warning_s,
    )
    if monotonic is None:
        return IracingTelemetryAdapter(
            config=config,
            source_provider=provider,
            sleeper=sleeper,
        )
    return IracingTelemetryAdapter(
        config=config,
        source_provider=provider,
        sleeper=sleeper,
        monotonic=monotonic,
    )


def test_adapter_drops_duplicate_ticks_and_closes_source(
    iracing_samples: tuple[IracingRawSample, IracingRawSample],
) -> None:
    first, second = iracing_samples
    duplicate = IracingRawSample.model_validate(
        {**first.model_dump(), "observed_at": "2026-08-30T12:00:00.050000Z"}
    )
    source = FakeSource([first, duplicate, second])
    frames = collect_frames(adapter_for(lambda: source), 2)
    assert [frame.sequence for frame in frames] == [100, 101]
    assert source.closed is True


def test_adapter_reconnects_after_unavailable_source(
    iracing_samples: tuple[IracingRawSample, IracingRawSample],
) -> None:
    unavailable = FakeSource([], connects=False)
    connected = FakeSource([iracing_samples[0]])
    sources = iter((unavailable, connected))
    frames = collect_frames(adapter_for(lambda: next(sources)), 1)
    assert frames[0].sequence == 100
    assert unavailable.closed is True
    assert connected.closed is True


def test_adapter_suppresses_replay_unless_enabled(
    iracing_samples: tuple[IracingRawSample, IracingRawSample],
) -> None:
    first, second = iracing_samples
    replay = IracingRawSample.model_validate({**first.model_dump(), "is_replay_playing": True})
    source = FakeSource([replay, second])
    frames = collect_frames(adapter_for(lambda: source), 1)
    assert [frame.sequence for frame in frames] == [101]

    replay_source = FakeSource([replay])
    included = collect_frames(adapter_for(lambda: replay_source, include_replay=True), 1)
    assert included[0].is_replay is True


def test_adapter_uses_deadline_schedule_instead_of_sleeping_after_reads(
    iracing_samples: tuple[IracingRawSample, IracingRawSample],
) -> None:
    clock = ManualClock()
    source = FakeSource(
        list(iracing_samples),
        clock=clock,
        read_latency_s=0.017,
    )
    frames = collect_frames(
        adapter_for(
            lambda: source,
            sleeper=clock.sleep,
            monotonic=clock,
        ),
        2,
    )

    assert len(frames) == 2
    assert source.read_started_at == pytest.approx([0.0, 0.1])
    assert clock.sleep_delays == pytest.approx([0.083])


def test_adapter_reports_overruns_and_sample_gaps(
    iracing_samples: tuple[IracingRawSample, IracingRawSample],
    caplog: pytest.LogCaptureFixture,
) -> None:
    clock = ManualClock()
    metrics = IracingReadMetrics(
        buffer_wait_ms=1.0,
        variable_read_ms=2.0,
        metadata_refresh_ms=247.0,
        total_ms=250.0,
        metadata_refreshed=True,
    )
    source = FakeSource(
        list(iracing_samples),
        clock=clock,
        read_latency_s=0.25,
        metrics=metrics,
    )

    with caplog.at_level(logging.WARNING):
        frames = collect_frames(
            adapter_for(
                lambda: source,
                sleeper=clock.sleep,
                monotonic=clock,
                sample_gap_warning_s=0.2,
            ),
            2,
        )

    assert len(frames) == 2
    assert source.read_started_at == pytest.approx([0.0, 0.25])
    assert clock.sleep_delays == []
    events = [getattr(record, "event", None) for record in caplog.records]
    assert "sampling_deadline_missed" in events
    assert "telemetry_read_timing" in events
    assert "telemetry_sample_gap" in events
