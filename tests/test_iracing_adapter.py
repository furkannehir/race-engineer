import asyncio
from collections.abc import Awaitable, Callable

from race_engineer.config import IracingTelemetryConfig
from race_engineer.core.contracts import TelemetryFrame
from race_engineer.telemetry.iracing import IracingRawSample, IracingTelemetryAdapter


class FakeSource:
    def __init__(self, samples: list[IracingRawSample], connects: bool = True) -> None:
        self._samples = samples
        self._connects = connects
        self.closed = False

    def connect(self) -> bool:
        return self._connects

    @property
    def connected(self) -> bool:
        return self._connects and bool(self._samples)

    def read(self) -> IracingRawSample:
        return self._samples.pop(0)

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
) -> IracingTelemetryAdapter:
    return IracingTelemetryAdapter(
        config=IracingTelemetryConfig(include_replay=include_replay),
        source_provider=provider,
        sleeper=sleeper,
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
