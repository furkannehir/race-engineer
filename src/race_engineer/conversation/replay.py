"""Step a fixture with an explicit replay clock; never present it as live data."""

from race_engineer.config import PolicyContextConfig
from race_engineer.core.contracts import RaceContext, RaceEvent
from race_engineer.core.conversation import RaceSnapshot
from race_engineer.fixtures import FixtureBundle
from race_engineer.policy import DefaultRaceContextBuilder


class ReplayRaceState:
    def __init__(self, fixture: FixtureBundle, config: PolicyContextConfig | None = None) -> None:
        if not fixture.frames:
            raise ValueError("conversation replay requires at least one frame")
        events: dict[tuple[str, int], list[RaceEvent]] = {}
        for event in fixture.expected_events:
            events.setdefault((event.session_id, event.source_sequence), []).append(event)
        builder = DefaultRaceContextBuilder(config)
        contexts: list[RaceContext] = []
        previous = None
        seen_sessions: set[str] = set()
        for frame in fixture.frames:
            if previous is not None and frame.session_id == previous.session_id:
                if (
                    frame.sequence <= previous.sequence
                    or frame.session_time_s < previous.session_time_s
                    or frame.observed_at < previous.observed_at
                ):
                    raise ValueError("conversation replay frames must be ordered within a session")
            elif frame.session_id in seen_sessions:
                raise ValueError("conversation replay cannot reenter an earlier session")
            seen_sessions.add(frame.session_id)
            contexts.append(
                builder.update(frame, events.get((frame.session_id, frame.sequence), ()))
            )
            previous = frame
        self._contexts = tuple(contexts)
        self._index = 0

    @property
    def index(self) -> int:
        return self._index

    @property
    def frame_count(self) -> int:
        return len(self._contexts)

    def seek(self, index: int) -> None:
        if not 0 <= index < len(self._contexts):
            raise ValueError(f"frame index must be between 0 and {len(self._contexts) - 1}")
        self._index = index

    def snapshot(self) -> RaceSnapshot:
        context = self._contexts[self._index]
        return RaceSnapshot(context=context, as_of=context.frame.observed_at, mode="replay")
