"""Text-first exercise of the evidence-driven intelligence path over a replay fixture."""

import logging
from collections import deque
from datetime import UTC, datetime
from pathlib import Path

from race_engineer.config import PolicyContextConfig, load_config
from race_engineer.core.contracts import RaceContext, RaceEvent
from race_engineer.core.conversation import RadioLanguage
from race_engineer.core.intelligence import DriverTurn
from race_engineer.fixtures import FixtureBundle, load_fixture
from race_engineer.intelligence.context_engine import ContextEngineerError
from race_engineer.intelligence.factory import live_intelligence
from race_engineer.intelligence.grounding import GroundingError
from race_engineer.intelligence.local_model import LocalIntelligenceError
from race_engineer.intelligence.orchestrator import IntelligenceBoundaryError
from race_engineer.intelligence.telemetry_memory import (
    BoundedTelemetryMemory,
    TelemetryMemoryError,
)
from race_engineer.observability import configure_logging
from race_engineer.policy import DefaultRaceContextBuilder

_LOGGER = logging.getLogger(__name__)


def _failure_hint(reason: str) -> str | None:
    if reason.endswith("model_unreachable"):
        return "The local Qwen server is not reachable. Start it, then try again."
    if reason.endswith("model_timeout"):
        return (
            "The local Qwen model timed out. It may still be loading or busy; "
            "check the server log."
        )
    if "model_http_" in reason:
        return (
            "The Qwen server is reachable but rejected or failed the structured response. "
            "Check the server log, then retry."
        )
    return None


def _contexts(
    fixture: FixtureBundle,
    policy_config: PolicyContextConfig,
) -> tuple[RaceContext, ...]:
    if not fixture.frames:
        raise ValueError("intelligence replay requires at least one frame")
    events: dict[tuple[str, int], list[RaceEvent]] = {}
    for event in fixture.expected_events:
        events.setdefault((event.session_id, event.source_sequence), []).append(event)
    builder = DefaultRaceContextBuilder(policy_config)
    return tuple(
        builder.update(
            frame,
            events.get((frame.session_id, frame.sequence), ()),
        )
        for frame in fixture.frames
    )


def _selected_index(frame_index: int, frame_count: int) -> int:
    selected = frame_count - 1 if frame_index == -1 else frame_index
    if not 0 <= selected < frame_count:
        raise ValueError(f"frame index must be between 0 and {frame_count - 1}")
    return selected


async def intelligence_replay(
    config_path: Path,
    directory: Path,
    *,
    question: str,
    frame_index: int = -1,
    reply_language: RadioLanguage | None = None,
    json_output: bool = False,
) -> int:
    """Run one typed question through the real Context/Core/Qwen/grounding path."""

    config = load_config(config_path)
    configure_logging(config.logging)
    fixture = load_fixture(directory)
    contexts = _contexts(fixture, config.policy.context)
    selected = _selected_index(frame_index, len(contexts))
    memory = BoundedTelemetryMemory()
    selected_session = contexts[selected].frame.session_id
    for context in contexts[: selected + 1]:
        memory.update(context)
    current = memory.latest_context()
    if current.frame.session_id != selected_session:
        raise ValueError("selected fixture frame is not the current replay session")

    engineer = live_intelligence(config.conversation, memory)
    language = reply_language or config.conversation.default_language
    recent_dialogue: deque[str] = deque(maxlen=config.conversation.history_turns * 2)
    turn = DriverTurn(
        turn_id=(f"replay:{current.frame.session_id}:{current.frame.sequence}:driver:1"),
        transcript=question.strip(),
        received_at=datetime.now(UTC),
        session_id=current.frame.session_id,
        generation=0,
        reply_language=language,
        recent_dialogue=tuple(recent_dialogue),
    )
    try:
        prepared = await engineer.prepare(turn)
        grounded = await engineer.ground(prepared)
    except (
        ContextEngineerError,
        GroundingError,
        IntelligenceBoundaryError,
        LocalIntelligenceError,
        TelemetryMemoryError,
    ) as error:
        _LOGGER.warning(
            "intelligence replay failed",
            extra={"event": "intelligence_replay_failed", "reason": str(error)},
        )
        print(f"The local intelligence pipeline could not process this question safely ({error}).")
        hint = _failure_hint(str(error))
        if hint is not None:
            print(hint)
        return 1

    state = {
        "fixture_id": fixture.manifest.fixture_id,
        "frame_index": selected,
        "frame_count": len(contexts),
        "session_id": current.frame.session_id,
        "source_sequence": grounded.source_sequence,
        "session_time_s": current.frame.session_time_s,
    }
    if json_output:
        import json

        print(
            json.dumps(
                {
                    "replay": state,
                    "context": prepared.context.model_dump(mode="json"),
                    "brief": prepared.brief.model_dump(mode="json"),
                    "generated": prepared.response.model_dump(mode="json"),
                    "grounded": grounded.model_dump(mode="json"),
                },
                indent=2,
                ensure_ascii=False,
                sort_keys=True,
            )
        )
    else:
        print(
            f"INTELLIGENCE REPLAY frame {selected}/{len(contexts) - 1}, "
            f"session={current.frame.session_id}, sequence={current.frame.sequence}, "
            f"session_time={current.frame.session_time_s:.1f}s"
        )
        print(
            "Engineer: [silence]"
            if grounded.text is None
            else f"Engineer ({grounded.language}): {grounded.text}"
        )
    return 0
