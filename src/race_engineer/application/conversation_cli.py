"""Text-first local conversation over an explicitly paused replay."""

import asyncio
from pathlib import Path

from race_engineer.config import load_config
from race_engineer.conversation.factory import conversation_planner
from race_engineer.conversation.replay import ReplayRaceState
from race_engineer.conversation.session import ConversationSession
from race_engineer.core.conversation import RadioLanguage
from race_engineer.fixtures import load_fixture
from race_engineer.observability import configure_logging


def _state_line(state: ReplayRaceState) -> str:
    frame = state.snapshot().context.frame
    return (
        f"REPLAY frame {state.index}/{state.frame_count - 1}, "
        f"session={frame.session_id}, sequence={frame.sequence}, "
        f"session_time={frame.session_time_s:.1f}s (paused; not live)"
    )


async def chat_replay(
    config_path: Path,
    directory: Path,
    *,
    frame_index: int = 0,
    question: str | None = None,
    reply_language: RadioLanguage | None = None,
    json_output: bool = False,
) -> int:
    if json_output and question is None:
        raise ValueError("--json requires --question")
    config = load_config(config_path)
    configure_logging(config.logging)
    fixture = load_fixture(directory)
    state = ReplayRaceState(fixture, config.policy.context)
    state.seek(frame_index)
    session = ConversationSession(
        conversation_planner(config.conversation), state.snapshot, config.conversation
    )
    if question is not None:
        reply = await session.ask(question, reply_language=reply_language)
        if json_output:
            print(reply.model_dump_json(indent=2))
        else:
            print(_state_line(state))
            print(f"Engineer: {reply.text}")
            if reply.status == "model_error":
                print("Start the local Qwen server; see docs/conversation.md.")
        return 1 if reply.status == "model_error" else 0

    print(_state_line(state))
    print("Ask freely in English or Turkish. Commands: /next [count], /frame index,")
    print("/state, /reset, /quit. /frame resets conversation history.")
    while True:
        try:
            line = (await asyncio.to_thread(input, "You: ")).strip()
        except EOFError:
            return 0
        if not line:
            continue
        if line == "/quit":
            return 0
        try:
            if line == "/state":
                print(_state_line(state))
            elif line == "/reset":
                session.reset()
                print("Conversation history cleared.")
            elif line.startswith("/"):
                parts = line.split()
                if parts[0] == "/next" and len(parts) in {1, 2}:
                    count = int(parts[1]) if len(parts) == 2 else 1
                    if count < 1:
                        raise ValueError("/next count must be positive")
                    state.seek(state.index + count)
                elif parts[0] == "/frame" and len(parts) == 2:
                    state.seek(int(parts[1]))
                    session.reset()
                else:
                    raise ValueError("unknown replay command")
                print(_state_line(state))
            else:
                reply = await session.ask(line, reply_language=reply_language)
                print(f"Engineer: {reply.text}")
                if reply.status == "model_error":
                    print("Start the local Qwen server; see docs/conversation.md.")
        except ValueError as error:
            # Do not echo validation errors containing user input into logs.
            del error
            print("Invalid input. Questions must be 1-1000 characters; check replay commands.")
