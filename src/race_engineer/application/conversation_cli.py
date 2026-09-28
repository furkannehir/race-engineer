"""Text-first local conversation over an explicitly paused replay."""

import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path

from race_engineer.config import load_config
from race_engineer.conversation.composer import compose
from race_engineer.conversation.dialogue_session import DialogueSession
from race_engineer.conversation.factory import conversation_planner, semantic_judge
from race_engineer.conversation.replay import ReplayRaceState
from race_engineer.conversation.session import ConversationSession
from race_engineer.core.contracts import ContractModel
from race_engineer.core.conversation import RadioLanguage
from race_engineer.core.dialogue import DeliveryEvent, ResponseDecision
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
    dialogue = (
        DialogueSession(
            semantic_judge(config.conversation),
            state.snapshot,
            config.conversation.dialogue,
        )
        if config.conversation.dialogue.enabled
        else None
    )
    legacy = (
        None
        if dialogue is not None
        else ConversationSession(
            conversation_planner(config.conversation), state.snapshot, config.conversation
        )
    )

    async def ask(text: str) -> tuple[ContractModel, str | None]:
        if dialogue is None:
            assert legacy is not None
            reply = await legacy.ask(text, reply_language=reply_language)
            return reply, reply.text
        decision = await dialogue.ask(text, reply_language=reply_language, output_mode="text")
        output = compose(decision)
        if output is not None and decision.outcome != "discarded":
            dialogue.record_delivery(
                DeliveryEvent(
                    turn_id=decision.turn_id,
                    response_id=decision.response_id,
                    session_id=decision.session_id,
                    generation=decision.generation,
                    output_mode="text",
                    status="completed",
                    occurred_at=datetime.now(UTC),
                )
            )
        return decision, output

    def failed(result: object) -> bool:
        if isinstance(result, ResponseDecision):
            return result.reason == "model_error"
        return getattr(result, "status", None) == "model_error"

    def reset() -> None:
        if dialogue is not None:
            dialogue.reset()
        else:
            assert legacy is not None
            legacy.reset()

    if question is not None:
        result, text = await ask(question)
        if json_output:
            if dialogue is None:
                # Preserve the established replay JSON contract on the default route.
                print(result.model_dump_json(indent=2))
            else:
                print(
                    json.dumps(
                        {"result": json.loads(result.model_dump_json()), "text": text},
                        ensure_ascii=False,
                        indent=2,
                    )
                )
        else:
            print(_state_line(state))
            print("Engineer: [no reply]" if text is None else f"Engineer: {text}")
            if failed(result):
                print("Start the local Qwen server; see docs/conversation.md.")
        if dialogue is not None:
            await dialogue.aclose()
        return 1 if failed(result) else 0

    print(_state_line(state))
    if dialogue is not None:
        print("CE-05 dialogue preview enabled for this replay.")
    print("Ask freely in English or Turkish. Commands: /next [count], /frame index,")
    print("/state, /reset, /quit. /frame resets conversation history.")
    while True:
        try:
            line = (await asyncio.to_thread(input, "You: ")).strip()
        except EOFError:
            if dialogue is not None:
                await dialogue.aclose()
            return 0
        if not line:
            continue
        if line == "/quit":
            if dialogue is not None:
                await dialogue.aclose()
            return 0
        try:
            if line == "/state":
                print(_state_line(state))
            elif line == "/reset":
                reset()
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
                    reset()
                else:
                    raise ValueError("unknown replay command")
                print(_state_line(state))
            else:
                result, text = await ask(line)
                print("Engineer: [no reply]" if text is None else f"Engineer: {text}")
                if failed(result):
                    print("Start the local Qwen server; see docs/conversation.md.")
        except ValueError as error:
            # Do not echo validation errors containing user input into logs.
            del error
            print("Invalid input. Questions must be 1-1000 characters; check replay commands.")
