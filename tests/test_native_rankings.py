import asyncio

import pytest
from test_context_engine import driver_turn, race_context
from test_native_lfm import Model, tool
from test_portable_engineer import FakeJsonModel

from race_engineer.config import ConversationConfig
from race_engineer.intelligence.capabilities import DeterministicRaceCapabilities
from race_engineer.intelligence.context_engine import QueryDrivenContextEngineer
from race_engineer.intelligence.context_planner import LFM_NATIVE_ID
from race_engineer.intelligence.grounding import StrictEvidenceGrounder
from race_engineer.intelligence.native_lfm import LfmNativePlanner
from race_engineer.intelligence.orchestrator import EngineerOrchestrator
from race_engineer.intelligence.portable_engineer import PortableQwenEngineer
from race_engineer.intelligence.telemetry_memory import BoundedTelemetryMemory


def engineer(position, opponents, check):
    memory = BoundedTelemetryMemory()
    memory.update(race_context(1, 100, position=position, opponent_positions=opponents))
    config = ConversationConfig(context_planner=LFM_NATIVE_ID)
    model = Model(tool(f"check_if_driver_is_{check}"))
    planner = LfmNativePlanner(config, model=model)
    context = QueryDrivenContextEngineer(
        memory, planner, DeterministicRaceCapabilities(memory, include_rank_checks=True)
    )
    voice = FakeJsonModel({"goal": "inform", "speech": "We're P{{position}}."})
    core = PortableQwenEngineer(config, model=voice)
    return memory, EngineerOrchestrator(context, core, core, StrictEvidenceGrounder()), model, voice


@pytest.mark.parametrize(
    "position,opponents,first,last",
    [
        (1, (2, 6), "Yes", "No"),
        (5, (1, 6), "No", "No"),
        (6, (1, 5), "No", "Yes"),
        (1, (), "Yes", "Yes"),
    ],
)
@pytest.mark.parametrize("check", ["first", "last"])
def test_rank_answer_is_calculated_from_telemetry_not_old_dialogue(
    position, opponents, first, last, check
):
    _, app, model, voice = engineer(position, opponents, check)
    request = driver_turn(f"Are we {check}?").model_copy(
        update={"recent_dialogue": ("driver: Position?", "engineer: We're P28; last is P28.")}
    )
    result = asyncio.run(app.respond(request))
    assert result.text == f"{first if check == 'first' else last}. We're P{position}."
    assert result.evidence_ids == (f"c1:is_{check}", "c1:player_position")
    assert model.requests[0]["messages"][-1]["content"] == request.transcript
    assert len(model.requests) == len(voice.calls) == 1


@pytest.mark.parametrize(
    "check,old,new,opponents,expected",
    [
        ("first", 5, 1, (2, 6), "Yes. We're P1."),
        ("first", 1, 2, (1, 6), "No. We're P2."),
        ("last", 5, 6, (1, 5), "Yes. We're P6."),
        ("last", 6, 5, (1, 6), "No. We're P5."),
    ],
)
def test_both_yes_no_and_position_refresh_before_speech(check, old, new, opponents, expected):
    memory, app, _, _ = engineer(old, opponents, check)

    async def run():
        prepared = await app.prepare(driver_turn(f"Are we {check}?"))
        memory.update(race_context(2, 101, position=new, opponent_positions=opponents))
        result = await app.ground(prepared)
        assert result.text == expected

    asyncio.run(run())
