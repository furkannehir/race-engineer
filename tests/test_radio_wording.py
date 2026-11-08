import asyncio
import json

import pytest
from test_portable_engineer import FakeJsonModel, fuel_item, item, packet, turn

from race_engineer.config import ConversationConfig
from race_engineer.intelligence.context_planner import LFM_NATIVE_ID
from race_engineer.intelligence.grounding import StrictEvidenceGrounder
from race_engineer.intelligence.local_model import LocalIntelligenceError
from race_engineer.intelligence.portable_engineer import PortableQwenEngineer


def native(model):
    return PortableQwenEngineer(ConversationConfig(context_planner=LFM_NATIVE_ID), model=model)


async def response(core, request, context):
    brief = await core.decide(request, context)
    generated = await core.generate(request, context, brief)
    return brief, generated, StrictEvidenceGrounder().ground(generated, context, brief)


@pytest.mark.parametrize(
    "framing,expected",
    [
        ("direct", "I don't have that data available here."),
        ("explain", "That information isn't available in the data I have here."),
        ("acknowledge", "Copy. I don't have that data available here."),
    ],
)
def test_limitation_framing_changes_wording_without_exposing_model_prose(framing, expected):
    model = FakeJsonModel({"framing": framing})
    current = packet(unknowns=("requested_analysis_unavailable",)).model_copy(
        update={"temporal_scope": "current"}
    )
    request = turn("You don't have access to the damage data?").model_copy(
        update={
            "recent_dialogue": (
                "driver: Damage?",
                "engineer: I don't have that data available here.",
            )
        }
    )
    brief, generated, result = asyncio.run(response(native(model), request, current))
    assert result.text == expected
    assert brief.unknown_ids == generated.limitation_references[0].unknown_ids == current.unknowns
    assert generated.references == () and result.evidence_ids == ()
    assert len(model.calls) == 1 and model.calls[0]["max_tokens"] == 64
    payload = json.loads(model.calls[0]["content"])
    assert payload["delivered_dialogue"] == list(request.recent_dialogue)
    assert payload["transcript"] == request.transcript
    assert "evidence" not in payload  # No old history value can become a new measurement.


@pytest.mark.parametrize(
    "scope,expected",
    [
        ("historical", "I don't have that earlier data available."),
        ("future_counterfactual", "I can't make that prediction from the data I have."),
    ],
)
def test_explanatory_style_cannot_turn_missing_history_or_projection_into_a_fact(scope, expected):
    model = FakeJsonModel({"framing": "explain"})
    current = packet(unknowns=("requested_analysis_unavailable",)).model_copy(
        update={"temporal_scope": scope}
    )
    _, _, result = asyncio.run(response(native(model), turn(), current))
    assert result.text == expected and result.evidence_ids == ()


@pytest.mark.parametrize(
    "raw",
    [
        {"framing": "the car is healthy"},
        {"framing": "explain", "speech": "No damage, keep pushing."},
        LocalIntelligenceError("model_response_incomplete"),
    ],
    ids=["invalid-choice", "extra-claim", "incomplete"],
)
def test_bad_or_failed_framing_uses_visible_safe_fallback_without_retry(raw, caplog):
    class Model:
        calls = 0

        async def request(self, **kwargs):
            self.calls += 1
            if isinstance(raw, Exception):
                raise raw
            return raw

    model = Model()
    current = packet(unknowns=("requested_analysis_unavailable",)).model_copy(
        update={"temporal_scope": "current"}
    )
    _, _, result = asyncio.run(response(native(model), turn("Damage?"), current))
    assert result.text == "I don't have that data available here."
    assert model.calls == 1
    assert any(r.event == "core_limitation_framing_fallback" for r in caplog.records)
    assert "the car is healthy" not in caplog.text and "keep pushing" not in caplog.text


def test_limitation_style_does_not_swallow_cancellation():
    class Model:
        async def request(self, **kwargs):
            raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(native(Model()).decide(turn(), packet(unknowns=("unavailable",))))


def test_clarification_still_asks_instead_of_saying_the_data_is_missing():
    model = FakeJsonModel(AssertionError("Ambiguous references must not call framing"))
    current = packet(unknowns=("requested_reference_ambiguous",))
    _, _, result = asyncio.run(response(native(model), turn("And him?"), current))
    assert result.action == "clarify" and result.text == "Could you clarify what you mean?"
    assert model.calls == []


@pytest.mark.parametrize(
    "body,expected",
    [
        ("Fuel is {{fuel}} litres.", "I don't have that data available here. Fuel is 0 litres."),
        (
            "and your fuel is {{fuel}} litres.",
            "I don't have that data available here. Your fuel is 0 litres.",
        ),
    ],
)
def test_radio_partial_keeps_known_zero_and_removes_unsupported_health_claim(body, expected):
    model = FakeJsonModel({"goal": "inform", "speech": "The tyres are healthy. " + body})
    fuel = fuel_item(value=0).model_copy(update={"temporal_scope": "current"})
    current = packet(fuel, unknowns=("requested_analysis_unavailable",)).model_copy(
        update={"temporal_scope": "current"}
    )
    _, generated, result = asyncio.run(response(native(model), turn("Tyres and fuel?"), current))
    assert result.text == expected
    assert generated.limitation_references[0].wording == "radio"
    assert result.evidence_ids == ("fuel",) and len(model.calls) == 1


@pytest.mark.parametrize("question", ["Are we last?", "Are we first?", "Are we P1?"])
def test_native_factual_turn_keeps_question_and_excludes_old_spoken_measurements(question):
    model = FakeJsonModel(
        {"goal": "inform", "speech": "No, we're P{{position}}; last is P{{lastposition}}."}
    )
    current = packet(item("position", "player", 5), item("last", "field", 6))
    request = turn(question).model_copy(
        update={"recent_dialogue": ("driver: Position?", "engineer: You're P1; last place is P28.")}
    )
    _, _, result = asyncio.run(response(native(model), request, current))
    payload = json.loads(model.calls[0]["content"])
    assert payload["driver_question"] == question
    assert "recent_dialogue" not in payload
    assert "[unverified_number]" not in model.calls[0]["content"]
    assert result.text == "No, we're P5; last is P6."


@pytest.mark.parametrize(
    "position,last,leader,result",
    [
        (1, 6, "player is first", "player is not last"),
        (5, 6, "player is not first", "player is not last"),
        (6, 6, "player is not first", "player is last"),
        (1, 1, "player is first", "player is last"),
    ],
)
def test_leader_and_last_place_are_independent_telemetry_checks(position, last, leader, result):
    model = FakeJsonModel(
        {"goal": "inform", "speech": "We're P{{position}}; last is P{{lastposition}}."}
    )
    current = packet(item("position", "player", position), item("last", "field", last))
    asyncio.run(response(native(model), turn("Position?"), current))
    relationship = json.loads(model.calls[0]["content"])["relationships"][0]
    assert relationship["leader_check"] == leader
    assert relationship["result"] == result
    assert relationship["required_placeholders"] == ["position", "lastposition"]


@pytest.mark.parametrize("literal", ["one hundred", "twenty-four", "five"])
def test_unverified_current_claim_cannot_escape_as_spelled_out_numeric_telemetry(literal):
    model = FakeJsonModel(
        [
            {"goal": "inform", "speech": f"Fuel is {{{{fuel}}}} litres, plus {literal} litres."},
            {"goal": "inform", "speech": "Fuel is {{fuel}} litres."},
        ]
    )
    _, _, result = asyncio.run(
        response(
            native(model), turn(f"I have {literal} litres, right?"), packet(fuel_item(value=24))
        )
    )
    assert result.text == "Fuel is 24 litres."
    assert len(model.calls) == 2


def test_native_redaction_marker_is_still_rejected_if_provider_returns_it():
    model = FakeJsonModel(
        [
            {"goal": "inform", "speech": "We're P[unverified_number]."},
            {"goal": "inform", "speech": "We're P{{position}}; last is P{{lastposition}}."},
        ]
    )
    current = packet(item("position", "player", 5), item("last", "field", 6))
    _, _, result = asyncio.run(response(native(model), turn("Position?"), current))
    assert result.text == "We're P5; last is P6." and len(model.calls) == 2


@pytest.mark.parametrize(
    "subject,metric,unit,alias,value,expected",
    [
        ("car_behind", "gap_s", "s", "carbehindgaps", 1.6, "Gap to the car behind is 1.6 seconds."),
        ("car_ahead", "gap_s", "s", "caraheadgaps", 0, "Gap to the car ahead is 0 seconds."),
        ("player", "fuel_l", "l", "fuel", 0, "Your fuel is 0 litres."),
    ],
)
def test_bare_numeric_reply_retains_direction_units_and_verified_reference(
    subject, metric, unit, alias, value, expected
):
    evidence = fuel_item(value=value).model_copy(
        update={"subject": subject, "metric": metric, "unit": unit}
    )
    model = FakeJsonModel({"goal": "inform", "speech": "{{" + alias + "}}"})
    _, generated, result = asyncio.run(response(native(model), turn("Update?"), packet(evidence)))
    assert result.text == expected and result.evidence_ids == (evidence.evidence_id,)
    assert generated.references[0].evidence_id == evidence.evidence_id
    assert generated.references[0].field == "value" and len(model.calls) == 1


def test_full_generated_sentence_keeps_its_wording():
    model = FakeJsonModel({"goal": "inform", "speech": "You've got {{fuel}} litres left."})
    _, _, result = asyncio.run(response(native(model), turn("Fuel?"), packet(fuel_item())))
    assert result.text == "You've got 30 litres left."
