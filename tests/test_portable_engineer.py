import asyncio
import json
from datetime import UTC, datetime, timedelta

import pytest

from race_engineer.config import ConversationConfig
from race_engineer.core.intelligence import (
    ContextPacket,
    DriverTurn,
    EvidenceItem,
)
from race_engineer.intelligence.grounding import StrictEvidenceGrounder
from race_engineer.intelligence.local_model import LocalIntelligenceError
from race_engineer.intelligence.portable_engineer import PortableQwenEngineer

NOW = datetime(2026, 9, 29, 12, tzinfo=UTC)


class FakeJsonModel:
    def __init__(self, result: object) -> None:
        self.result = result
        self.calls: list[dict[str, object]] = []

    async def request(self, **kwargs):
        self.calls.append(kwargs)
        if isinstance(self.result, list):
            return self.result[len(self.calls) - 1]
        return self.result


def turn(text: str = "Am I last?", language: str = "en") -> DriverTurn:
    return DriverTurn.model_validate(
        {
            "turn_id": "turn-ranking",
            "transcript": text,
            "received_at": NOW,
            "session_id": "iracing:1",
            "generation": 2,
            "asr_language": language,
            "reply_language": language,
        }
    )


def item(evidence_id: str, subject: str, value: int) -> EvidenceItem:
    return EvidenceItem(
        evidence_id=evidence_id,
        session_id="iracing:1",
        source_sequence=10,
        observed_at=NOW,
        kind="derived" if subject == "field" else "measurement",
        subject=subject,
        metric="position",
        value=value,
        unit="position",
        source_fields=(f"{subject}.position",),
        valid_until=NOW + timedelta(seconds=3),
    )


def fuel_item(evidence_id: str = "fuel", value: float = 30.0) -> EvidenceItem:
    return EvidenceItem(
        evidence_id=evidence_id,
        session_id="iracing:1",
        source_sequence=10,
        observed_at=NOW,
        kind="measurement",
        subject="player",
        metric="fuel_l",
        value=value,
        unit="l",
        source_fields=("player.fuel_l",),
        valid_until=NOW + timedelta(seconds=3),
    )


def is_last_item(value: bool) -> EvidenceItem:
    return EvidenceItem(
        evidence_id="c1:is_last",
        session_id="iracing:1",
        source_sequence=10,
        observed_at=NOW,
        kind="derived",
        subject="player",
        metric="is_last",
        value=value,
        source_fields=("player.position", "field.position"),
        valid_until=NOW + timedelta(seconds=3),
    )


def position_change_item(
    evidence_id: str,
    metric: str,
    value: str | int,
    unit: str | None = None,
) -> EvidenceItem:
    return EvidenceItem(
        evidence_id=evidence_id,
        session_id="iracing:1",
        source_sequence=10,
        observed_at=NOW,
        kind="derived",
        subject="player",
        metric=metric,
        value=value,
        unit=unit,
        source_fields=("player.position[30s]",),
        valid_until=NOW + timedelta(seconds=3),
    )


def packet(*evidence: EvidenceItem, unknowns: tuple[str, ...] = ()) -> ContextPacket:
    return ContextPacket(
        turn_id="turn-ranking",
        session_id="iracing:1",
        generation=2,
        source_sequence=10,
        assembled_at=NOW + timedelta(milliseconds=50),
        evidence=evidence,
        unknowns=unknowns,
    )


def ranking_draft() -> dict[str, object]:
    return {
        "goal": "inform",
        "tone": "calm_teammate",
        "guidance": ["answer the comparison directly"],
        "confidence": 1,
        "speech_template": "Yes, you're last right now, P{{position}} of {{last_position}}.",
        "references": [
            {
                "placeholder": "position",
                "evidence_id": "position",
                "field": "value",
            },
            {
                "placeholder": "last_position",
                "evidence_id": "last-position",
                "field": "value",
            },
        ],
    }


def test_portable_engineer_decides_and_writes_response_in_one_inference():
    model = FakeJsonModel(ranking_draft())
    engineer = PortableQwenEngineer(ConversationConfig(), model=model)
    current = packet(item("position", "player", 28), item("last-position", "field", 28))
    request = turn()

    brief = asyncio.run(engineer.decide(request, current))
    response = asyncio.run(engineer.generate(request, current, brief))
    grounded = StrictEvidenceGrounder().ground(response, current, brief)

    assert grounded.text == "Yes, you're last right now, P28 of 28."
    assert brief.evidence_ids == ("position", "last-position")
    assert len(model.calls) == 1
    payload = json.loads(str(model.calls[0]["content"]))
    assert payload["required_language"] == "en"
    assert [entry["value"] for entry in payload["evidence"]] == [28, 28]
    assert payload["rendering_bindings"] == [
        {
            "placeholder": "player_position",
            "evidence_id": "position",
            "field": "value",
            "meaning": "player position",
        },
        {
            "placeholder": "field_position",
            "evidence_id": "last-position",
            "field": "value",
            "meaning": "last occupied current field position; not a car count",
        },
    ]
    assert payload["deterministic_relationships"] == [
        {
            "relationship": "current_classification",
            "result": "player_is_last",
            "evidence_ids": ["position", "last-position"],
        }
    ]


def test_portable_engineer_allows_evidence_free_social_response():
    model = FakeJsonModel(
        {
            "goal": "acknowledge",
            "tone": "calm_teammate",
            "guidance": ["acknowledge briefly then refocus"],
            "confidence": 0.9,
            "speech_template": "Copy. Keep your head down and focus on the exits.",
            "references": [],
        }
    )
    engineer = PortableQwenEngineer(ConversationConfig(), model=model)
    request = turn("He has no idea about racing.")
    current = packet()

    brief = asyncio.run(engineer.decide(request, current))
    response = asyncio.run(engineer.generate(request, current, brief))

    assert brief.goal == "acknowledge"
    assert response.references == ()
    assert "equivalent Turkish shape" in model.calls[0]["system_prompt"]
    assert "do not claim to have seen" in model.calls[0]["system_prompt"]


def test_silence_schema_requires_null_speech_and_no_guidance_or_references():
    model = FakeJsonModel(
        {
            "goal": "silence",
            "tone": "calm_teammate",
            "guidance": [],
            "confidence": 1,
            "speech_template": None,
            "references": [],
        }
    )
    engineer = PortableQwenEngineer(ConversationConfig(), model=model)
    request = turn("He has no idea about racing.")
    current = packet()
    brief = asyncio.run(engineer.decide(request, current))
    response = asyncio.run(engineer.generate(request, current, brief))
    grounded = StrictEvidenceGrounder().ground(response, current, brief)
    assert grounded.action == "silence" and grounded.text is None
    assert len(model.calls) == 1
    schema = model.calls[0]["schema"]
    silent, spoken = schema["oneOf"]
    assert silent["properties"]["goal"]["const"] == "silence"
    assert silent["properties"]["speech_template"] == {"type": "null"}
    assert silent["properties"]["guidance"]["maxItems"] == 0
    assert silent["properties"]["references"]["maxItems"] == 0
    assert spoken["properties"]["goal"]["enum"] == [
        "inform",
        "analyze",
        "coach",
        "acknowledge",
        "clarify",
    ]
    assert spoken["properties"]["speech_template"]["type"] == "string"
    assert spoken["properties"]["speech_template"]["pattern"] == "^[^0-9]+$"
    assert "EvidenceReference" in schema["$defs"]


def test_empty_silent_draft_from_provider_bypass_is_still_rejected():
    model = FakeJsonModel(
        {"goal": "silence", "tone": "calm", "speech_template": "", "references": []}
    )
    engineer = PortableQwenEngineer(ConversationConfig(), model=model)
    with pytest.raises(LocalIntelligenceError, match="engineer_model_response_invalid"):
        asyncio.run(engineer.decide(turn("Thanks"), packet()))
    assert len(model.calls) == 2


def test_portable_engineer_uses_capability_relationship_without_speaking_boolean():
    model = FakeJsonModel(
        {
            "goal": "inform",
            "tone": "calm_teammate",
            "guidance": [],
            "confidence": 1,
            "speech_template": "No, you're P{{position}}. Last place is P{{last_position}}.",
            "references": [
                {
                    "placeholder": "position",
                    "evidence_id": "c1:player_position",
                    "field": "value",
                },
                {
                    "placeholder": "last_position",
                    "evidence_id": "c1:last_position",
                    "field": "value",
                },
            ],
        }
    )
    engineer = PortableQwenEngineer(ConversationConfig(), model=model)
    current = packet(
        item("c1:player_position", "player", 5),
        item("c1:last_position", "field", 6),
        is_last_item(False),
    )

    brief = asyncio.run(engineer.decide(turn(), current))

    assert brief.evidence_ids == ("c1:player_position", "c1:last_position")
    payload = json.loads(str(model.calls[0]["content"]))
    assert payload["deterministic_relationships"] == [
        {
            "relationship": "current_classification",
            "result": "player_is_not_last",
            "evidence_ids": ["c1:player_position", "c1:last_position"],
        }
    ]
    assert all(binding["evidence_id"] != "c1:is_last" for binding in payload["rendering_bindings"])
    schema = model.calls[0]["schema"]
    allowed_ids = schema["$defs"]["EvidenceReference"]["properties"]["evidence_id"]["enum"]
    assert "c1:is_last" not in allowed_ids


def test_portable_engineer_preserves_position_change_direction_and_magnitude():
    model = FakeJsonModel(
        {
            "goal": "inform",
            "tone": "calm_teammate",
            "guidance": [],
            "confidence": 1,
            "speech_template": "We've {{direction}} {{magnitude}} positions.",
            "references": [
                {
                    "placeholder": "direction",
                    "evidence_id": "c1:direction",
                    "field": "value",
                },
                {
                    "placeholder": "magnitude",
                    "evidence_id": "c1:positions_changed",
                    "field": "value",
                },
            ],
        }
    )
    engineer = PortableQwenEngineer(ConversationConfig(), model=model)
    current = packet(
        position_change_item(
            "c1:direction",
            "position_change_direction",
            "gained",
        ),
        position_change_item(
            "c1:positions_changed",
            "positions_changed",
            3,
            "positions",
        ),
    )

    brief = asyncio.run(engineer.decide(turn("Have we gained any positions?"), current))
    response = asyncio.run(engineer.generate(turn("Have we gained any positions?"), current, brief))
    grounded = StrictEvidenceGrounder().ground(response, current, brief)

    assert grounded.text == "We've gained 3 positions."
    payload = json.loads(str(model.calls[0]["content"]))
    assert payload["deterministic_relationships"] == [
        {
            "relationship": "position_change",
            "result": "gained",
            "evidence_ids": ["c1:direction", "c1:positions_changed"],
        }
    ]


def test_portable_engineer_supplies_generic_scalar_binding_example():
    model = FakeJsonModel(
        {
            "goal": "inform",
            "tone": "calm_teammate",
            "guidance": [],
            "confidence": 1,
            "speech_template": "Fuel remaining is {{player_fuel_l}} litres.",
            "references": [
                {
                    "placeholder": "player_fuel_l",
                    "evidence_id": "fuel",
                    "field": "value",
                }
            ],
        }
    )
    engineer = PortableQwenEngineer(ConversationConfig(), model=model)
    request = turn("What's the fuel situation?")
    current = packet(fuel_item())

    brief = asyncio.run(engineer.decide(request, current))
    response = asyncio.run(engineer.generate(request, current, brief))
    grounded = StrictEvidenceGrounder().ground(response, current, brief)

    assert grounded.text == "Fuel remaining is 30 litres."
    prompt = str(model.calls[0]["system_prompt"])
    assert '"speech_template":"Current fuel: {{player_fuel_l}} litres."' in prompt
    assert '"evidence_id":"fuel"' in prompt


def test_portable_engineer_reports_unsupported_projection_without_clarifying():
    model = FakeJsonModel(
        {
            "goal": "inform",
            "tone": "calm_teammate",
            "guidance": ["state the unavailable projection"],
            "confidence": 1,
            "speech_template": (
                "I can't project the pit exit position yet; "
                "the pit-loss and traffic model isn't available."
            ),
            "references": [],
        }
    )
    engineer = PortableQwenEngineer(ConversationConfig(), model=model)
    request = turn("Where would I be if I pit this lap?")
    current = packet(unknowns=("pit_exit_projection_unavailable",))

    brief = asyncio.run(engineer.decide(request, current))
    response = asyncio.run(engineer.generate(request, current, brief))

    assert brief.goal == "inform"
    assert response.action == "speak"
    assert "position yet" in (response.speech_template or "")
    assert response.references == ()


def test_portable_engineer_corrects_driver_claim_from_current_evidence():
    model = FakeJsonModel(
        {
            "goal": "inform",
            "tone": "calm_teammate",
            "guidance": ["correct the claim calmly"],
            "confidence": 1,
            "speech_template": (
                "Telemetry has you P{{position}}; the current field runs to P{{last_position}}."
            ),
            "references": [
                {
                    "placeholder": "position",
                    "evidence_id": "position",
                    "field": "value",
                },
                {
                    "placeholder": "last_position",
                    "evidence_id": "last-position",
                    "field": "value",
                },
            ],
        }
    )
    engineer = PortableQwenEngineer(ConversationConfig(), model=model)
    request = turn("I'm P6 in a ten-car grid.")
    current = packet(item("position", "player", 5), item("last-position", "field", 6))

    brief = asyncio.run(engineer.decide(request, current))
    response = asyncio.run(engineer.generate(request, current, brief))
    grounded = StrictEvidenceGrounder().ground(response, current, brief)

    assert grounded.text == "Telemetry has you P5; the current field runs to P6."
    assert "P6 in a ten-car grid" not in grounded.text
    payload = json.loads(str(model.calls[0]["content"]))
    assert payload["transcript"] == ("I'm P[unverified_number] in a [unverified_number]-car grid.")


@pytest.mark.parametrize(
    "draft",
    [
        {
            **ranking_draft(),
            "references": [{"placeholder": "missing", "evidence_id": "missing", "field": "value"}],
            "speech_template": "You're {{missing}}.",
        },
        {
            **ranking_draft(),
            "references": [],
            "speech_template": "You're P28.",
        },
    ],
)
def test_portable_engineer_rejects_unavailable_evidence_or_raw_numbers(draft):
    engineer = PortableQwenEngineer(ConversationConfig(), model=FakeJsonModel(draft))
    with pytest.raises(LocalIntelligenceError):
        asyncio.run(engineer.decide(turn(), packet()))


def test_portable_engineer_retries_one_structurally_invalid_response():
    invalid = {
        **ranking_draft(),
        "speech_template": "You're P{{position}}.",
        "references": [
            {
                "placeholder": "position",
                "evidence_id": "position",
                "field": "value",
            }
        ],
    }
    model = FakeJsonModel([invalid, ranking_draft()])
    engineer = PortableQwenEngineer(ConversationConfig(), model=model)
    current = packet(item("position", "player", 28), item("last-position", "field", 28))

    brief = asyncio.run(engineer.decide(turn(), current))
    response = asyncio.run(engineer.generate(turn(), current, brief))

    assert response.speech_template == ranking_draft()["speech_template"]
    assert len(model.calls) == 2
    assert "previous JSON did not satisfy" in str(model.calls[1]["system_prompt"])


def test_response_can_ignore_unrelated_retrieved_comparison():
    draft = {
        "goal": "inform",
        "tone": "calm_teammate",
        "speech_template": "Fuel remaining: {{player_fuel_l}} litres.",
        "references": [{"placeholder": "player_fuel_l", "evidence_id": "fuel", "field": "value"}],
    }
    model = FakeJsonModel(draft)
    engineer = PortableQwenEngineer(ConversationConfig(), model=model)
    current = packet(fuel_item(), item("position", "player", 5), item("last-position", "field", 6))
    request = turn("How much fuel do I have?")

    brief = asyncio.run(engineer.decide(request, current))
    response = asyncio.run(engineer.generate(request, current, brief))
    grounded = StrictEvidenceGrounder().ground(response, current, brief)

    assert grounded.text == "Fuel remaining: 30 litres."
    assert brief.evidence_ids == ("fuel",)
    assert len(model.calls) == 1


@pytest.mark.parametrize("include_fuel", [False, True])
def test_using_part_of_a_comparison_still_requires_all_its_evidence(include_fuel):
    references = [{"placeholder": "position", "evidence_id": "position", "field": "value"}]
    template = "You're P{{position}}."
    if include_fuel:
        references.append({"placeholder": "fuel", "evidence_id": "fuel", "field": "value"})
        template += " Fuel: {{fuel}} litres."
    model = FakeJsonModel(
        {**ranking_draft(), "speech_template": template, "references": references}
    )
    engineer = PortableQwenEngineer(ConversationConfig(), model=model)
    current = packet(fuel_item(), item("position", "player", 5), item("last-position", "field", 6))

    with pytest.raises(LocalIntelligenceError, match="engineer_model_response_invalid"):
        asyncio.run(engineer.decide(turn(), current))
    assert len(model.calls) == 2


def test_evidence_free_comparison_remains_rejected():
    model = FakeJsonModel({**ranking_draft(), "speech_template": "You're last.", "references": []})
    engineer = PortableQwenEngineer(ConversationConfig(), model=model)
    current = packet(item("position", "player", 28), item("last-position", "field", 28))
    with pytest.raises(LocalIntelligenceError, match="engineer_model_response_invalid"):
        asyncio.run(engineer.decide(turn(), current))


def test_using_one_relationship_does_not_require_an_unrelated_relationship():
    model = FakeJsonModel(ranking_draft())
    engineer = PortableQwenEngineer(ConversationConfig(), model=model)
    current = packet(
        item("position", "player", 28),
        item("last-position", "field", 28),
        position_change_item("c2:direction", "position_change_direction", "gained"),
        position_change_item("c2:positions_changed", "positions_changed", 2, "position"),
    )
    brief = asyncio.run(engineer.decide(turn(), current))
    assert brief.evidence_ids == ("position", "last-position")
    assert len(model.calls) == 1
