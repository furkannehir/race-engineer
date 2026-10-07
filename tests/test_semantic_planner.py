import asyncio
from collections.abc import Sequence
from datetime import UTC, datetime

from race_engineer.core.intelligence import (
    CapabilityDescriptor,
    CapabilityOutputDescriptor,
    DriverTurn,
    SignalDescriptor,
    SignalSelector,
)
from race_engineer.intelligence.semantic_planner import CatalogSemanticPlanner


class SelectingScorer:
    def __init__(self, needle: str) -> None:
        self.needle = needle
        self.hypotheses: tuple[str, ...] = ()

    async def score(
        self,
        text: str,
        hypotheses: Sequence[str],
        *,
        language: str | None,
    ) -> tuple[float, ...]:
        del text, language
        self.hypotheses = tuple(hypotheses)
        return tuple(1.0 if self.needle in item else 0.0 for item in hypotheses)


def _turn(text: str) -> DriverTurn:
    return DriverTurn(
        turn_id="semantic-turn",
        transcript=text,
        received_at=datetime(2026, 10, 4, tzinfo=UTC),
        session_id="semantic-session",
        generation=0,
        reply_language="en",
    )


def _capability() -> CapabilityDescriptor:
    return CapabilityDescriptor(
        capability_id="current_classification",
        description="Calculate current classification and whether the player is last.",
        temporal_scope="current",
        outputs=(
            CapabilityOutputDescriptor(
                output_id="player_position",
                description="current position",
            ),
        ),
        freshness_s=3.0,
        uncertainty="Only currently classified cars are included.",
        available=True,
    )


def test_semantic_candidate_selects_dynamic_capability_description() -> None:
    scorer = SelectingScorer("whether the player is last")
    planner = CatalogSemanticPlanner(scorer, planner_id="semantic-test")

    result = asyncio.run(planner.plan(_turn("Where are we?"), (), (_capability(),)))

    assert result.temporal_scope == "current"
    assert result.capability_requests[0].capability_id == "current_classification"
    assert result.capability_requests[0].request_id == "c1"
    assert result.queries == ()
    assert any("current classification" in item for item in scorer.hypotheses)


def test_semantic_candidate_selects_normalized_raw_signal() -> None:
    scorer = SelectingScorer("speed mps")
    planner = CatalogSemanticPlanner(scorer, planner_id="semantic-test")
    signals = (
        SignalDescriptor(
            selector=SignalSelector(source="player", signal="lap_number"),
            unit="lap",
            available=True,
        ),
        SignalDescriptor(
            selector=SignalSelector(source="player", signal="speed_mps"),
            unit="m/s",
            available=True,
        ),
    )

    result = asyncio.run(planner.plan(_turn("Tell me that value."), signals, ()))

    assert result.capability_requests == ()
    assert result.queries[0].query_id == "e1"
    assert result.queries[0].selector.signal == "speed_mps"
    assert result.queries[0].operation == "latest"


def test_semantic_candidate_can_choose_evidence_free_social_turn() -> None:
    scorer = SelectingScorer("social or emotional remark")
    planner = CatalogSemanticPlanner(scorer, planner_id="semantic-test")

    result = asyncio.run(planner.plan(_turn("Thanks."), (), (_capability(),)))

    assert result.temporal_scope == "social"
    assert result.capability_requests == ()
    assert result.queries == ()
