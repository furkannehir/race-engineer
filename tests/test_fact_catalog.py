from datetime import UTC, datetime, timedelta

import pytest

from race_engineer.config import DialogueConfig
from race_engineer.conversation.context_view import ConversationContextAssembler
from race_engineer.conversation.controller import DialogueController
from race_engineer.conversation.facts import (
    DEFAULT_FACT_PROVIDERS,
    FactCatalog,
    FactIdentityChanged,
    FactProvider,
)
from race_engineer.core.contracts import OpponentState, PlayerState, RaceContext, TelemetryFrame
from race_engineer.core.conversation import RaceAnswer, RaceQuery, RaceSnapshot
from race_engineer.core.dialogue import (
    DialogueState,
    OpponentReference,
    QueryPart,
    SemanticProposal,
)

NOW = datetime(2026, 9, 27, 12, tzinfo=UTC)


def snapshot() -> RaceSnapshot:
    frame = TelemetryFrame(
        source="synthetic",
        session_id="fact-catalog",
        sequence=7,
        observed_at=NOW,
        session_time_s=7,
        player=PlayerState(driver_id="player", position=6, lap_number=4, fuel_l=25.0),
        opponents=(
            OpponentState(driver_id="front", position=5, gap_to_player_s=-1.2),
            OpponentState(driver_id="rear", position=7, gap_to_player_s=0.8),
        ),
        capabilities=("opponents", "position"),
    )
    return RaceSnapshot(
        context=RaceContext(
            frame=frame,
            stint_lap=3,
            fuel_trend_l_per_lap=2.4,
            gap_ahead_s=1.2,
            gap_behind_s=0.8,
        ),
        as_of=NOW,
        mode="replay",
    )


def test_default_catalog_has_one_provider_for_every_bounded_race_query():
    catalog = FactCatalog()
    assert {provider.race_query for provider in catalog.providers} == set(RaceQuery)
    assert len({provider.key for provider in catalog.providers}) == len(catalog.providers)
    with pytest.raises(ValueError, match="every race query"):
        FactCatalog(DEFAULT_FACT_PROVIDERS[:-1])
    with pytest.raises(ValueError, match="unique"):
        FactCatalog((*DEFAULT_FACT_PROVIDERS, DEFAULT_FACT_PROVIDERS[0]))


def test_one_provider_registration_drives_capability_and_grounded_resolution():
    def alternate_position(snapshot, relation, opponent):
        del snapshot, relation, opponent
        return RaceAnswer(
            query=RaceQuery.POSITION,
            status="available",
            value=42,
            unit="position",
        )

    providers = tuple(
        FactProvider("position", RaceQuery.POSITION, alternate_position)
        if provider.race_query is RaceQuery.POSITION
        else provider
        for provider in DEFAULT_FACT_PROVIDERS
    )
    catalog = FactCatalog(providers)
    assembler = ConversationContextAssembler(DialogueConfig(), catalog)
    state = DialogueState(session_id="fact-catalog")
    request = assembler.assemble(
        snapshot(),
        state,
        turn_id="turn-1",
        question="Position?",
        received_at=NOW,
        deadline=NOW + timedelta(seconds=10),
    )
    position = next(item for item in request.context.capabilities if item.query == "position")
    assert position.status == "available"

    decision = DialogueController(DialogueConfig(), catalog).decide(
        request,
        SemanticProposal(
            language="en",
            requests=(QueryPart(part_id="q1", query="position"),),
            model_id="scripted",
        ),
        snapshot(),
        NOW,
    )
    assert decision.answers[0].query is RaceQuery.POSITION
    assert decision.answers[0].value == 42


def test_opponent_provider_uses_stable_identity_and_side():
    race = snapshot()
    catalog = FactCatalog()
    part = QueryPart(part_id="gap", query="gap", reference="ahead")
    answer = catalog.resolve_part(
        race,
        part,
        OpponentReference(side="ahead", driver_id="front"),
    )
    assert answer.query is RaceQuery.GAP_AHEAD and answer.value == 1.2
    with pytest.raises(FactIdentityChanged):
        catalog.resolve_part(
            race,
            part,
            OpponentReference(side="ahead", driver_id="rear"),
        )


def test_legacy_gap_query_keeps_nearest_aggregate_without_an_identity_anchor():
    answer = FactCatalog().resolve_query(snapshot().context, RaceQuery.GAP_BEHIND)
    assert answer.status == "available" and answer.value == 0.8
