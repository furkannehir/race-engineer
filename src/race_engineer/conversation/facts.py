"""Declarative registry of deterministic, capability-aware race facts."""

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Literal, cast

from race_engineer.core.contracts import OpponentState, RaceContext
from race_engineer.core.conversation import FieldRelation, RaceAnswer, RaceQuery, RaceSnapshot
from race_engineer.core.dialogue import (
    DialogueQuery,
    OpponentReference,
    QueryCapability,
    QueryPart,
)

type ProviderReference = Literal["none", "ahead", "behind"]
type FactResolver = Callable[
    [RaceSnapshot, FieldRelation | None, OpponentReference | None], RaceAnswer
]


class FactIdentityChanged(Exception):
    """A previously grounded opponent no longer exists on the same side."""


@dataclass(frozen=True, slots=True)
class FactProvider:
    dialogue_query: DialogueQuery
    race_query: RaceQuery
    resolver: FactResolver
    reference: ProviderReference = "none"

    @property
    def key(self) -> tuple[DialogueQuery, ProviderReference]:
        return self.dialogue_query, self.reference


def _answer(query: RaceQuery, value: int | float | None, unit: str) -> RaceAnswer:
    if value is None or value < 0:
        return RaceAnswer(query=query, status="missing")
    return RaceAnswer.model_validate(
        {"query": query, "status": "available", "value": value, "unit": unit}
    )


def _scalar(
    query: RaceQuery,
    unit: str,
    value: Callable[[RaceContext], int | float | None],
) -> FactResolver:
    def resolve(
        snapshot: RaceSnapshot,
        relation: FieldRelation | None,
        opponent: OpponentReference | None,
    ) -> RaceAnswer:
        del relation, opponent
        return _answer(query, value(snapshot.context), unit)

    return resolve


def _field_status(
    snapshot: RaceSnapshot,
    relation: FieldRelation | None,
    opponent: OpponentReference | None,
) -> RaceAnswer:
    del opponent
    selected = relation or "last"
    frame = snapshot.context.frame
    position = frame.player.position
    if position is None or not {"position", "opponents"}.issubset(frame.capabilities):
        return RaceAnswer(query=RaceQuery.FIELD_STATUS, status="missing", field_relation=selected)
    positions = [position]
    for item in frame.opponents:
        if item.position is None:
            return RaceAnswer(
                query=RaceQuery.FIELD_STATUS, status="missing", field_relation=selected
            )
        positions.append(item.position)
    if len(positions) != len(set(positions)):
        return RaceAnswer(query=RaceQuery.FIELD_STATUS, status="missing", field_relation=selected)
    total = max(positions)
    if set(positions) != set(range(1, total + 1)):
        return RaceAnswer(query=RaceQuery.FIELD_STATUS, status="missing", field_relation=selected)
    return RaceAnswer(
        query=RaceQuery.FIELD_STATUS,
        status="available",
        value=position,
        total=total,
        unit="position",
        field_relation=selected,
    )


def _current_opponent(
    snapshot: RaceSnapshot,
    side: Literal["ahead", "behind"],
    opponent: OpponentReference | None,
) -> OpponentState | None:
    if opponent is None:
        return None
    current = next(
        (item for item in snapshot.context.frame.opponents if item.driver_id == opponent.driver_id),
        None,
    )
    expected_ahead = side == "ahead"
    if (
        opponent.side != side
        or current is None
        or current.gap_to_player_s is None
        or (current.gap_to_player_s < 0) != expected_ahead
    ):
        raise FactIdentityChanged(opponent.driver_id)
    return current


def _opponent_gap(side: Literal["ahead", "behind"], query: RaceQuery) -> FactResolver:
    def resolve(
        snapshot: RaceSnapshot,
        relation: FieldRelation | None,
        opponent: OpponentReference | None,
    ) -> RaceAnswer:
        del relation
        current = _current_opponent(snapshot, side, opponent)
        if current is None:
            return RaceAnswer(query=query, status="missing")
        gap = current.gap_to_player_s
        assert gap is not None
        return _answer(query, abs(gap), "s")

    return resolve


def _opponent_unsupported(side: Literal["ahead", "behind"], query: RaceQuery) -> FactResolver:
    def resolve(
        snapshot: RaceSnapshot,
        relation: FieldRelation | None,
        opponent: OpponentReference | None,
    ) -> RaceAnswer:
        del relation
        if opponent is not None:
            _current_opponent(snapshot, side, opponent)
        return RaceAnswer(query=query, status="unsupported")

    return resolve


def _unsupported(query: RaceQuery) -> FactResolver:
    def resolve(
        snapshot: RaceSnapshot,
        relation: FieldRelation | None,
        opponent: OpponentReference | None,
    ) -> RaceAnswer:
        del snapshot, relation, opponent
        return RaceAnswer(query=query, status="unsupported")

    return resolve


DEFAULT_FACT_PROVIDERS: tuple[FactProvider, ...] = (
    FactProvider(
        "position",
        RaceQuery.POSITION,
        _scalar(RaceQuery.POSITION, "position", lambda context: context.frame.player.position),
    ),
    FactProvider("field_status", RaceQuery.FIELD_STATUS, _field_status),
    FactProvider(
        "lap",
        RaceQuery.LAP,
        _scalar(RaceQuery.LAP, "lap", lambda context: context.frame.player.lap_number),
    ),
    FactProvider(
        "fuel_remaining",
        RaceQuery.FUEL_REMAINING,
        _scalar(RaceQuery.FUEL_REMAINING, "l", lambda context: context.frame.player.fuel_l),
    ),
    FactProvider(
        "fuel_consumption",
        RaceQuery.FUEL_CONSUMPTION,
        _scalar(
            RaceQuery.FUEL_CONSUMPTION,
            "l/lap",
            lambda context: context.fuel_trend_l_per_lap,
        ),
    ),
    FactProvider(
        "fuel_to_finish",
        RaceQuery.FUEL_TO_FINISH,
        _unsupported(RaceQuery.FUEL_TO_FINISH),
    ),
    FactProvider("unsupported", RaceQuery.UNSUPPORTED, _unsupported(RaceQuery.UNSUPPORTED)),
    FactProvider("gap", RaceQuery.GAP_AHEAD, _opponent_gap("ahead", RaceQuery.GAP_AHEAD), "ahead"),
    FactProvider(
        "gap_trend",
        RaceQuery.GAP_TREND_AHEAD,
        _opponent_unsupported("ahead", RaceQuery.GAP_TREND_AHEAD),
        "ahead",
    ),
    FactProvider(
        "gap", RaceQuery.GAP_BEHIND, _opponent_gap("behind", RaceQuery.GAP_BEHIND), "behind"
    ),
    FactProvider(
        "gap_trend",
        RaceQuery.GAP_TREND_BEHIND,
        _opponent_unsupported("behind", RaceQuery.GAP_TREND_BEHIND),
        "behind",
    ),
)


class FactCatalog:
    """One registry drives capability projection, resolution, and live refresh."""

    def __init__(self, providers: Iterable[FactProvider] = DEFAULT_FACT_PROVIDERS) -> None:
        self.providers = tuple(providers)
        self._semantic = {provider.key: provider for provider in self.providers}
        self._race = {provider.race_query: provider for provider in self.providers}
        if len(self._semantic) != len(self.providers) or len(self._race) != len(self.providers):
            raise ValueError("fact providers must have unique semantic and race-query keys")
        if set(self._race) != set(RaceQuery):
            raise ValueError("fact catalog must define every race query exactly once")

    @staticmethod
    def _snapshot(context: RaceContext) -> RaceSnapshot:
        return RaceSnapshot(
            context=context,
            as_of=context.frame.observed_at,
            mode="replay",
        )

    def provider_for_part(self, part: QueryPart) -> FactProvider:
        reference = cast(
            ProviderReference,
            part.reference if part.reference in {"ahead", "behind"} else "none",
        )
        provider = self._semantic.get((part.query, reference))
        if provider is None:
            return self._race[RaceQuery.UNSUPPORTED]
        return provider

    def part_for_query(
        self,
        query: RaceQuery,
        *,
        part_id: str,
        field_relation: FieldRelation | None = None,
    ) -> QueryPart:
        provider = self._race[query]
        return QueryPart(
            part_id=part_id,
            query=provider.dialogue_query,
            reference=provider.reference,
            field_relation=field_relation if query is RaceQuery.FIELD_STATUS else None,
        )

    def resolve_part(
        self,
        snapshot: RaceSnapshot,
        part: QueryPart,
        opponent: OpponentReference | None = None,
    ) -> RaceAnswer:
        provider = self.provider_for_part(part)
        return provider.resolver(snapshot, part.field_relation, opponent)

    def resolve_query(
        self,
        context: RaceContext,
        query: RaceQuery,
        *,
        field_relation: FieldRelation | None = None,
    ) -> RaceAnswer:
        # Legacy factual conversation has no stable opponent identity; preserve its
        # aggregate nearest-gap behavior while CE-05 uses identity-bound resolution.
        if query is RaceQuery.GAP_AHEAD:
            return _answer(query, context.gap_ahead_s, "s")
        if query is RaceQuery.GAP_BEHIND:
            return _answer(query, context.gap_behind_s, "s")
        provider = self._race[query]
        return provider.resolver(self._snapshot(context), field_relation, None)

    def capabilities(
        self,
        snapshot: RaceSnapshot,
        opponents: tuple[OpponentReference, ...],
    ) -> tuple[QueryCapability, ...]:
        by_side = {opponent.side: opponent for opponent in opponents}
        capabilities: list[QueryCapability] = []
        for index, provider in enumerate(self.providers):
            part = self.part_for_query(
                provider.race_query,
                part_id=f"cap-{index}",
                field_relation="last" if provider.race_query is RaceQuery.FIELD_STATUS else None,
            )
            opponent = by_side.get(provider.reference) if provider.reference != "none" else None
            try:
                status = self.resolve_part(snapshot, part, opponent).status
            except FactIdentityChanged:
                status = "missing"
            capabilities.append(
                QueryCapability(
                    query=provider.dialogue_query,
                    reference=provider.reference,
                    status=status,
                )
            )
        return tuple(capabilities)


DEFAULT_FACT_CATALOG = FactCatalog()
