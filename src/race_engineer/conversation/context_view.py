"""Stateless projection of race evidence into bounded, immutable judge input."""

from collections import Counter
from datetime import datetime

from race_engineer.config import DialogueConfig
from race_engineer.conversation.facts import DEFAULT_FACT_CATALOG, FactCatalog
from race_engineer.core.conversation import RaceSnapshot, RadioLanguage
from race_engineer.core.dialogue import (
    ConversationContext,
    DialogueState,
    DialogueTurnInput,
    EventSummary,
    OpponentReference,
)


def current_opponents(snapshot: RaceSnapshot) -> tuple[OpponentReference, ...]:
    """Only unambiguous nearest normalized identities qualify as anchors."""
    opponents = snapshot.context.frame.opponents
    counts = Counter(opponent.driver_id for opponent in opponents)
    selected: list[OpponentReference] = []
    for side in ("ahead", "behind"):
        candidates = [
            opponent
            for opponent in opponents
            if opponent.gap_to_player_s is not None
            and (opponent.gap_to_player_s < 0 if side == "ahead" else opponent.gap_to_player_s > 0)
        ]
        candidates.sort(key=lambda opponent: abs(opponent.gap_to_player_s or 0))
        if not candidates:
            continue
        nearest = candidates[0]
        if counts[nearest.driver_id] != 1 or (
            len(candidates) > 1 and nearest.gap_to_player_s == candidates[1].gap_to_player_s
        ):
            continue
        selected.append(OpponentReference(side=side, driver_id=nearest.driver_id))
    return tuple(selected)


def fresh(snapshot: RaceSnapshot, config: DialogueConfig) -> bool:
    age = (snapshot.as_of - snapshot.context.frame.observed_at).total_seconds()
    return 0 <= age <= config.max_snapshot_age_s


class ConversationContextAssembler:
    def __init__(
        self,
        config: DialogueConfig | None = None,
        fact_catalog: FactCatalog = DEFAULT_FACT_CATALOG,
    ) -> None:
        self._config = config or DialogueConfig()
        self._facts = fact_catalog

    def assemble(
        self,
        snapshot: RaceSnapshot,
        state: DialogueState,
        *,
        turn_id: str,
        question: str,
        received_at: datetime,
        deadline: datetime,
        asr_language: RadioLanguage | None = None,
        reply_language: RadioLanguage | None = None,
    ) -> DialogueTurnInput:
        if not fresh(snapshot, self._config):
            raise ValueError("stale_snapshot")
        opponents = current_opponents(snapshot)
        capabilities = self._facts.capabilities(snapshot, opponents)
        events = tuple(
            EventSummary(
                event_id=event.event_id,
                category=event.event_type.value,
                occurred_at=event.occurred_at,
            )
            for event in snapshot.context.recent_events
            if event.occurred_at <= snapshot.as_of
            and event.source_sequence <= snapshot.context.frame.sequence
            and (event.expires_at is None or event.expires_at > snapshot.as_of)
        )
        events = events[-self._config.event_limit :] if self._config.event_limit else ()
        battle = snapshot.context.battle_state
        if battle not in {"clear", "attacking", "defending", "sandwiched", "contested"}:
            battle = None
        context = ConversationContext.model_validate(
            {
                "session_id": snapshot.context.frame.session_id,
                "generation": state.generation,
                "source_sequence": snapshot.context.frame.sequence,
                "observed_at": snapshot.context.frame.observed_at,
                "as_of": snapshot.as_of,
                "capabilities": capabilities,
                "opponents": opponents,
                "recent_events": events,
                "battle_state": battle,
            }
        )
        return DialogueTurnInput(
            turn_id=turn_id,
            question=question.strip(),
            received_at=received_at,
            deadline=deadline,
            asr_language=asr_language,
            reply_language=reply_language,
            default_language=(
                state.history[-1].language if state.history else self._config.default_language
            ),
            context=context,
            dialogue=state,
        )
