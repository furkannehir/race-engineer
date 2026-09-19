"""Deterministic event derivation from normalized iRacing frames."""

from collections.abc import Sequence
from datetime import timedelta

from pydantic import JsonValue

from race_engineer.core.contracts import RaceEvent, TelemetryFrame
from race_engineer.core.enums import EventType, RaceFlag, SessionPhase, Urgency


def _event_id(frame: TelemetryFrame, suffix: str) -> str:
    return f"{frame.session_id}:{frame.sequence}:{suffix}"


class IracingEventDeriver:
    def derive(
        self,
        previous: TelemetryFrame | None,
        current: TelemetryFrame,
    ) -> Sequence[RaceEvent]:
        if previous is None or previous.session_id != current.session_id:
            return ()

        events: list[RaceEvent] = []
        if previous.session_phase is not current.session_phase:
            urgency = (
                Urgency.CRITICAL
                if current.session_phase is SessionPhase.CAUTION
                else Urgency.IMPORTANT
            )
            events.append(
                RaceEvent(
                    event_id=_event_id(current, f"phase:{current.session_phase.value}"),
                    session_id=current.session_id,
                    source_sequence=current.sequence,
                    occurred_at=current.observed_at,
                    event_type=EventType.SESSION_PHASE_CHANGED,
                    facts={
                        "from": previous.session_phase.value,
                        "to": current.session_phase.value,
                    },
                    urgency=urgency,
                    expires_at=current.observed_at + timedelta(seconds=5),
                )
            )

        if previous.flags != current.flags:
            previous_flags = set(previous.flags)
            current_flags = set(current.flags)
            added = sorted(flag.value for flag in current_flags - previous_flags)
            removed = sorted(flag.value for flag in previous_flags - current_flags)
            added_facts: list[JsonValue] = list(added)
            removed_facts: list[JsonValue] = list(removed)
            critical = bool(
                current_flags.intersection({RaceFlag.YELLOW, RaceFlag.RED, RaceFlag.BLACK})
            )
            events.append(
                RaceEvent(
                    event_id=_event_id(current, "flags"),
                    session_id=current.session_id,
                    source_sequence=current.sequence,
                    occurred_at=current.observed_at,
                    event_type=EventType.FLAG_CHANGED,
                    facts={"added": added_facts, "removed": removed_facts},
                    urgency=Urgency.CRITICAL if critical else Urgency.IMPORTANT,
                    expires_at=current.observed_at + timedelta(seconds=3),
                )
            )

        previous_position = previous.player.position
        current_position = current.player.position
        if (
            previous_position is not None
            and current_position is not None
            and previous_position != current_position
        ):
            events.append(
                RaceEvent(
                    event_id=_event_id(current, f"position:{current_position}"),
                    session_id=current.session_id,
                    source_sequence=current.sequence,
                    occurred_at=current.observed_at,
                    event_type=EventType.POSITION_CHANGED,
                    facts={"from": previous_position, "to": current_position},
                    urgency=Urgency.ROUTINE,
                    expires_at=current.observed_at + timedelta(seconds=10),
                )
            )

        if previous.player.in_pit_lane != current.player.in_pit_lane:
            events.append(
                RaceEvent(
                    event_id=_event_id(current, f"pit:{int(current.player.in_pit_lane)}"),
                    session_id=current.session_id,
                    source_sequence=current.sequence,
                    occurred_at=current.observed_at,
                    event_type=EventType.PIT_STATE_CHANGED,
                    facts={"in_pit_lane": current.player.in_pit_lane},
                    urgency=Urgency.ROUTINE,
                    expires_at=current.observed_at + timedelta(seconds=5),
                )
            )

        return tuple(events)
