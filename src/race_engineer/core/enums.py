"""Shared enumerations kept independent from simulator terminology."""

from enum import StrEnum


class SessionPhase(StrEnum):
    UNKNOWN = "unknown"
    FORMATION = "formation"
    GREEN = "green"
    CAUTION = "caution"
    CHECKERED = "checkered"


class RaceFlag(StrEnum):
    GREEN = "green"
    YELLOW = "yellow"
    RED = "red"
    BLUE = "blue"
    WHITE = "white"
    BLACK = "black"
    CHECKERED = "checkered"


class EventType(StrEnum):
    SESSION_PHASE_CHANGED = "session_phase_changed"
    FLAG_CHANGED = "flag_changed"
    POSITION_CHANGED = "position_changed"
    FUEL_THRESHOLD = "fuel_threshold"
    PIT_STATE_CHANGED = "pit_state_changed"
    CUSTOM = "custom"


class Urgency(StrEnum):
    ROUTINE = "routine"
    IMPORTANT = "important"
    CRITICAL = "critical"


class MessageCategory(StrEnum):
    SAFETY = "safety"
    SITUATION = "situation"
    STRATEGY = "strategy"
    SPOTTER = "spotter"
    PREFERENCE = "preference"


class Tone(StrEnum):
    NEUTRAL = "neutral"
    CALM = "calm"
    URGENT = "urgent"


class InterruptionPolicy(StrEnum):
    NEVER = "never"
    INTERRUPT_LOWER_PRIORITY = "interrupt_lower_priority"
    INTERRUPT_ANY = "interrupt_any"


class PolicyDecisionOutcome(StrEnum):
    APPROVED = "approved"
    SUPPRESSED = "suppressed"


class PolicyDecisionReason(StrEnum):
    APPROVED = "approved"
    DISABLED = "disabled"
    EXPIRED = "expired"
    DUPLICATE = "duplicate"
    COOLDOWN = "cooldown"
    SUPERSEDED = "superseded"


class PreferenceSource(StrEnum):
    CONFIG = "config"
    UI = "ui"
    VOICE = "voice"


class PreferenceScope(StrEnum):
    SESSION = "session"
    DRIVER = "driver"
    GLOBAL = "global"


class PlaybackStatus(StrEnum):
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    EXPIRED = "expired"
    FAILED = "failed"
