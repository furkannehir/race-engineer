"""Fact-bound deterministic wording for approved speech intents."""

from typing import Final

from pydantic import JsonValue

from race_engineer.core.contracts import SpeechIntent, Utterance

_PHASE_TEMPLATES: Final[dict[str, str]] = {
    "formation": "Formation lap",
    "green": "Green flag",
    "caution": "Caution, slow down",
    "checkered": "Checkered flag",
}

_FLAG_TEMPLATES: Final[dict[str, str]] = {
    "green": "Green flag",
    "yellow": "Yellow flag",
    "red": "Red flag",
    "blue": "Blue flag",
    "white": "White flag",
    "black": "Black flag",
    "checkered": "Checkered flag",
}


class UnsupportedIntentError(ValueError):
    """Raised when approved facts have no deterministic wording template."""


def _number(value: JsonValue | None) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def _format_number(value: float) -> str:
    return f"{value:.1f}".rstrip("0").rstrip(".")


class DeterministicLanguageGenerator:
    """Renders only known, policy-approved facts into short English phrases."""

    @staticmethod
    def _render(intent: SpeechIntent) -> tuple[str, str]:
        if intent.critical_template is not None:
            return intent.critical_template, "critical.fixed"

        normalized_language = intent.language.lower()
        if normalized_language != "en" and not normalized_language.startswith("en-"):
            raise UnsupportedIntentError(
                f"deterministic templates do not support language {intent.language!r}"
            )

        phase = intent.facts.get("phase")
        if isinstance(phase, str) and phase in _PHASE_TEMPLATES:
            return _PHASE_TEMPLATES[phase], f"phase.{phase}"

        flag = intent.facts.get("flag")
        if isinstance(flag, str) and flag in _FLAG_TEMPLATES:
            return _FLAG_TEMPLATES[flag], f"flag.{flag}"

        position = intent.facts.get("position")
        if isinstance(position, int) and not isinstance(position, bool) and position >= 1:
            return f"Position {position}", "situation.position"

        in_pit_lane = intent.facts.get("in_pit_lane")
        if isinstance(in_pit_lane, bool):
            return (
                ("Pit lane", "situation.pit_entry")
                if in_pit_lane
                else ("Pit exit", "situation.pit_exit")
            )

        laps_remaining = _number(intent.facts.get("laps_remaining"))
        if laps_remaining is not None and laps_remaining >= 0:
            return (
                f"Fuel, {_format_number(laps_remaining)} laps remaining",
                "strategy.fuel_laps",
            )

        fuel_l = _number(intent.facts.get("fuel_l"))
        if fuel_l is not None and fuel_l >= 0:
            return f"Fuel, {_format_number(fuel_l)} liters", "strategy.fuel_liters"

        message = intent.facts.get("message")
        if isinstance(message, str) and message.strip():
            return message.strip(), "legacy.approved_message"

        raise UnsupportedIntentError(f"no deterministic template for intent {intent.intent_id!r}")

    async def generate(self, intent: SpeechIntent) -> Utterance:
        text, template_id = self._render(intent)
        word_count = len(text.split())
        if not text.strip():
            raise UnsupportedIntentError("a deterministic template produced empty text")
        if word_count > intent.max_words:
            raise UnsupportedIntentError(
                f"template {template_id!r} exceeds the {intent.max_words}-word intent limit"
            )
        return Utterance(
            intent_id=intent.intent_id,
            text=text,
            generator_metadata={
                "adapter": "deterministic-template",
                "adapter_version": "1",
                "template_id": template_id,
            },
        )
