import asyncio
from datetime import UTC, datetime

import pytest

from race_engineer.core.contracts import SpeechIntent, Utterance
from race_engineer.core.enums import MessageCategory
from race_engineer.language import (
    DeterministicLanguageGenerator,
    FallbackLanguageGenerator,
    UnsupportedIntentError,
)


def make_intent(
    facts: dict[str, object],
    *,
    critical_template: str | None = None,
    language: str = "en",
    max_words: int = 20,
) -> SpeechIntent:
    return SpeechIntent(
        intent_id="intent-1",
        category=MessageCategory.SAFETY,
        facts=facts,  # type: ignore[arg-type]
        priority=95 if critical_template else 70,
        deadline=datetime(2026, 1, 1, tzinfo=UTC),
        critical_template=critical_template,
        language=language,
        max_words=max_words,
    )


def generate(intent: SpeechIntent) -> Utterance:
    return asyncio.run(DeterministicLanguageGenerator().generate(intent))


@pytest.mark.parametrize(
    ("facts", "expected", "template_id"),
    [
        ({"phase": "green"}, "Green flag", "phase.green"),
        ({"flag": "blue", "state": "shown"}, "Blue flag", "flag.blue"),
        ({"position": 3}, "Position 3", "situation.position"),
        ({"in_pit_lane": True}, "Pit lane", "situation.pit_entry"),
        ({"laps_remaining": 2.5}, "Fuel, 2.5 laps remaining", "strategy.fuel_laps"),
    ],
)
def test_deterministic_templates_use_only_approved_facts(
    facts: dict[str, object], expected: str, template_id: str
) -> None:
    utterance = generate(make_intent(facts))
    assert utterance.text == expected
    assert utterance.generator_metadata["template_id"] == template_id


def test_critical_template_bypasses_fact_rendering() -> None:
    utterance = generate(make_intent({}, critical_template="Caution, slow down"))
    assert utterance.text == "Caution, slow down"
    assert utterance.generator_metadata["template_id"] == "critical.fixed"


def test_generator_rejects_unknown_facts_languages_and_word_limit_violations() -> None:
    with pytest.raises(UnsupportedIntentError, match="no deterministic template"):
        generate(make_intent({"unknown": True}))
    with pytest.raises(UnsupportedIntentError, match="do not support language"):
        generate(make_intent({"phase": "green"}, language="tr"))
    with pytest.raises(UnsupportedIntentError, match="word intent limit"):
        generate(make_intent({"phase": "green"}, max_words=1))


class FailingGenerator:
    async def generate(self, intent: SpeechIntent) -> Utterance:
        del intent
        raise RuntimeError("primary failed")


def test_fallback_generator_uses_deterministic_template() -> None:
    generator = FallbackLanguageGenerator(
        primary=FailingGenerator(),
        fallback=DeterministicLanguageGenerator(),
        timeout_s=0.1,
    )

    utterance = asyncio.run(generator.generate(make_intent({"phase": "green"})))

    assert utterance.text == "Green flag"
    assert utterance.generator_metadata["adapter"] == "deterministic-template"
