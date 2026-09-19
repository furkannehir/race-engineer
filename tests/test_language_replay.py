import asyncio
from pathlib import Path

from race_engineer.config import load_config
from race_engineer.fixtures import load_fixture
from race_engineer.language import language_factory

ROOT = Path(__file__).parents[1]


def test_m3_language_replay_matches_expected_utterances() -> None:
    fixture = load_fixture(ROOT / "fixtures" / "synthetic" / "m3_green_flag")
    config = load_config(ROOT / "config" / "default.toml", environ={})
    generator = language_factory(config.language)
    utterances = tuple(
        asyncio.run(generator.generate(intent)) for intent in fixture.expected_intents
    )

    assert utterances == fixture.expected_utterances
