import asyncio
from pathlib import Path

from race_engineer.application import PipelineResult, RaceEngineerApplication
from race_engineer.fixtures import load_fixture
from race_engineer.testing import (
    EchoPolicy,
    InMemoryContextBuilder,
    RecordingTextToSpeechEngine,
    ScriptedEventDeriver,
    SequenceTelemetryAdapter,
    TemplateLanguageGenerator,
)

ROOT = Path(__file__).parents[1]


def test_synthetic_replay_is_deterministic_end_to_end() -> None:
    bundle = load_fixture(ROOT / "fixtures" / "synthetic" / "green_flag")
    tts = RecordingTextToSpeechEngine()
    app = RaceEngineerApplication(
        telemetry=SequenceTelemetryAdapter(bundle.frames),
        event_deriver=ScriptedEventDeriver({1: bundle.expected_events}),
        context_builder=InMemoryContextBuilder(),
        policy=EchoPolicy(),
        language=TemplateLanguageGenerator(),
        tts=tts,
    )

    result = asyncio.run(app.run())

    assert result == PipelineResult(frames=2, events=1, intents=1, utterances=1, playbacks=1)
    assert [utterance.text for utterance in tts.utterances] == ["Green flag"]
    assert tts.utterances[0].intent_id == bundle.expected_intents[0].intent_id
