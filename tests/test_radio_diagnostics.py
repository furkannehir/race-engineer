import asyncio
import json
import logging

import pytest
from test_context_engine import ScriptedPlanner, driver_turn, evidence_query, plan, race_context
from test_intelligence_architecture import (
    ScriptedContextEngineer,
    brief,
    evidence,
    packet,
    response,
    run_pipeline,
    turn,
)
from test_live_conversation import context
from test_stt_audio import tone

from race_engineer.application.live_conversation import live_dialogue
from race_engineer.config import AppConfig, PrivacyConfig, SttConfig
from race_engineer.conversation.live import LiveRaceState
from race_engineer.core.intelligence import EvidenceReference, GroundedResponse
from race_engineer.core.speech_input import Transcription
from race_engineer.core.speech_output import SpeechOutputError, SpeechOutputResult
from race_engineer.intelligence.context_engine import QueryDrivenContextEngineer
from race_engineer.intelligence.local_model import LocalIntelligenceError, LocalJsonModel
from race_engineer.intelligence.telemetry_memory import BoundedTelemetryMemory
from race_engineer.observability import JsonFormatter
from race_engineer.radio_diagnostics import RadioTrace, radio_event, radio_stage
from race_engineer.tts.live_radio import LiveRadio


def records(caplog):
    formatter = JsonFormatter()
    return [
        json.loads(formatter.format(record))
        for record in caplog.records
        if record.name == "race_engineer.radio_diagnostics"
    ]


@pytest.mark.parametrize("record_text", [False, True])
def test_live_attempts_report_no_speech_timeouts_silence_and_tts_failures(
    monkeypatch, caplog, record_text
):
    caplog.set_level(logging.INFO)
    state = LiveRaceState(3)
    state.update(context())

    class Microphone:
        def __init__(self, config):
            self.clips = iter((tone(), tone(), tone(), tone(), tone(), None))

        async def next_clip(self, *, before_capture):
            before_capture()
            return next(self.clips)

        async def aclose(self):
            pass

    class Recognizer:
        calls = 0

        async def start(self):
            pass

        async def transcribe(self, audio):
            self.calls += 1
            if self.calls == 1:
                return Transcription(status="no_speech", audio_duration_s=1, reason="silence")
            return Transcription(
                status="transcribed",
                text="how's the race looking?",
                language="en",
                audio_duration_s=1,
            )

    class Intelligence:
        calls = 0

        async def prepare(self, turn):
            self.calls += 1
            if self.calls == 1:
                raise LocalIntelligenceError("context_model_timeout")
            return turn

        async def ground(self, turn):
            silent = self.calls == 2
            return GroundedResponse(
                response_id=f"{turn.turn_id}:response",
                turn_id=turn.turn_id,
                session_id=turn.session_id,
                generation=turn.generation,
                source_sequence=state.snapshot().context.frame.sequence,
                language="en",
                action="silence" if silent else "speak",
                text=None if silent else "We're P25.",
            )

    class Speaker:
        calls = 0

        async def start(self):
            pass

        async def speak(self, reply, *, before_playback):
            self.calls += 1
            if self.calls == 1:
                raise SpeechOutputError("tts_reply_expired")
            assert before_playback()
            return SpeechOutputResult(
                language="en",
                played=True,
                audio_duration_s=1,
                synthesis_ms=2,
                playback_ms=3,
            )

    monkeypatch.setattr(
        "race_engineer.application.live_conversation.PushToTalkMicrophone", Microphone
    )

    async def run():
        radio = LiveRadio(None)
        try:
            await live_dialogue(
                AppConfig(privacy=PrivacyConfig(record_radio_text=record_text)),
                SttConfig(),
                state,
                radio,
                Recognizer(),
                Speaker(),
                None,
                intelligence=Intelligence(),
            )
        finally:
            await radio.aclose()

    asyncio.run(run())
    observed = records(caplog)
    terminal = [record for record in observed if record["event"] == "radio_turn_finished"]
    assert [record["outcome"] for record in terminal] == [
        "no_question",
        "intelligence_failed",
        "silence",
        "expired",
        "spoken",
        "capture_ended",
    ]
    assert terminal[1]["reason"] == "context_model_timeout"
    assert len({record["trace_id"] for record in terminal}) == 6
    assert len({record["run_id"] for record in terminal}) == 1
    for completed in terminal:
        matching = [record for record in observed if record["trace_id"] == completed["trace_id"]]
        assert len([record for record in matching if record["event"] == "radio_turn_finished"]) == 1
    spoken = [record for record in observed if record["trace_id"] == terminal[4]["trace_id"]]
    assert any(record["event"] == "radio_tts_result" and record["played"] for record in spoken)
    assert any(record["event"] == "radio_delivery_started" for record in spoken)
    serialized = json.dumps(observed)
    assert ("how's the race looking?" in serialized) == record_text
    assert ("We're P25." in serialized) == record_text
    assert "pcm16" not in serialized


def test_pipeline_logs_selection_and_refreshed_evidence_without_values(caplog):
    caplog.set_level(logging.INFO)
    initial = packet(evidence("position", "position", 6, unit="position"))
    latest = packet(evidence("position", "position", 5, sequence=11, unit="position"), sequence=11)
    trace = RadioTrace("run", "trace", record_text=False, record_values=False)
    with trace.bind():
        result = run_pipeline(
            turn("Position?"),
            ScriptedContextEngineer(initial, latest),
            brief("inform", "position"),
            response(
                "P{{position}}.",
                EvidenceReference(
                    placeholder="position",
                    evidence_id="position",
                    field="value",
                ),
            ),
        )
    assert result.text == "P5."
    observed = records(caplog)
    refreshed = next(record for record in observed if record["event"] == "radio_evidence_refreshed")
    assert refreshed["source_sequence"] == 11
    assert refreshed["evidence"][0]["metric"] == "position"
    assert "value" not in refreshed["evidence"][0]
    assert any(
        record["event"] == "radio_core_decision" and record["goal"] == "inform"
        for record in observed
    )
    assert "Position?" not in json.dumps(observed)


def test_concurrent_model_requests_keep_trace_and_stage_ids_and_omit_provider_content(
    monkeypatch, caplog
):
    caplog.set_level(logging.INFO)
    model = LocalJsonModel(AppConfig().conversation)

    def fake_request(**kwargs):
        return {"private": "provider secret"}, {"prompt_tokens": 100, "completion_tokens": 20}

    monkeypatch.setattr(model, "_request", fake_request)

    async def request(trace_id):
        with RadioTrace("run", trace_id).bind(), radio_stage(trace_id):
            await model.request(system_prompt="private prompt", content="private driver", schema={})

    async def run():
        await asyncio.gather(request("context"), request("core"))
        radio_event("must_not_leak_scope")

    asyncio.run(run())
    observed = records(caplog)
    finished = [record for record in observed if record["event"] == "radio_model_finished"]
    assert {(record["trace_id"], record["stage"]) for record in finished} == {
        ("context", "context"),
        ("core", "core"),
    }
    assert all(record["prompt_tokens"] == 100 and record["duration_ms"] >= 0 for record in finished)
    assert "provider secret" not in json.dumps(observed)
    assert "private prompt" not in json.dumps(observed)
    assert "private driver" not in json.dumps(observed)
    assert not any(record["event"] == "must_not_leak_scope" for record in observed)


def test_context_plan_queries_are_traceable_without_logging_driver_text(caplog):
    caplog.set_level(logging.INFO)
    memory = BoundedTelemetryMemory()
    memory.update(race_context(1, 0))
    context_engineer = QueryDrivenContextEngineer(
        memory,
        ScriptedPlanner(plan(evidence_query("position", "position"))),
    )
    with RadioTrace("run", "plan").bind():
        asyncio.run(context_engineer.analyze(driver_turn("Where are we?")))
    selection = next(
        record for record in records(caplog) if record["event"] == "radio_context_plan"
    )
    assert selection["queries"][0]["selector"] == {
        "source": "player",
        "signal": "position",
        "subject_id": None,
    }
    assert selection["queries"][0]["operation"] == "latest"
    assert "Where are we?" not in json.dumps(records(caplog))


def test_model_failure_records_the_stage_and_sanitized_reason(caplog, monkeypatch):
    caplog.set_level(logging.INFO)
    model = LocalJsonModel(AppConfig().conversation)

    def failed_request(**kwargs):
        raise LocalIntelligenceError("model_timeout")

    monkeypatch.setattr(model, "_request", failed_request)

    async def run():
        with RadioTrace("run", "failed").bind(), radio_stage("context_planner"):
            await model.request(system_prompt="private", content="private", schema={})

    with pytest.raises(LocalIntelligenceError, match="model_timeout"):
        asyncio.run(run())
    observed = records(caplog)
    failed = next(record for record in observed if record["event"] == "radio_model_finished")
    assert failed["stage"] == "context_planner" and failed["reason"] == "model_timeout"
    assert failed["outcome"] == "failed" and failed["duration_ms"] >= 0
    assert "private" not in json.dumps(observed)
