import asyncio
from types import SimpleNamespace

import pytest
from pydantic import ValidationError
from test_stt_worker import FakeProcess

from race_engineer.config import RadioTtsConfig
from race_engineer.core.conversation import ConversationReply
from race_engineer.core.speech_output import SpeechOutputError, SpeechOutputResult
from race_engineer.tts.piper import PiperConversationSpeaker
from race_engineer.tts.piper_worker import SpeakRequest, render_request, synthesize_pcm


def reply(language="en", text="You're P6 overall."):
    return ConversationReply(
        language=language,
        text=text,
        status="answered",
        session_id="test",
        source_sequence=1,
        mode="replay",
    )


def response(language="en", played=True):
    return {
        "result": SpeechOutputResult(
            language=language,
            played=played,
            audio_duration_s=0.5,
            synthesis_ms=20,
            playback_ms=500 if played else 0,
        ).model_dump()
    }


def settings(tmp_path, **overrides):
    python = tmp_path / "python.exe"
    python.touch()
    english = tmp_path / "en.onnx"
    turkish = tmp_path / "tr.onnx"
    for path in (english, turkish):
        path.touch()
        path.with_suffix(".onnx.json").touch()
    return RadioTtsConfig(
        python_path=python, english_model_path=english, turkish_model_path=turkish, **overrides
    )


def test_worker_reused_for_both_languages_and_text_is_only_pipe_data(tmp_path, monkeypatch):
    worker = FakeProcess([{"ready": True}, response(), response("tr")])
    calls = []

    async def create(*args, **kwargs):
        calls.append((args, kwargs))
        return worker

    monkeypatch.setattr("race_engineer.tts.piper.asyncio.create_subprocess_exec", create)
    speaker = PiperConversationSpeaker(settings(tmp_path, output_device=7))
    private_text = "$(private) 'quoted'; <not shell syntax>"

    async def run():
        await speaker.start()
        assert (await speaker.speak(reply(text=private_text))).played
        assert (await speaker.speak(reply("tr", "Genel sıralamada 6. sıradayız."))).language == "tr"
        await speaker.aclose()

    asyncio.run(run())
    assert len(calls) == 1 and worker.killed
    args, kwargs = calls[0]
    assert private_text not in repr((args, kwargs))
    assert args[-2:] == ("--output-device", "7")
    assert worker.stdin.lines[0]["text"] == private_text
    assert worker.stdin.lines[1]["language"] == "tr"
    assert "shell" not in kwargs


@pytest.mark.parametrize(
    "bad",
    [
        {"error": "private response text"},
        {"result": {}},
        None,
        response("tr"),
        response(played=False),
    ],
)
def test_worker_failure_is_sanitized_and_killed(tmp_path, monkeypatch, bad):
    worker = FakeProcess([{"ready": True}, bad])

    async def create(*args, **kwargs):
        return worker

    monkeypatch.setattr("race_engineer.tts.piper.asyncio.create_subprocess_exec", create)
    speaker = PiperConversationSpeaker(settings(tmp_path))
    with pytest.raises(SpeechOutputError) as caught:
        asyncio.run(speaker.speak(reply()))
    assert "private response text" not in str(caught.value)
    assert worker.killed


def test_timeout_kills_worker_and_next_reply_restarts(tmp_path, monkeypatch):
    stuck = FakeProcess([{"ready": True}])
    fresh = FakeProcess([{"ready": True}, response()])
    workers = iter((stuck, fresh))

    async def create(*args, **kwargs):
        return next(workers)

    monkeypatch.setattr("race_engineer.tts.piper.asyncio.create_subprocess_exec", create)
    speaker = PiperConversationSpeaker(settings(tmp_path, playback_timeout_s=0.01))

    async def run():
        await speaker.start()
        stuck.stdout.hang = True
        with pytest.raises(SpeechOutputError, match="tts_playback_timeout"):
            await speaker.speak(reply())
        assert stuck.killed
        assert (await speaker.speak(reply())).played
        await speaker.aclose()

    asyncio.run(run())
    assert fresh.killed


def test_startup_timeout_kills_worker(tmp_path, monkeypatch):
    worker = FakeProcess([])
    worker.stdout.hang = True

    async def create(*args, **kwargs):
        return worker

    monkeypatch.setattr("race_engineer.tts.piper.asyncio.create_subprocess_exec", create)
    speaker = PiperConversationSpeaker(settings(tmp_path, startup_timeout_s=0.01))
    with pytest.raises(SpeechOutputError, match="tts_startup_timeout"):
        asyncio.run(speaker.start())
    assert worker.killed


def test_cancellation_stops_inflight_audio(tmp_path, monkeypatch):
    worker = FakeProcess([{"ready": True}])

    async def create(*args, **kwargs):
        return worker

    monkeypatch.setattr("race_engineer.tts.piper.asyncio.create_subprocess_exec", create)
    speaker = PiperConversationSpeaker(settings(tmp_path))

    async def run():
        await speaker.start()
        worker.stdout.hang = True
        task = asyncio.create_task(speaker.speak(reply()))
        await asyncio.sleep(0.01)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await speaker.aclose()

    asyncio.run(run())
    assert worker.killed


def test_missing_runtime_or_voice_fails_before_subprocess(tmp_path):
    with pytest.raises(SpeechOutputError, match="tts_runtime_missing"):
        asyncio.run(PiperConversationSpeaker(RadioTtsConfig(python_path=tmp_path / "no")).start())
    config = settings(tmp_path).model_copy(update={"turkish_model_path": tmp_path / "missing.onnx"})
    with pytest.raises(SpeechOutputError, match="tts_voice_missing"):
        asyncio.run(PiperConversationSpeaker(config).start())


class Voice:
    config = SimpleNamespace(sample_rate=16000)

    def __init__(self, chunks=None):
        self.chunks = chunks if chunks is not None else [chunk()]
        self.texts = []

    def synthesize(self, text, syn_config):
        self.texts.append(text)
        return iter(self.chunks)


def chunk(**overrides):
    values = {
        "sample_rate": 16000,
        "sample_width": 2,
        "sample_channels": 1,
        "audio_int16_bytes": b"\x01\x00" * 8000,
    }
    return SimpleNamespace(**(values | overrides))


@pytest.mark.parametrize(
    "chunks",
    [
        [],
        [chunk(sample_channels=2)],
        [chunk(sample_rate=22050)],
        [chunk(sample_width=4)],
        [chunk(audio_int16_bytes=b"x")],
        [chunk(audio_int16_bytes=b"\0\0" * (16000 * 60 + 1))],
    ],
)
def test_invalid_or_overlong_audio_never_reaches_playback(chunks):
    with pytest.raises(ValueError):
        synthesize_pcm(Voice(chunks), "Hello", None)


def test_voice_selection_and_no_playback_smoke_mode(monkeypatch):
    voices = {"en": Voice(), "tr": Voice()}
    played = []
    monkeypatch.setattr(
        "race_engineer.tts.piper_worker.play_pcm", lambda *args: played.append(args)
    )
    request = SpeakRequest(text="Altıncıyız.", language="tr", play_audio=False)
    result = render_request(request, voices, None, 7)
    assert result.language == "tr" and not result.played and not played
    assert result.audio_duration_s == 0.5 and result.playback_ms == 0
    assert voices["tr"].texts == ["Altıncıyız."] and not voices["en"].texts
    render_request(SpeakRequest(text="Position six.", language="en"), voices, None, 7)
    assert played[0][1:] == (16000, 7)


@pytest.mark.parametrize(
    "values",
    [
        {"text": "", "language": "en"},
        {"text": "x" * 1501, "language": "en"},
        {"text": "Bonjour", "language": "fr"},
    ],
)
def test_request_bounds(values):
    with pytest.raises(ValidationError):
        SpeakRequest.model_validate(values)


@pytest.mark.parametrize(
    "values",
    [
        {"output_device": -1},
        {"volume": 1.1},
        {"length_scale": 0},
        {"playback_timeout_s": float("nan")},
        {"threads": 0},
    ],
)
def test_radio_configuration_bounds(values):
    with pytest.raises(ValidationError):
        RadioTtsConfig(**values)


@pytest.mark.parametrize("allowed", [True, False])
def test_live_playback_requires_freshness_ack_after_synthesis(tmp_path, monkeypatch, allowed):
    worker = FakeProcess([{"ready": True}, {"prepared": True}, response()])

    async def create(*args, **kwargs):
        return worker

    monkeypatch.setattr("race_engineer.tts.piper.asyncio.create_subprocess_exec", create)
    speaker = PiperConversationSpeaker(settings(tmp_path))
    guards = []

    def guard():
        guards.append(True)
        return allowed

    async def run():
        if allowed:
            assert (await speaker.speak(reply(), before_playback=guard)).played
            assert worker.stdin.lines[-1] == {"play": True}
        else:
            with pytest.raises(SpeechOutputError, match="tts_reply_expired"):
                await speaker.speak(reply(), before_playback=guard)
            assert len(worker.stdin.lines) == 1  # No playback acknowledgement.
            assert worker.killed
        await speaker.aclose()

    asyncio.run(run())
    assert guards == [True]
    assert worker.stdin.lines[0]["defer_playback"] is True
