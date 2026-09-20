import asyncio
import json

import pytest
from test_stt_audio import tone

from race_engineer.config import SttConfig
from race_engineer.core.speech_input import AudioClip, SpeechInputError
from race_engineer.stt.qwen import QwenSpeechRecognizer


class FakeInput:
    def __init__(self):
        self.lines = []

    def write(self, value):
        self.lines.append(json.loads(value))

    async def drain(self):
        pass


class FakeOutput:
    def __init__(self, messages, *, hang=False):
        self.messages = iter(messages)
        self.hang = hang

    async def readline(self):
        if self.hang:
            await asyncio.sleep(10)
        value = next(self.messages, None)
        return json.dumps(value).encode() + b"\n" if value is not None else b""


class FakeProcess:
    def __init__(self, messages):
        self.stdin = FakeInput()
        self.stdout = FakeOutput(messages)
        self.returncode = None
        self.killed = False

    def kill(self):
        self.killed = True
        self.returncode = -9

    async def wait(self):
        return self.returncode


def runtime_config(tmp_path, **kwargs):
    python = tmp_path / "python.exe"
    python.touch()
    model = tmp_path / "model"
    model.mkdir()
    return SttConfig(python_path=python, model_path=model, **kwargs)


def test_silence_never_starts_a_worker(tmp_path):
    recognizer = QwenSpeechRecognizer(SttConfig(python_path=tmp_path / "missing"))
    result = asyncio.run(recognizer.transcribe(AudioClip(b"\0\0" * 16000)))
    assert result.status == "no_speech"


def test_persistent_worker_uses_offline_environment_and_cleans_up(tmp_path, monkeypatch):
    answer = {
        "result": {
            "status": "transcribed",
            "text": "Where are we?",
            "language": "en",
            "audio_duration_s": 0.4,
        }
    }
    worker = FakeProcess([{"ready": True}, answer, answer])
    calls = []

    async def create(*args, **kwargs):
        calls.append((args, kwargs))
        return worker

    monkeypatch.setattr("race_engineer.stt.qwen.asyncio.create_subprocess_exec", create)
    recognizer = QwenSpeechRecognizer(runtime_config(tmp_path))

    async def run():
        await recognizer.start()
        assert (await recognizer.transcribe(tone())).text == "Where are we?"
        assert (await recognizer.transcribe(tone())).text == "Where are we?"
        await recognizer.aclose()

    asyncio.run(run())
    assert len(calls) == 1
    assert calls[0][1]["env"]["HF_HUB_OFFLINE"] == "1"
    assert calls[0][1]["env"]["TRANSFORMERS_OFFLINE"] == "1"
    assert worker.killed
    assert worker.stdin.lines[0]["sample_rate_hz"] == 16000


def test_timeout_kills_worker_and_does_not_leave_inference_running(tmp_path, monkeypatch):
    worker = FakeProcess([{"ready": True}])

    async def create(*args, **kwargs):
        return worker

    monkeypatch.setattr("race_engineer.stt.qwen.asyncio.create_subprocess_exec", create)
    recognizer = QwenSpeechRecognizer(runtime_config(tmp_path, timeout_s=0.01))

    async def run():
        await recognizer.start()
        worker.stdout.hang = True
        with pytest.raises(SpeechInputError, match="stt_timeout"):
            await recognizer.transcribe(tone())
        await recognizer.aclose()

    asyncio.run(run())
    assert worker.killed


@pytest.mark.parametrize("response", [{"error": "private transcript"}, {"result": {}}, None])
def test_worker_failures_are_sanitized(tmp_path, monkeypatch, response):
    worker = FakeProcess([{"ready": True}, response])

    async def create(*args, **kwargs):
        return worker

    monkeypatch.setattr("race_engineer.stt.qwen.asyncio.create_subprocess_exec", create)
    recognizer = QwenSpeechRecognizer(runtime_config(tmp_path))
    with pytest.raises(SpeechInputError) as caught:
        asyncio.run(recognizer.transcribe(tone()))
    assert "private transcript" not in str(caught.value)
    assert worker.killed


def test_cancelled_inference_kills_worker(tmp_path, monkeypatch):
    worker = FakeProcess([{"ready": True}])

    async def create(*args, **kwargs):
        return worker

    monkeypatch.setattr("race_engineer.stt.qwen.asyncio.create_subprocess_exec", create)
    recognizer = QwenSpeechRecognizer(runtime_config(tmp_path))

    async def run():
        await recognizer.start()
        worker.stdout.hang = True
        task = asyncio.create_task(recognizer.transcribe(tone()))
        await asyncio.sleep(0.01)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(run())
    assert worker.killed
