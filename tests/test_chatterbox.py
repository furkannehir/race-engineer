import asyncio
import json
import threading
from contextlib import nullcontext
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError
from test_live_radio import submit
from test_stt_worker import FakeInput, FakeProcess

from race_engineer.application.live_conversation import voice_iracing
from race_engineer.config import AppConfig, TtsConfig
from race_engineer.core.contracts import Utterance
from race_engineer.core.conversation import ConversationReply
from race_engineer.core.enums import PlaybackStatus
from race_engineer.core.speech_output import SpeechOutputError, SpeechOutputResult
from race_engineer.tts.chatterbox import ChatterboxAnnouncementEngine, ChatterboxSpeechBackend
from race_engineer.tts.chatterbox_worker import (
    SpeakRequest,
    SpeechCancelled,
    check_cancelled,
    load_model,
    play_pcm,
    select_device,
    synthesize_pcm,
)
from race_engineer.tts.live_radio import LiveRadio
from race_engineer.tts_metadata import CHATTERBOX_VERSION, MODEL_FILES


def reply(language="en", text="Radio check."):
    return ConversationReply(
        language=language,
        text=text,
        status="answered",
        session_id="test",
        source_sequence=1,
        mode="live",
    )


def settings(tmp_path, **kwargs):
    python = tmp_path / "python.exe"
    python.touch()
    model = tmp_path / "model"
    model.mkdir()
    for name in MODEL_FILES:
        (model / name).touch()
    return TtsConfig(python_path=python, model_path=model, **kwargs)


def ready(device="cuda"):
    return {"ready": True, "device": device, "version": CHATTERBOX_VERSION}


def done(identifier, language="en", played=True):
    return {
        "id": identifier,
        "result": SpeechOutputResult(
            language=language,
            played=played,
            audio_duration_s=0.5,
            synthesis_ms=20,
            playback_ms=500 if played else 0,
        ).model_dump(),
    }


def install_process(monkeypatch, process):
    calls = []

    async def spawn(*args, **kwargs):
        calls.append((args, kwargs))
        return process

    monkeypatch.setattr("race_engineer.tts.chatterbox.start_owned_process", spawn)
    return calls


def test_one_worker_default_voice_for_replies_and_announcements(tmp_path, monkeypatch):
    process = FakeProcess(
        [ready(), {"id": 1, "prepared": True}, done(1), {"id": 2, "prepared": True}, done(2, "tr")]
    )
    calls = install_process(monkeypatch, process)
    backend = ChatterboxSpeechBackend(settings(tmp_path, output_device=7))
    engine = ChatterboxAnnouncementEngine(backend)
    private = "$(private); <not shell syntax>"

    async def run():
        await asyncio.gather(backend.start(), backend.start())
        assert (await backend.speak(reply(text=private))).played
        result = await engine.speak(
            Utterance(intent_id="flag", text="Sarı bayrak.", generator_metadata={"language": "tr"})
        )
        assert result.status is PlaybackStatus.COMPLETED
        assert engine.backend is backend
        assert not process.killed
        await backend.aclose()

    asyncio.run(run())
    assert len(calls) == 1 and process.killed
    args, kwargs = calls[0]
    assert private not in repr((args, kwargs)) and "shell" not in kwargs
    assert args[-2:] == ("--output-device", "7")
    assert kwargs["env"]["HF_HUB_OFFLINE"] == "1"
    assert kwargs["env"]["TRANSFORMERS_OFFLINE"] == "1"
    assert process.stdin.lines[0]["text"] == private
    assert process.stdin.lines[2]["language"] == "tr"
    assert not any(
        "prompt" in key or "reference" in key for line in process.stdin.lines for key in line
    )


def test_expired_audio_is_discarded_without_reloading_the_model(tmp_path, monkeypatch):
    process = FakeProcess(
        [
            ready(),
            {"id": 1, "prepared": True},
            {"id": 1, "cancelled": True},
            {"id": 1, "cancel_ack": True},
            done(2, played=False),
        ]
    )
    calls = install_process(monkeypatch, process)
    backend = ChatterboxSpeechBackend(settings(tmp_path))

    async def run():
        with pytest.raises(SpeechOutputError, match="tts_reply_expired"):
            await backend.speak(reply(), before_playback=lambda: False)
        assert not (await backend.speak(reply(), play_audio=False)).played
        assert not process.killed
        await backend.aclose()

    asyncio.run(run())
    assert len(calls) == 1
    assert process.stdin.lines[1] == {"command": "cancel", "id": 1}
    assert not any(line.get("command") == "play" for line in process.stdin.lines)


class InteractiveProcess(FakeProcess):
    def __init__(self):
        super().__init__([])
        self.messages = asyncio.Queue()
        self.messages.put_nowait(ready())
        self.started = asyncio.Event()
        self.finished = asyncio.Event()
        self.output_active = False
        self.maximum_outputs = 0
        self.stdout = self
        self.stdin = FakeInput()
        self.stdin.write = self.write

    def write(self, raw):
        message = json.loads(raw)
        self.stdin.lines.append(message)
        identifier = message["id"]
        if "text" in message:
            assert not self.output_active
            self.messages.put_nowait({"id": identifier, "prepared": True})
        elif message["command"] == "cancel":
            self.output_active = False
            self.messages.put_nowait({"id": identifier, "cancelled": True})
            self.messages.put_nowait({"id": identifier, "cancel_ack": True})
        elif message["command"] == "play":
            assert not self.output_active
            self.output_active = True
            self.maximum_outputs = max(self.maximum_outputs, 1)
            if identifier == 1:
                self.started.set()
            else:
                self.output_active = False
                self.messages.put_nowait(done(identifier))
                self.finished.set()

    async def readline(self):
        return json.dumps(await self.messages.get()).encode() + b"\n"


def test_critical_interrupt_preserves_worker_and_never_overlaps_playback(tmp_path, monkeypatch):
    async def run():
        process = InteractiveProcess()
        calls = install_process(monkeypatch, process)
        backend = ChatterboxSpeechBackend(settings(tmp_path))
        radio = LiveRadio(ChatterboxAnnouncementEngine(backend))
        answering = asyncio.create_task(radio.answer(lambda: backend.speak(reply()), 0, 10))
        await process.started.wait()
        assert await submit(radio, "critical", 100)
        assert await answering == "interrupted"
        await process.finished.wait()
        assert len(calls) == 1 and not process.killed
        assert process.maximum_outputs == 1
        assert [line.get("command") for line in process.stdin.lines] == [
            None,
            "play",
            "cancel",
            None,
            "play",
        ]
        await radio.aclose()
        await backend.aclose()
        assert process.killed

    asyncio.run(asyncio.wait_for(run(), 2))


@pytest.mark.parametrize("action", ["mute", "reset", "shutdown"])
def test_control_cancellation_stops_audio_before_releasing_radio(tmp_path, monkeypatch, action):
    async def run():
        process = InteractiveProcess()
        calls = install_process(monkeypatch, process)
        backend = ChatterboxSpeechBackend(settings(tmp_path))
        radio = LiveRadio(ChatterboxAnnouncementEngine(backend))
        answering = asyncio.create_task(radio.answer(lambda: backend.speak(reply()), 0, 10))
        await process.started.wait()
        if action == "mute":
            radio.set_muted(True)
        elif action == "reset":
            radio.reset(1)
        else:
            await radio.aclose()
        assert await answering != "completed"
        assert not process.output_active and len(calls) == 1
        if action != "shutdown":
            await radio.aclose()
        await backend.aclose()

    asyncio.run(asyncio.wait_for(run(), 2))


def test_failed_generation_is_visible_and_does_not_select_another_voice(tmp_path, monkeypatch):
    process = FakeProcess(
        [ready(), {"id": 1, "error": "tts_generation_or_playback_failed"}, done(2, played=False)]
    )
    calls = install_process(monkeypatch, process)
    notices = []
    backend = ChatterboxSpeechBackend(settings(tmp_path), notice=notices.append)

    async def run():
        engine = ChatterboxAnnouncementEngine(backend)
        assert (
            await engine.speak(Utterance(intent_id="flag", text="Green flag"))
        ).status is PlaybackStatus.FAILED
        assert not (await backend.speak(reply(), play_audio=False)).played
        await backend.aclose()

    asyncio.run(run())
    assert notices and "No voice fallback" in notices[0]
    assert len(calls) == 1


@pytest.mark.parametrize(
    "failure", ["tts_default_voice_missing", "tts_cuda_unavailable", "tts_model_load_failed"]
)
def test_initialization_failure_is_latched_and_reported(tmp_path, monkeypatch, failure):
    process = FakeProcess([{"error": failure}])
    calls = install_process(monkeypatch, process)
    notices = []
    backend = ChatterboxSpeechBackend(settings(tmp_path), notice=notices.append)

    async def run():
        for _ in range(2):
            with pytest.raises(SpeechOutputError):
                await backend.start()
        await backend.aclose()

    asyncio.run(run())
    assert len(calls) == 1 and process.killed and notices


def test_cpu_fallback_is_visible_and_frozen_worker_path_is_supported(tmp_path, monkeypatch):
    process = FakeProcess([ready("cpu")])
    calls = install_process(monkeypatch, process)
    executable = tmp_path / "PitwardTTSWorker.exe"
    executable.touch()
    notices = []
    backend = ChatterboxSpeechBackend(
        settings(tmp_path, worker_path=executable), notice=notices.append
    )

    async def run():
        await backend.start()
        assert backend.effective_device == "cpu"
        await backend.aclose()

    asyncio.run(run())
    assert "not guaranteed" in notices[0]
    assert calls[0][0][0] == str(executable.resolve())
    assert "-m" not in calls[0][0]


@pytest.mark.parametrize(
    "requested,fallback,expected",
    [
        ("auto", True, "cpu"),
        ("cpu", False, "cpu"),
        ("cuda", True, "cpu"),
    ],
)
def test_device_selection_without_cuda_is_explicit(requested, fallback, expected):
    torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
    device, warning = select_device(torch, requested, fallback)
    assert device == expected and warning
    with pytest.raises(RuntimeError, match="tts_cuda_unavailable"):
        select_device(torch, "cuda", False)


def test_usable_cuda_is_preferred():
    torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: True, synchronize=lambda: None),
        ones=lambda *args, **kwargs: SimpleNamespace(add_=lambda _: None),
    )
    assert select_device(torch, "auto", True) == ("cuda", None)


def test_default_voice_and_language_requirements_are_checked(tmp_path, monkeypatch):
    config = settings(tmp_path)
    model = SimpleNamespace(
        conds=object(), get_supported_languages=lambda: {"en": "English", "tr": "Turkish"}
    )
    calls = []

    def from_local(path, device):
        calls.append((path, device))
        return model

    torch = SimpleNamespace(set_num_threads=lambda _: None)
    api = SimpleNamespace(ChatterboxMultilingualTTS=SimpleNamespace(from_local=from_local))
    monkeypatch.setattr(
        "race_engineer.tts.chatterbox_worker.importlib.metadata.version",
        lambda _: CHATTERBOX_VERSION,
    )
    monkeypatch.setattr(
        "race_engineer.tts.chatterbox_worker.importlib.import_module",
        lambda name: torch if name == "torch" else api,
    )
    assert load_model(config.model_path, "cpu", False, 4)[0] is model
    assert calls == [(config.model_path, "cpu")]
    model.conds = None
    with pytest.raises(RuntimeError, match="tts_default_voice_missing"):
        load_model(config.model_path, "cpu", False, 4)


def test_cancelled_audio_is_not_opened(monkeypatch):
    cancelled = threading.Event()
    cancelled.set()
    with pytest.raises(SpeechCancelled):
        check_cancelled(cancelled)
    sd = SimpleNamespace(
        check_output_settings=lambda **kwargs: None,
        RawOutputStream=lambda **kwargs: pytest.fail("cancelled output opened"),
    )
    monkeypatch.setattr("race_engineer.tts.chatterbox_worker.importlib.import_module", lambda _: sd)
    with pytest.raises(SpeechCancelled):
        play_pcm(b"\0\0" * 10, 24000, 7, cancelled)


def test_removed_voice_settings_fail_explicitly_instead_of_falling_back():
    for value in (
        {"adapter": "piper"},
        {"adapter": "windows-sapi"},
        {"voice": "old"},
        {"english_model_path": "old.onnx"},
        {"length_scale": 1.0},
    ):
        with pytest.raises(ValidationError):
            TtsConfig.model_validate(value)
    with pytest.raises(ValidationError):
        AppConfig.model_validate({"radio_tts": {}})


@pytest.mark.parametrize("load_failure", [False, True])
def test_live_composition_shares_exact_backend_and_shuts_it_down(monkeypatch, load_failure):
    instances = []
    observations = []

    class Backend:
        def __init__(self, config, **kwargs):
            self.config, self.closed = config, False
            instances.append(self)

        async def start(self):
            if load_failure:
                raise SpeechOutputError("tts_model_missing")

        async def aclose(self):
            self.closed = True

    class Recognizer:
        def __init__(self, config):
            pass

        async def aclose(self):
            pass

    async def telemetry(*args, **kwargs):
        assert kwargs["live"].radio._engine.backend is instances[0]
        observations.append("automatic")
        await asyncio.sleep(0)

    async def dialogue(*args):
        assert args[5] is instances[0]
        observations.append("conversation")
        await asyncio.Event().wait()

    monkeypatch.setattr(
        "race_engineer.application.live_conversation.ChatterboxSpeechBackend", Backend
    )
    monkeypatch.setattr(
        "race_engineer.application.live_conversation.QwenSpeechRecognizer", Recognizer
    )
    monkeypatch.setattr("race_engineer.cli._read_iracing", telemetry)
    monkeypatch.setattr("race_engineer.application.live_conversation.live_dialogue", dialogue)
    asyncio.run(voice_iracing(None, app_config=AppConfig()))
    assert len(instances) == 1 and instances[0].closed
    assert set(observations) == {"automatic", "conversation"}


def test_automatic_deadline_is_checked_after_synthesis(tmp_path, monkeypatch):
    process = FakeProcess(
        [
            ready(),
            {"id": 1, "prepared": True},
            {"id": 1, "cancelled": True},
            {"id": 1, "cancel_ack": True},
        ]
    )
    install_process(monkeypatch, process)
    backend = ChatterboxSpeechBackend(settings(tmp_path))
    engine = ChatterboxAnnouncementEngine(backend)

    async def run():
        result = await engine.speak(
            Utterance(
                intent_id="old-position",
                text="Position five",
                generator_metadata={
                    "speech_deadline": (datetime.now(UTC) - timedelta(seconds=1)).isoformat()
                },
            )
        )
        assert result.status is PlaybackStatus.EXPIRED
        assert not process.killed
        await backend.aclose()

    asyncio.run(run())
    assert not any(line.get("command") == "play" for line in process.stdin.lines)


def test_synthesis_uses_builtin_voice_and_removes_interruption_hook(monkeypatch):
    numpy = pytest.importorskip("numpy")
    hooks = []
    removed = []
    calls = []

    def register(hook):
        hooks.append(hook)
        return SimpleNamespace(remove=lambda: removed.append(True))

    torch = SimpleNamespace(
        inference_mode=nullcontext,
        nn=SimpleNamespace(
            modules=SimpleNamespace(
                module=SimpleNamespace(register_module_forward_pre_hook=register)
            )
        ),
    )
    tensor = SimpleNamespace(
        detach=lambda: SimpleNamespace(
            cpu=lambda: SimpleNamespace(
                numpy=lambda: numpy.array([[0.0, 0.5, -0.5]], dtype="float32")
            )
        )
    )

    def generate(text, **kwargs):
        calls.append((text, kwargs))
        hooks[-1](None, None)
        return tensor

    original_import = __import__("importlib").import_module
    monkeypatch.setattr(
        "race_engineer.tts.chatterbox_worker.importlib.import_module",
        lambda name: torch if name == "torch" else original_import(name),
    )
    model = SimpleNamespace(sr=24000, generate=generate)
    for language in ("en", "tr"):
        pcm, rate = synthesize_pcm(
            model, SpeakRequest(id=1, text="Radio check", language=language), threading.Event(), 0.8
        )
        assert len(pcm) == 6 and rate == 24000
    assert [kwargs for _, kwargs in calls] == [{"language_id": "en"}, {"language_id": "tr"}]
    assert removed == [True, True]


def test_pinned_manifest_contains_exact_default_voice_assets():
    from race_engineer.components import load_manifest
    from race_engineer.tts_metadata import MODEL_RELATIVE, MODEL_REVISION

    _, artifacts = load_manifest()
    speech = [a for a in artifacts if a.artifact_id.startswith("chatterbox-")]
    assert {a.destination.name for a in speech} == set(MODEL_FILES)
    assert all(a.destination.parent == MODEL_RELATIVE and MODEL_REVISION in a.url for a in speech)
    assert "conds.pt" in {a.destination.name for a in speech}
    assert Path("src/race_engineer/stt/assets/radio-open.wav").is_file()
    assert Path("src/race_engineer/stt/assets/radio-close.wav").is_file()


def test_worker_protocol_accepts_cancel_during_generation_and_reuses_model(monkeypatch):
    import queue

    from race_engineer.tts.chatterbox_worker import serve

    async def run():
        inputs = queue.Queue()
        entered = threading.Event()
        stopped, finished = asyncio.Event(), asyncio.Event()
        calls, messages = [], []
        model = object()

        def synthesize(used_model, request, cancelled, volume):
            assert used_model is model
            calls.append((request.id, request.language))
            if request.id == 1:
                entered.set()
                assert cancelled.wait(1)
                raise SpeechCancelled()
            return b"\0\0" * 240, 24000

        def send(message):
            messages.append(message)
            if message.get("cancelled"):
                stopped.set()
            if "result" in message:
                finished.set()

        monkeypatch.setattr("race_engineer.tts.chatterbox_worker.synthesize_pcm", synthesize)
        monkeypatch.setattr("race_engineer.tts.chatterbox_worker.send", send)
        monkeypatch.setattr(
            "sys.stdin",
            SimpleNamespace(buffer=SimpleNamespace(readline=lambda limit: inputs.get(timeout=2))),
        )

        def command(value):
            inputs.put(json.dumps(value).encode() + b"\n")

        worker = asyncio.create_task(serve(model, volume=0.8, output_device=7))
        command({"id": 1, "text": "Private speech", "language": "en", "play_audio": False})
        assert await asyncio.to_thread(entered.wait, 1)
        command({"command": "cancel", "id": 1})
        await stopped.wait()
        command({"id": 2, "text": "Telsiz", "language": "tr", "play_audio": False})
        await finished.wait()
        inputs.put(b"")
        await worker
        assert calls == [(1, "en"), (2, "tr")]
        assert messages[0] == {"id": 1, "cancelled": True}
        assert messages[1] == {"id": 1, "cancel_ack": True}
        assert messages[2]["result"]["played"] is False

    asyncio.run(asyncio.wait_for(run(), 3))


def test_output_rate_fallback_keeps_the_selected_device(monkeypatch):
    numpy = pytest.importorskip("numpy")
    from race_engineer.tts import chatterbox_worker

    writes, streams = [], []

    class Stream:
        def __init__(self, **kwargs):
            streams.append(kwargs)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def write(self, pcm):
            writes.append(pcm)
            return False

    def check(**kwargs):
        assert kwargs["device"] == 20
        raise RuntimeError("unsupported native rate")

    def device(index, direction):
        assert index == 20 and direction == "output"
        return {"default_samplerate": 48000}

    sd = SimpleNamespace(check_output_settings=check, query_devices=device, RawOutputStream=Stream)
    librosa = SimpleNamespace(resample=lambda values, **kwargs: numpy.repeat(values, 2))
    original_import = __import__("importlib").import_module

    def modules(name):
        return (
            sd if name == "sounddevice" else librosa if name == "librosa" else original_import(name)
        )

    monkeypatch.setattr(chatterbox_worker.importlib, "import_module", modules)
    play_pcm(b"\0\0" * 240, 24000, 20, threading.Event())
    assert streams[0]["device"] == 20 and streams[0]["samplerate"] == 48000
    assert sum(len(pcm) for pcm in writes) == 960


def test_cancel_racing_terminal_result_drains_ack_before_next_request(tmp_path, monkeypatch):
    process = FakeProcess([ready(), done(1), {"id": 1, "cancel_ack": True}, done(2, played=False)])
    calls = install_process(monkeypatch, process)
    backend = ChatterboxSpeechBackend(settings(tmp_path))

    async def run():
        await backend.start()
        backend._sequence = 1
        await backend._cancel_request(1)
        assert not (await backend.speak(reply(), play_audio=False)).played
        assert not process.killed
        await backend.aclose()

    asyncio.run(run())
    assert len(calls) == 1


def test_cuda_load_oom_falls_back_to_same_default_model_on_cpu(tmp_path, monkeypatch):
    config = settings(tmp_path)
    calls = []
    model = SimpleNamespace(conds=object(), get_supported_languages=lambda: {"en": "", "tr": ""})

    def from_local(path, device):
        calls.append(device)
        if device == "cuda":
            raise MemoryError()
        return model

    torch = SimpleNamespace(
        set_num_threads=lambda _: None,
        cuda=SimpleNamespace(OutOfMemoryError=MemoryError, empty_cache=lambda: None),
    )
    api = SimpleNamespace(ChatterboxMultilingualTTS=SimpleNamespace(from_local=from_local))
    monkeypatch.setattr(
        "race_engineer.tts.chatterbox_worker.importlib.metadata.version",
        lambda _: CHATTERBOX_VERSION,
    )
    monkeypatch.setattr(
        "race_engineer.tts.chatterbox_worker.importlib.import_module",
        lambda name: torch if name == "torch" else api,
    )
    monkeypatch.setattr(
        "race_engineer.tts.chatterbox_worker.select_device", lambda *_: ("cuda", None)
    )
    actual, device, warning = load_model(config.model_path, "auto", True, 4)
    assert actual is model and device == "cpu" and warning
    assert calls == ["cuda", "cpu"]
