import asyncio
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_chatterbox import done, install_process, ready, reply, settings
from test_stt_worker import FakeProcess

from race_engineer.core.speech_output import SpeechOutputError
from race_engineer.tts.chatterbox import ChatterboxSpeechBackend
from race_engineer.tts.chatterbox_worker import load_model
from race_engineer.tts_metadata import (
    CHATTERBOX_VERSION,
    V3_CHECKPOINT,
    V3_CHECKPOINT_SHA256,
    model_files,
)


def v3_settings(tmp_path):
    config = settings(tmp_path).model_copy(update={"model_variant": "v3"})
    (config.model_path / V3_CHECKPOINT).touch()
    return config


def test_v3_downloads_match_selected_worker_files_and_preserve_v2_manifest():
    spec = importlib.util.spec_from_file_location(
        "tts_setup", Path(__file__).parents[1] / "scripts/setup_radio_tts.py"
    )
    setup = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(setup)
    _, manifest = setup.load_manifest()
    assert setup.speech_artifacts("v2") == tuple(
        item for item in manifest if item.artifact_id.startswith("chatterbox-")
    )
    artifacts = setup.speech_artifacts("v3")
    assert {item.destination.name for item in artifacts} == set(model_files("v3"))
    checkpoint = next(item for item in artifacts if item.destination.name == V3_CHECKPOINT)
    assert checkpoint.sha256 == V3_CHECKPOINT_SHA256
    assert "/resolve/" in checkpoint.url and checkpoint.url.endswith(V3_CHECKPOINT)
    assert "t3_mtl23ls_v2.safetensors" not in model_files("v3")


def test_v3_loader_selects_checkpoint_explicitly_and_keeps_default_voice(tmp_path, monkeypatch):
    config = v3_settings(tmp_path)
    calls = []
    model = SimpleNamespace(conds=object(), get_supported_languages=lambda: {"en": "", "tr": ""})

    def from_local(path, device, t3_model=None):
        calls.append((path, device, t3_model))
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
    monkeypatch.setattr(
        "race_engineer.tts.chatterbox_worker.select_device", lambda *_: ("cpu", None)
    )
    actual, device, _ = load_model(config.model_path, "cpu", False, 4, model_variant="v3")
    assert actual is model and device == "cpu"
    assert calls == [(config.model_path, "cpu", "v3")]

    def old_from_local(path, device):
        pytest.fail("An unsupported runtime must not silently load V2")

    api.ChatterboxMultilingualTTS.from_local = old_from_local
    with pytest.raises(RuntimeError, match="tts_model_variant_unsupported"):
        load_model(config.model_path, "cpu", False, 4, model_variant="v3")


def test_v3_backend_checks_loaded_variant_and_reuses_worker_for_both_languages(
    tmp_path, monkeypatch, caplog
):
    process = FakeProcess(
        [ready() | {"model_variant": "v3"}, done(1, played=False), done(2, "tr", played=False)]
    )
    calls = install_process(monkeypatch, process)
    backend = ChatterboxSpeechBackend(v3_settings(tmp_path))

    async def run():
        with caplog.at_level("INFO"):
            await backend.start()
        await backend.speak(reply(), play_audio=False)
        await backend.speak(reply("tr"), play_audio=False)
        await backend.aclose()

    asyncio.run(run())
    assert len(calls) == 1
    command = calls[0][0]
    assert command[command.index("--model-variant") + 1] == "v3"
    assert (
        next(r for r in caplog.records if getattr(r, "event", None) == "tts_ready").model_variant
        == "v3"
    )


@pytest.mark.parametrize("reported", [None, "v2"])
def test_v3_request_cannot_accept_legacy_v2_worker(tmp_path, monkeypatch, reported):
    message = ready()
    if reported is not None:
        message["model_variant"] = reported
    process = FakeProcess([message])
    install_process(monkeypatch, process)
    backend = ChatterboxSpeechBackend(v3_settings(tmp_path))
    with pytest.raises(SpeechOutputError, match="tts_model_variant_mismatch"):
        asyncio.run(backend.start())
    assert process.killed
