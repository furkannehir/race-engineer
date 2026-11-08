"""Device-boundary checks without installing torch into the host application."""

import asyncio
from types import SimpleNamespace

import pytest
from test_chatterbox import install_process, ready, settings
from test_stt_worker import FakeProcess

from race_engineer.core.speech_output import SpeechOutputError
from race_engineer.tts.chatterbox import ChatterboxSpeechBackend
from race_engineer.tts.default_voice import place_default_voice_cuda


class Tensor:
    def __init__(self, device="cpu", dtype="fp32"):
        self.device = SimpleNamespace(type=device)
        self.dtype = dtype

    def to(self, *, device=None, dtype=None):
        return Tensor(device or self.device.type, dtype or self.dtype)


class Module:
    def __init__(self, name, calls):
        self.name, self.calls = name, calls
        self.weight = Tensor()

    def parameters(self):
        return iter([self.weight])

    def to(self, **kwargs):
        self.calls.append((self.name, kwargs))
        self.weight = self.weight.to(**kwargs)
        return self


class Decoder:
    def __init__(self, calls):
        self.children = {
            name: Module(name, calls)
            for name in ("tokenizer", "speaker_encoder", "flow", "mel2wav")
        }
        for name, child in self.children.items():
            setattr(self, name, child)
        self.trim_fade = Tensor()

    def named_children(self):
        return iter(self.children.items())

    def named_buffers(self, *, recurse):
        assert recurse is False
        return iter([("trim_fade", self.trim_fade)])

    def parameters(self, *, recurse):
        assert recurse is False
        return iter([])

    @property
    def device(self):
        return next(self.tokenizer.parameters()).device

    @property
    def dtype(self):
        return next(self.flow.parameters()).dtype


def voice_model(calls):
    conds = SimpleNamespace(t3=Module("t3_cond", calls))
    conds.to = lambda device: calls.append(("conds", device))
    return SimpleNamespace(
        t3=Module("t3", calls), s3gen=Decoder(calls), ve=Module("ve", calls), conds=conds
    )


@pytest.mark.parametrize("precision", ["bf16", "fp32"])
def test_cached_voice_keeps_reference_weights_on_cpu_and_generation_on_cuda(precision):
    calls = []
    model = voice_model(calls)
    original_decoder_type = type(model.s3gen)
    place_default_voice_cuda(
        model, SimpleNamespace(bfloat16="bf16", float32="fp32"), precision=precision
    )
    assert model.device == model.s3gen.device.type == "cuda"
    assert model.t3.weight.device.type == "cuda" and model.t3.weight.dtype == precision
    assert model.s3gen.dtype == "fp32"
    for name in ("flow", "mel2wav"):
        weight = getattr(model.s3gen, name).weight
        assert weight.device.type == "cuda" and weight.dtype == "fp32"
    for module in (model.s3gen.tokenizer, model.s3gen.speaker_encoder, model.ve):
        assert module.weight.device.type == "cpu" and module.weight.dtype == "fp32"
        assert not any(name == module.name for name, _ in calls)
    assert model.s3gen.trim_fade.device.type == "cuda"
    assert model.s3gen.trim_fade.dtype == "fp32"
    assert original_decoder_type(calls).device.type == "cpu"  # No global class patch.
    assert calls[-1] == ("conds", "cuda")
    if precision == "bf16":
        assert calls[:3] == [
            ("t3", {"dtype": "bf16"}),
            ("t3_cond", {"dtype": "bf16"}),
            ("t3", {"device": "cuda"}),
        ]
    with pytest.raises(RuntimeError, match="tts_reference_audio_unsupported"):
        model.s3gen.embed_ref(None, 24000)


def test_changed_upstream_layout_is_rejected_before_any_cuda_placement():
    calls = []
    model = voice_model(calls)
    model.s3gen.children["new_generation_module"] = Module("unknown", calls)
    with pytest.raises(RuntimeError, match="tts_default_voice_layout_unsupported"):
        place_default_voice_cuda(model, SimpleNamespace(), precision="bf16")
    assert not calls


def test_layout_incompatibility_is_visible_and_stops_worker(tmp_path, monkeypatch):
    process = FakeProcess([{"error": "tts_default_voice_layout_unsupported"}])
    install_process(monkeypatch, process)
    notices = []
    backend = ChatterboxSpeechBackend(settings(tmp_path), notice=notices.append)
    with pytest.raises(SpeechOutputError, match="tts_default_voice_layout_unsupported"):
        asyncio.run(backend.start())
    assert process.killed and any("tts_default_voice_layout_unsupported" in s for s in notices)


def test_reference_placement_is_logged_at_readiness(tmp_path, monkeypatch, caplog):
    process = FakeProcess([ready() | {"reference_modules_device": "cpu"}])
    install_process(monkeypatch, process)
    backend = ChatterboxSpeechBackend(settings(tmp_path))

    async def run():
        await backend.start()
        await backend.aclose()

    with caplog.at_level("INFO"):
        asyncio.run(run())
    entry = next(r for r in caplog.records if getattr(r, "event", "") == "tts_ready")
    assert entry.reference_modules_device == "cpu" and entry.device == "cuda"
