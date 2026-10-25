"""Private, offline JSON-lines speech process. One multilingual model, no reference audio.

The protocol loop stays responsive while a single generation/playback task runs.
Cancellation is checked at PyTorch module boundaries and between output blocks;
the loaded model survives ordinary interruptions. No audio or driver text is saved.
"""

import argparse
import asyncio
import gc
import importlib
import importlib.metadata
import json
import os
import sys
import threading
import time
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any

from pydantic import Field

from race_engineer.core.contracts import ContractModel
from race_engineer.core.conversation import RadioLanguage
from race_engineer.core.speech_output import SpeechOutputResult
from race_engineer.tts_metadata import CHATTERBOX_VERSION, MODEL_FILES

MAX_REQUEST_BYTES = 16_384


class SpeakRequest(ContractModel):
    id: int = Field(ge=1)
    text: str = Field(min_length=1, max_length=1500)
    language: RadioLanguage
    play_audio: bool = True
    defer_playback: bool = False


class SpeechCancelled(Exception):
    pass


def check_cancelled(cancelled: threading.Event) -> None:
    if cancelled.is_set():
        raise SpeechCancelled()


def select_device(torch: Any, requested: str, fallback: bool) -> tuple[str, str | None]:
    if requested == "cpu":
        return "cpu", "CPU speech selected; real-time performance is not guaranteed."
    try:
        available = torch.cuda.is_available()
        if available:
            # A real kernel catches unsupported GPU architectures/driver combinations.
            torch.ones(1, device="cuda").add_(1)
            torch.cuda.synchronize()
            return "cuda", None
    except Exception:
        available = False
    if not fallback:
        raise RuntimeError("tts_cuda_unavailable")
    return "cpu", "CUDA unavailable; using CPU speech. Real-time performance is not guaranteed."


def load_model(
    model_path: Path, device: str, fallback: bool, threads: int
) -> tuple[Any, str, str | None]:
    if importlib.metadata.version("chatterbox-tts") != CHATTERBOX_VERSION:
        raise RuntimeError("tts_version_mismatch")
    if any(not (model_path / name).is_file() for name in MODEL_FILES):
        raise RuntimeError("tts_model_missing")
    torch = importlib.import_module("torch")
    torch.set_num_threads(threads)
    api = importlib.import_module("chatterbox.mtl_tts")
    selected, warning = select_device(torch, device, fallback)
    try:
        model = api.ChatterboxMultilingualTTS.from_local(model_path, device=selected)
    except torch.cuda.OutOfMemoryError:
        if selected != "cuda" or not fallback:
            raise
        gc.collect()
        torch.cuda.empty_cache()
        selected = "cpu"
        warning = "Insufficient CUDA memory; CPU speech may miss racing deadlines."
        model = api.ChatterboxMultilingualTTS.from_local(model_path, device=selected)
    if model.conds is None:
        raise RuntimeError("tts_default_voice_missing")
    if not {"en", "tr"}.issubset(model.get_supported_languages()):
        raise RuntimeError("tts_languages_missing")
    return model, selected, warning


def synthesize_pcm(
    model: Any, request: SpeakRequest, cancelled: threading.Event, volume: float
) -> tuple[bytes, int]:
    torch = importlib.import_module("torch")
    numpy = importlib.import_module("numpy")

    def interrupt(_module: Any, _inputs: Any) -> None:
        check_cancelled(cancelled)

    check_cancelled(cancelled)
    # Confined to this single-model worker; remove even after cancelled generation.
    hook = torch.nn.modules.module.register_module_forward_pre_hook(interrupt)
    try:
        with torch.inference_mode():
            # conds.pt supplies the built-in voice. Never prepare/upload a reference.
            wav = model.generate(request.text, language_id=request.language)
        check_cancelled(cancelled)
        samples = wav.detach().cpu().numpy()
        if samples.ndim != 2 or samples.shape[0] != 1:
            raise ValueError("tts_audio_format_invalid")
        rate = int(model.sr)
        if not 8000 <= rate <= 48000 or not 0 < samples.shape[1] <= rate * 60:
            raise ValueError("tts_audio_duration_invalid")
        if not numpy.isfinite(samples).all():
            raise ValueError("tts_audio_nonfinite")
        pcm = numpy.rint(numpy.clip(samples[0] * volume, -1, 1) * 32767).astype("<i2")
        return bytes(pcm.tobytes()), rate
    finally:
        hook.remove()


def prepare_output(
    pcm: bytes, rate: int, device: int | None, cancelled: threading.Event
) -> tuple[bytes, int]:
    check_cancelled(cancelled)
    sd = importlib.import_module("sounddevice")
    # Prefer the model's native 24 kHz, resample to the selected device's preferred
    # rate when WASAPI rejects it. Never silently choose a different output device.
    try:
        sd.check_output_settings(device=device, channels=1, dtype="int16", samplerate=rate)
    except Exception:
        numpy = importlib.import_module("numpy")
        librosa = importlib.import_module("librosa")
        target = int(sd.query_devices(device, "output")["default_samplerate"])
        if not 8000 <= target <= 192000:
            raise ValueError("tts_output_rate_invalid") from None
        values = numpy.frombuffer(pcm, dtype="<i2").astype("float32") / 32768
        values = librosa.resample(values, orig_sr=rate, target_sr=target)
        pcm = numpy.rint(numpy.clip(values, -1, 1) * 32767).astype("<i2").tobytes()
        rate = target
    check_cancelled(cancelled)
    return pcm, rate


def play_pcm(
    pcm: bytes, rate: int, device: int | None, cancelled: threading.Event, *, prepared: bool = False
) -> None:
    if not prepared:
        pcm, rate = prepare_output(pcm, rate, device, cancelled)
    check_cancelled(cancelled)
    sd = importlib.import_module("sounddevice")
    chunk_size = max(1, rate // 50) * 2
    with sd.RawOutputStream(
        samplerate=rate,
        channels=1,
        dtype="int16",
        device=device,
        blocksize=rate // 50,
        latency="low",
    ) as stream:
        try:
            for offset in range(0, len(pcm), chunk_size):
                check_cancelled(cancelled)
                if stream.write(pcm[offset : offset + chunk_size]):
                    raise RuntimeError("tts_output_underflow")
            check_cancelled(cancelled)
        except BaseException:
            stream.abort()
            raise


def send(value: dict[str, object]) -> None:
    print(json.dumps(value, ensure_ascii=True), file=sys.__stdout__, flush=True)


async def serve(model: Any, *, volume: float, output_device: int | None) -> None:
    active: asyncio.Task[None] | None = None
    active_id: int | None = None
    cancelled = threading.Event()
    playback = asyncio.Event()

    async def render(request: SpeakRequest) -> None:
        started = time.perf_counter()
        try:
            with redirect_stdout(sys.stderr):
                pcm, rate = await asyncio.to_thread(
                    synthesize_pcm, model, request, cancelled, volume
                )
                if request.play_audio:
                    # Finish device negotiation/resampling before telling the parent
                    # audio is prepared; freshness is checked after all preparation.
                    pcm, rate = await asyncio.to_thread(
                        prepare_output, pcm, rate, output_device, cancelled
                    )
            synthesis_ms = (time.perf_counter() - started) * 1000
            if request.play_audio and request.defer_playback:
                send({"id": request.id, "prepared": True})
                await playback.wait()
            check_cancelled(cancelled)
            started = time.perf_counter()
            if request.play_audio:
                await asyncio.to_thread(
                    play_pcm, pcm, rate, output_device, cancelled, prepared=True
                )
            result = SpeechOutputResult(
                language=request.language,
                played=request.play_audio,
                audio_duration_s=len(pcm) / (rate * 2),
                synthesis_ms=synthesis_ms,
                playback_ms=(time.perf_counter() - started) * 1000 if request.play_audio else 0,
            )
            send({"id": request.id, "result": result.model_dump()})
        except SpeechCancelled:
            send({"id": request.id, "cancelled": True})
        except Exception:
            # Never send exception bodies, native logs or driver text over the protocol.
            send({"id": request.id, "error": "tts_generation_or_playback_failed"})

    try:
        while line := await asyncio.to_thread(sys.stdin.buffer.readline, MAX_REQUEST_BYTES + 1):
            if len(line) > MAX_REQUEST_BYTES:
                return
            try:
                message = json.loads(line)
                if message.get("command") in {"cancel", "play"}:
                    if message.get("id") == active_id:
                        if message["command"] == "cancel":
                            cancelled.set()
                            playback.set()
                            if active is not None:
                                await active
                            send({"id": active_id, "cancel_ack": True})
                        else:
                            playback.set()
                    continue
                request = SpeakRequest.model_validate(message)
                if active is not None and not active.done():
                    send({"id": request.id, "error": "tts_worker_busy"})
                    continue
                if active is not None:
                    await active
                active_id = request.id
                cancelled.clear()
                playback.clear()
                active = asyncio.create_task(render(request))
            except (ValueError, AttributeError):
                send({"error": "tts_protocol_error"})
    finally:
        cancelled.set()
        playback.set()
        if active is not None:
            await active


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--no-cpu-fallback", action="store_true")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--volume", type=float, default=0.8)
    parser.add_argument("--output-device", type=int)
    args = parser.parse_args()
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", DO_NOT_TRACK="1")
    started = time.perf_counter()
    try:
        with redirect_stdout(sys.stderr):
            model, device, warning = load_model(
                args.model, args.device, not args.no_cpu_fallback, args.threads
            )
            # Pay lazy CUDA/vocoder initialization before any deadline-bound race
            # speech. This fixed authored check is neither played nor persisted.
            synthesize_pcm(
                model,
                SpeakRequest(id=1, text="Radio check.", language="en", play_audio=False),
                threading.Event(),
                args.volume,
            )
        send(
            {
                "ready": True,
                "device": device,
                "warning": warning,
                "version": CHATTERBOX_VERSION,
                "initialization_ms": round((time.perf_counter() - started) * 1000, 3),
            }
        )
    except Exception as error:
        code = str(error)
        allowed = {
            "tts_version_mismatch",
            "tts_model_missing",
            "tts_default_voice_missing",
            "tts_languages_missing",
            "tts_cuda_unavailable",
        }
        send({"error": code if code in allowed else "tts_model_load_failed"})
        return
    asyncio.run(serve(model, volume=args.volume, output_device=args.output_device))


if __name__ == "__main__":
    main()
