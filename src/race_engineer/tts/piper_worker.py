"""Private JSON-lines CPU synthesizer/player. No server, recording, or downloads."""

import argparse
import importlib
import json
import sys
import time
from collections.abc import Callable
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any

from pydantic import Field

from race_engineer.core.contracts import ContractModel
from race_engineer.core.conversation import RadioLanguage
from race_engineer.core.speech_output import SpeechOutputResult

MAX_AUDIO_S = 60
MAX_REQUEST_BYTES = 16_384


class SpeakRequest(ContractModel):
    text: str = Field(min_length=1, max_length=1500)
    language: RadioLanguage
    play_audio: bool = True
    defer_playback: bool = False


def load_voice(path: Path, language: RadioLanguage, threads: int) -> Any:
    piper = importlib.import_module("piper")
    configuration = importlib.import_module("piper.config")
    ort = importlib.import_module("onnxruntime")
    ort.disable_telemetry_events()
    with path.with_suffix(".onnx.json").open(encoding="utf-8") as stream:
        raw = json.load(stream)
    # No automatic foreign-language fallback or remotely fetched phonemizers.
    if raw.get("language", {}).get("code", "").split("_")[0] != language:
        raise ValueError("voice_language_mismatch")
    if raw.get("phoneme_type") != "espeak":
        raise ValueError("unsupported_phonemizer")
    expected_espeak = {"en", "en-us", "en-gb"} if language == "en" else {"tr"}
    if raw.get("espeak", {}).get("voice") not in expected_espeak:
        raise ValueError("voice_phonemizer_mismatch")
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(
        str(path), sess_options=options, providers=["CPUExecutionProvider"]
    )
    return piper.PiperVoice(session=session, config=configuration.PiperConfig.from_dict(raw))


def synthesize_pcm(voice: Any, text: str, synthesis_config: Any) -> tuple[bytes, int]:
    rate = int(voice.config.sample_rate)
    if not 8000 <= rate <= 48000:
        raise ValueError("invalid_voice_sample_rate")
    maximum_bytes = rate * MAX_AUDIO_S * 2
    audio = bytearray()
    for chunk in voice.synthesize(text, syn_config=synthesis_config):
        if chunk.sample_rate != rate or chunk.sample_width != 2 or chunk.sample_channels != 1:
            raise ValueError("invalid_voice_audio_format")
        pcm = chunk.audio_int16_bytes
        if len(pcm) % 2 or len(audio) + len(pcm) > maximum_bytes:
            raise ValueError("voice_audio_limit")
        audio.extend(pcm)
    if not audio:
        raise ValueError("voice_audio_empty")
    return bytes(audio), rate


def play_pcm(pcm: bytes, rate: int, device: int | None) -> None:
    sd = importlib.import_module("sounddevice")
    # Blocking audio I/O is confined to this disposable worker. Context exit drains audio.
    with sd.RawOutputStream(samplerate=rate, channels=1, dtype="int16", device=device) as stream:
        if stream.write(pcm):
            raise RuntimeError("audio_output_underflow")


def render_request(
    request: SpeakRequest,
    voices: dict[str, Any],
    synthesis_config: Any,
    output_device: int | None,
    before_playback: Callable[[], None] | None = None,
) -> SpeechOutputResult:
    started = time.perf_counter()
    pcm, rate = synthesize_pcm(voices[request.language], request.text, synthesis_config)
    synthesis_ms = (time.perf_counter() - started) * 1000
    playback_ms = 0.0
    if request.play_audio:
        if before_playback is not None:
            before_playback()
        started = time.perf_counter()
        play_pcm(pcm, rate, output_device)
        playback_ms = (time.perf_counter() - started) * 1000
    return SpeechOutputResult(
        language=request.language,
        played=request.play_audio,
        audio_duration_s=len(pcm) / (rate * 2),
        synthesis_ms=synthesis_ms,
        playback_ms=playback_ms,
    )


def confirm_playback() -> None:
    # Stdout may be redirected while native synthesis runs, so use the original pipe.
    print(json.dumps({"prepared": True}), file=sys.__stdout__, flush=True)
    line = sys.stdin.buffer.readline(128)
    if json.loads(line) != {"play": True}:
        raise ValueError("playback_not_confirmed")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--english-model", type=Path, required=True)
    parser.add_argument("--turkish-model", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--volume", type=float, default=0.8)
    parser.add_argument("--length-scale", type=float, default=1.0)
    parser.add_argument("--output-device", type=int)
    args = parser.parse_args()
    try:
        with redirect_stdout(sys.stderr):
            piper = importlib.import_module("piper")
            voices = {
                "en": load_voice(args.english_model, "en", args.threads),
                "tr": load_voice(args.turkish_model, "tr", args.threads),
            }
            synthesis_config = piper.SynthesisConfig(
                volume=args.volume, length_scale=args.length_scale
            )
        print(json.dumps({"ready": True}), flush=True)
    except Exception:
        print(json.dumps({"error": "voice_load_failed"}), flush=True)
        return
    while line := sys.stdin.buffer.readline(MAX_REQUEST_BYTES + 1):
        if len(line) > MAX_REQUEST_BYTES:
            print(json.dumps({"error": "request_too_large"}), flush=True)
            return
        try:
            request = SpeakRequest.model_validate_json(line)
            with redirect_stdout(sys.stderr):
                result = render_request(
                    request,
                    voices,
                    synthesis_config,
                    args.output_device,
                    confirm_playback if request.defer_playback else None,
                )
            print(json.dumps({"result": result.model_dump()}), flush=True)
        except Exception:
            # Never echo potentially private reply text or native-library exceptions.
            print(json.dumps({"error": "synthesis_or_playback_failed"}), flush=True)
            return


if __name__ == "__main__":
    main()
