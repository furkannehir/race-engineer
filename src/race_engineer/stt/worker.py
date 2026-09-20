"""Isolated, offline Qwen3-ASR worker. Private pipe protocol, no audio files or sockets."""

import argparse
import base64
import contextlib
import importlib
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

from race_engineer.core.speech_input import AudioClip, SpeechInputError, Transcription


def normalize_result(
    parsed: dict[str, Any], audio: AudioClip, elapsed_ms: float, language_hint: str
) -> Transcription:
    text = parsed.get("transcription", "")
    if not isinstance(text, str) or len(text) > 1000:
        raise SpeechInputError("stt_output_invalid")
    text = " ".join(text.split())
    language = language_hint if language_hint != "auto" else parsed.get("language")
    if not text or language in {None, "None", "none"}:
        return Transcription(
            status="no_speech",
            audio_duration_s=audio.duration_s,
            elapsed_ms=elapsed_ms,
            reason="model_no_speech",
        )
    code = (
        {"English": "en", "Turkish": "tr", "en": "en", "tr": "tr"}.get(language)
        if isinstance(language, str)
        else None
    )
    if code is None:
        return Transcription(
            status="unsupported_language",
            audio_duration_s=audio.duration_s,
            elapsed_ms=elapsed_ms,
            reason="unsupported_language",
        )
    return Transcription.model_validate(
        {
            "status": "transcribed",
            "text": text,
            "language": code,
            "audio_duration_s": audio.duration_s,
            "elapsed_ms": elapsed_ms,
        }
    )


class QwenAsrRuntime:
    def __init__(self, model_path: Path, device: str, threads: int) -> None:
        if not model_path.is_dir():
            raise SpeechInputError("stt_model_missing")
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
        self.torch = importlib.import_module("torch")
        self.np = importlib.import_module("numpy")
        transformers = importlib.import_module("transformers")
        self.torch.set_num_threads(threads)
        if device == "cuda" and not self.torch.cuda.is_available():
            raise SpeechInputError("stt_cuda_unavailable")
        dtype = self.torch.float32 if device == "cpu" else self.torch.float16
        self.processor = transformers.AutoProcessor.from_pretrained(
            str(model_path), local_files_only=True, trust_remote_code=False
        )
        self.model = (
            transformers.AutoModelForMultimodalLM.from_pretrained(
                str(model_path),
                local_files_only=True,
                trust_remote_code=False,
                dtype=dtype,
                attn_implementation="sdpa",
            )
            .to(device)
            .eval()
        )

    def transcribe(self, audio: AudioClip, language: str) -> Transcription:
        started = time.perf_counter()
        waveform = self.np.frombuffer(audio.pcm16, dtype="<i2").astype(self.np.float32) / 32768
        if audio.sample_rate_hz != 16000:
            signal = importlib.import_module("scipy.signal")
            divisor = math.gcd(audio.sample_rate_hz, 16000)
            waveform = signal.resample_poly(
                waveform, 16000 // divisor, audio.sample_rate_hz // divisor
            )
        inputs = self.processor.apply_transcription_request(
            audio=waveform,
            language=None if language == "auto" else language,
        ).to(self.model.device, self.model.dtype)
        with self.torch.inference_mode():
            output = self.model.generate(**inputs, max_new_tokens=256, do_sample=False)
        generated = output[:, inputs["input_ids"].shape[1] :]
        if generated.shape[1] >= 256:
            raise SpeechInputError("stt_output_truncated")
        parsed = self.processor.decode(generated, return_format="parsed")[0]
        return normalize_result(parsed, audio, (time.perf_counter() - started) * 1000, language)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--threads", type=int, required=True)
    args = parser.parse_args()
    output_stream = sys.stdout

    def emit(payload: dict[str, Any]) -> None:
        print(json.dumps(payload, ensure_ascii=True), file=output_stream, flush=True)

    try:
        with contextlib.redirect_stdout(sys.stderr):
            runtime = QwenAsrRuntime(args.model, args.device, args.threads)
    except Exception as error:
        reason = str(error) if isinstance(error, SpeechInputError) else "stt_model_load_failed"
        emit({"ready": False, "reason": reason, "error_type": type(error).__name__})
        return
    emit({"ready": True})
    while line := sys.stdin.buffer.readline(4_000_001):
        if len(line) > 4_000_000 or not line.endswith(b"\n"):
            return
        try:
            payload = json.loads(line)
            language = payload["language"]
            if language not in {"auto", "en", "tr"}:
                raise SpeechInputError("stt_language_invalid")
            audio = AudioClip(
                base64.b64decode(payload["pcm16"], validate=True), payload["sample_rate_hz"]
            )
            with contextlib.redirect_stdout(sys.stderr):
                result = runtime.transcribe(audio, language)
            emit({"result": result.model_dump(mode="json")})
        except Exception as error:
            reason = str(error) if isinstance(error, SpeechInputError) else "stt_inference_failed"
            emit({"error": reason})


if __name__ == "__main__":
    main()
