"""Async ASR adapter with a persistent worker and killable startup/inference timeouts."""

import asyncio
import base64
import json
import os
import subprocess
from contextlib import suppress

from race_engineer.config import SttConfig
from race_engineer.core.speech_input import AudioClip, SpeechInputError, Transcription
from race_engineer.processes import start_owned_process
from race_engineer.stt.audio import silence_reason


class QwenSpeechRecognizer:
    def __init__(self, config: SttConfig) -> None:
        self._config = config
        self._process: asyncio.subprocess.Process | None = None
        self._lock = asyncio.Lock()

    async def _stop(self) -> None:
        process, self._process = self._process, None
        if process is not None:
            if process.returncode is None:
                with suppress(ProcessLookupError):
                    process.kill()
            await process.wait()

    async def _read(self) -> dict[str, object]:
        assert self._process is not None and self._process.stdout is not None
        line = await self._process.stdout.readline()
        if not line:
            raise SpeechInputError("stt_worker_exited")
        data = json.loads(line)
        if not isinstance(data, dict):
            raise SpeechInputError("stt_worker_protocol_error")
        return data

    async def _start(self) -> None:
        if self._process is not None and self._process.returncode is None:
            return
        if not self._config.python_path.is_file():
            raise SpeechInputError("stt_runtime_missing")
        if not self._config.model_path.is_dir():
            raise SpeechInputError("stt_model_missing")
        environment = dict(os.environ)
        environment.update(
            HF_HUB_OFFLINE="1",
            TRANSFORMERS_OFFLINE="1",
            HF_HUB_DISABLE_TELEMETRY="1",
            PYTHONIOENCODING="utf-8",
        )
        self._process = await start_owned_process(
            str(self._config.python_path.resolve()),
            "-u",
            "-m",
            "race_engineer.stt.worker",
            "--model",
            str(self._config.model_path.resolve()),
            "--device",
            self._config.device,
            "--threads",
            str(self._config.threads),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
            env=environment,
            limit=16_384,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        ready = await self._read()
        if ready.get("ready") is not True:
            raise SpeechInputError("stt_model_load_failed")

    async def start(self) -> None:
        async with self._lock:
            try:
                await asyncio.wait_for(self._start(), self._config.startup_timeout_s)
            except BaseException as error:
                await self._stop()
                if isinstance(error, TimeoutError):
                    raise SpeechInputError("stt_startup_timeout") from error
                if isinstance(error, (OSError, ValueError)):
                    raise SpeechInputError("stt_worker_start_failed") from error
                raise

    async def transcribe(self, audio: AudioClip) -> Transcription:
        reason = silence_reason(audio, self._config)
        if reason is not None:
            return Transcription(
                status="no_speech", audio_duration_s=audio.duration_s, reason=reason
            )
        await self.start()
        async with self._lock:

            async def exchange() -> Transcription:
                assert self._process is not None and self._process.stdin is not None
                payload = {
                    "pcm16": base64.b64encode(audio.pcm16).decode("ascii"),
                    "sample_rate_hz": audio.sample_rate_hz,
                    "language": self._config.language,
                }
                self._process.stdin.write(json.dumps(payload).encode("ascii") + b"\n")
                await self._process.stdin.drain()
                reply = await self._read()
                if "error" in reply:
                    # Do not trust worker error strings as loggable user-safe text.
                    raise SpeechInputError("stt_inference_failed")
                return Transcription.model_validate(reply.get("result"))

            try:
                return await asyncio.wait_for(exchange(), self._config.timeout_s)
            except BaseException as error:
                await self._stop()
                if isinstance(error, TimeoutError):
                    raise SpeechInputError("stt_timeout") from error
                if isinstance(error, (OSError, ValueError)):
                    raise SpeechInputError("stt_worker_protocol_error") from error
                raise

    async def aclose(self) -> None:
        async with self._lock:
            await self._stop()
