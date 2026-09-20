"""Persistent, killable local voice worker, separate from automatic SAPI calls."""

import asyncio
import json
import os
import subprocess
from collections.abc import Callable
from contextlib import suppress

from race_engineer.config import RadioTtsConfig
from race_engineer.core.conversation import ConversationReply
from race_engineer.core.speech_output import SpeechOutputError, SpeechOutputResult
from race_engineer.processes import start_owned_process


class PiperConversationSpeaker:
    def __init__(self, config: RadioTtsConfig) -> None:
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
            raise SpeechOutputError("tts_worker_exited")
        data = json.loads(line)
        if not isinstance(data, dict):
            raise SpeechOutputError("tts_protocol_error")
        return data

    async def _start(self) -> None:
        if self._process is not None and self._process.returncode is None:
            return
        if not self._config.python_path.is_file():
            raise SpeechOutputError("tts_runtime_missing")
        for path in (self._config.english_model_path, self._config.turkish_model_path):
            if not path.is_file() or not path.with_suffix(".onnx.json").is_file():
                raise SpeechOutputError("tts_voice_missing")
        environment = dict(os.environ)
        environment.update(PYTHONIOENCODING="utf-8", HF_HUB_OFFLINE="1")
        arguments = [
            str(self._config.python_path.resolve()),
            "-u",
            "-m",
            "race_engineer.tts.piper_worker",
            "--english-model",
            str(self._config.english_model_path.resolve()),
            "--turkish-model",
            str(self._config.turkish_model_path.resolve()),
            "--threads",
            str(self._config.threads),
            "--volume",
            str(self._config.volume),
            "--length-scale",
            str(self._config.length_scale),
        ]
        if self._config.output_device is not None:
            arguments.extend(("--output-device", str(self._config.output_device)))
        self._process = await start_owned_process(
            *arguments,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
            env=environment,
            limit=16_384,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        if (await self._read()).get("ready") is not True:
            raise SpeechOutputError("tts_voice_load_failed")

    async def _ensure_started(self) -> None:
        try:
            await asyncio.wait_for(self._start(), self._config.startup_timeout_s)
        except BaseException as error:
            await self._stop()
            if isinstance(error, TimeoutError):
                raise SpeechOutputError("tts_startup_timeout") from error
            if isinstance(error, (OSError, ValueError)):
                raise SpeechOutputError("tts_start_failed") from error
            raise

    async def start(self) -> None:
        async with self._lock:
            await self._ensure_started()

    async def speak(
        self,
        reply: ConversationReply,
        *,
        play_audio: bool = True,
        before_playback: Callable[[], bool] | None = None,
    ) -> SpeechOutputResult:
        async with self._lock:
            await self._ensure_started()

            async def exchange() -> SpeechOutputResult:
                assert self._process is not None and self._process.stdin is not None
                # Text is private pipe data, never shell syntax, process arguments, or logs.
                payload = {"text": reply.text, "language": reply.language, "play_audio": play_audio}
                if before_playback is not None and play_audio:
                    payload["defer_playback"] = True
                self._process.stdin.write(json.dumps(payload).encode("utf-8") + b"\n")
                await self._process.stdin.drain()
                response = await self._read()
                if before_playback is not None and play_audio:
                    if response.get("prepared") is not True:
                        raise SpeechOutputError("tts_protocol_error")
                    if not before_playback():
                        raise SpeechOutputError("tts_reply_expired")
                    self._process.stdin.write(b'{"play":true}\n')
                    await self._process.stdin.drain()
                    response = await self._read()
                if "error" in response:
                    raise SpeechOutputError("tts_synthesis_or_playback_failed")
                result = SpeechOutputResult.model_validate(response.get("result"))
                if result.language != reply.language or result.played != play_audio:
                    raise SpeechOutputError("tts_protocol_error")
                return result

            try:
                return await asyncio.wait_for(exchange(), self._config.playback_timeout_s)
            except BaseException as error:
                await self._stop()
                if isinstance(error, TimeoutError):
                    raise SpeechOutputError("tts_playback_timeout") from error
                if isinstance(error, (OSError, ValueError)):
                    raise SpeechOutputError("tts_protocol_error") from error
                raise

    async def aclose(self) -> None:
        async with self._lock:
            await self._stop()
