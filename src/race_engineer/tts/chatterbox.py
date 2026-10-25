"""One persistent local speech backend, shared by both radio output paths."""

import asyncio
import json
import logging
import os
import subprocess
from collections.abc import Callable
from contextlib import suppress
from datetime import UTC, datetime
from typing import cast

from race_engineer.config import TtsConfig
from race_engineer.core.contracts import PlaybackResult, Utterance
from race_engineer.core.conversation import ConversationReply, RadioLanguage
from race_engineer.core.enums import PlaybackStatus
from race_engineer.core.speech_output import SpeechOutputError, SpeechOutputResult
from race_engineer.processes import start_owned_process
from race_engineer.tts_metadata import CHATTERBOX_VERSION, MODEL_FILES

_LOGGER = logging.getLogger(__name__)


class ChatterboxSpeechBackend:
    """A session owns this instance. No torch imports, synthesis or playback in the host."""

    def __init__(self, config: TtsConfig, *, notice: Callable[[str], None] | None = None) -> None:
        self._config = config
        self._notice = notice
        self._process: asyncio.subprocess.Process | None = None
        self._lock = asyncio.Lock()
        self._failed: str | None = None
        self._closed = False
        self._sequence = 0
        self.effective_device: str | None = None

    def _report(self, message: str) -> None:
        _LOGGER.warning(message, extra={"event": "tts_notice"})
        print(message, flush=True)
        if self._notice is not None:
            try:
                self._notice(message)
            except Exception:
                _LOGGER.warning(
                    "TTS notice delivery unavailable", extra={"event": "tts_notice_failed"}
                )

    @property
    def worker_pid(self) -> int | None:
        return self._process.pid if self._process is not None else None

    async def _stop(self) -> None:
        process, self._process = self._process, None
        if process is not None:
            if process.returncode is None:
                with suppress(ProcessLookupError):
                    process.kill()
            await process.wait()

    async def _read(self) -> dict[str, object]:
        if self._process is None or self._process.stdout is None:
            raise SpeechOutputError("tts_worker_exited")
        line = await self._process.stdout.readline()
        if not line:
            raise SpeechOutputError("tts_worker_exited")
        value = json.loads(line)
        if not isinstance(value, dict):
            raise SpeechOutputError("tts_protocol_error")
        return value

    async def _send(self, payload: dict[str, object]) -> None:
        if self._process is None or self._process.stdin is None:
            raise SpeechOutputError("tts_worker_exited")
        data = json.dumps(payload, ensure_ascii=True).encode("utf-8") + b"\n"
        if len(data) > 16_384:
            raise SpeechOutputError("tts_request_too_large")
        self._process.stdin.write(data)
        await self._process.stdin.drain()

    async def _start(self) -> None:
        if self._closed:
            raise SpeechOutputError("tts_closed")
        if self._failed is not None:
            raise SpeechOutputError(self._failed)
        if self._process is not None:
            if self._process.returncode is not None:
                raise SpeechOutputError("tts_worker_exited")
            return
        if not self._config.enabled:
            raise SpeechOutputError("tts_disabled")
        worker = self._config.worker_path
        runtime = worker or self._config.python_path
        if not runtime.is_file():
            raise SpeechOutputError("tts_runtime_missing")
        if any(not (self._config.model_path / name).is_file() for name in MODEL_FILES):
            raise SpeechOutputError("tts_model_missing")
        command = (
            [str(worker.resolve())]
            if worker
            else [str(runtime.resolve()), "-u", "-m", "race_engineer.tts.chatterbox_worker"]
        )
        arguments = [
            *command,
            "--model",
            str(self._config.model_path.resolve()),
            "--device",
            self._config.device,
            "--threads",
            str(self._config.threads),
            "--volume",
            str(self._config.volume),
        ]
        if not self._config.fallback_to_cpu:
            arguments.append("--no-cpu-fallback")
        if self._config.output_device is not None:
            arguments.extend(("--output-device", str(self._config.output_device)))
        environment = dict(os.environ)
        environment.update(
            PYTHONIOENCODING="utf-8", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", DO_NOT_TRACK="1"
        )
        self._process = await start_owned_process(
            *arguments,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
            env=environment,
            limit=16_384,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        ready = await self._read()
        if ready.get("ready") is not True or ready.get("version") != CHATTERBOX_VERSION:
            allowed = {
                "tts_version_mismatch",
                "tts_model_missing",
                "tts_default_voice_missing",
                "tts_cuda_unavailable",
                "tts_languages_missing",
            }
            reason = ready.get("error")
            raise SpeechOutputError(str(reason) if reason in allowed else "tts_model_load_failed")
        if ready.get("device") not in {"cpu", "cuda"}:
            raise SpeechOutputError("tts_protocol_error")
        self.effective_device = str(ready["device"])
        if self.effective_device == "cpu":
            self._report(
                "Chatterbox is using CPU (selected or CUDA unavailable/insufficient memory); "
                "real-time speech performance is not guaranteed."
            )
        _LOGGER.info(
            "Chatterbox default voice ready",
            extra={
                "event": "tts_ready",
                "device": self.effective_device,
                "version": CHATTERBOX_VERSION,
                "initialization_ms": ready.get("initialization_ms"),
            },
        )

    async def _ensure_started(self) -> None:
        try:
            await asyncio.wait_for(self._start(), self._config.startup_timeout_s)
        except BaseException as error:
            await self._stop()
            self._failed = (
                str(error)
                if isinstance(error, SpeechOutputError)
                else "tts_startup_timeout"
                if isinstance(error, TimeoutError)
                else "tts_start_failed"
            )
            if not isinstance(error, asyncio.CancelledError):
                self._report(
                    f"Speech unavailable ({self._failed}). Run scripts/setup_radio_tts.py; "
                    "telemetry and text replies remain active. No alternate voice is used."
                )
                raise SpeechOutputError(self._failed) from error
            raise

    async def start(self) -> None:
        async with self._lock:
            await self._ensure_started()

    async def _cancel_request(self, identifier: int) -> None:
        await self._send({"command": "cancel", "id": identifier})
        while True:
            response = await self._read()
            if response.get("id") != identifier:
                raise SpeechOutputError("tts_protocol_error")
            if response.get("cancel_ack") is True:
                return

    async def _cleanup_cancelled(self, identifier: int) -> None:
        # Do not release the shared radio floor until the worker confirms that both
        # generation and output stopped. Repeated cancellation cannot detach cleanup.
        cleanup = asyncio.create_task(
            asyncio.wait_for(self._cancel_request(identifier), self._config.shutdown_timeout_s)
        )
        while not cleanup.done():
            try:
                await asyncio.shield(cleanup)
            except asyncio.CancelledError:
                continue
            except Exception:
                break
        try:
            cleanup.result()
        except Exception:
            self._failed = "tts_cancel_failed"
            await self._stop()
            self._report(
                "Speech worker could not stop safely; speech disabled until session restart."
            )

    async def speak(
        self,
        reply: ConversationReply,
        *,
        play_audio: bool = True,
        before_playback: Callable[[], bool] | None = None,
    ) -> SpeechOutputResult:
        async with self._lock:
            await self._ensure_started()
            self._sequence += 1
            identifier = self._sequence
            sent = False

            async def exchange() -> SpeechOutputResult:
                nonlocal sent
                await self._send(
                    {
                        "id": identifier,
                        "text": reply.text,
                        "language": reply.language,
                        "play_audio": play_audio,
                        "defer_playback": play_audio,
                    }
                )
                sent = True
                response = await self._read()
                if response.get("id") != identifier:
                    raise SpeechOutputError("tts_protocol_error")
                if play_audio:
                    if response.get("prepared") is not True:
                        raise SpeechOutputError("tts_generation_or_playback_failed")
                    if before_playback is not None and not before_playback():
                        await self._cancel_request(identifier)
                        raise SpeechOutputError("tts_reply_expired")
                    await self._send({"command": "play", "id": identifier})
                    response = await self._read()
                if response.get("id") != identifier or "error" in response:
                    raise SpeechOutputError("tts_generation_or_playback_failed")
                result = SpeechOutputResult.model_validate(response.get("result"))
                if result.language != reply.language or result.played != play_audio:
                    raise SpeechOutputError("tts_protocol_error")
                return result

            try:
                return await asyncio.wait_for(exchange(), self._config.playback_timeout_s)
            except asyncio.CancelledError:
                if sent:
                    await self._cleanup_cancelled(identifier)
                raise
            except TimeoutError as error:
                if sent:
                    await self._cleanup_cancelled(identifier)
                self._report("Speech generation/playback timed out; telemetry continues.")
                raise SpeechOutputError("tts_playback_timeout") from error
            except (OSError, ValueError, SpeechOutputError) as error:
                reason = (
                    str(error) if isinstance(error, SpeechOutputError) else "tts_protocol_error"
                )
                if reason in {"tts_protocol_error", "tts_worker_exited"}:
                    self._failed = reason
                    await self._stop()
                if reason != "tts_reply_expired":
                    self._report(
                        f"Speech failed ({reason}); telemetry continues. No voice fallback."
                    )
                raise SpeechOutputError(reason) from error

    async def aclose(self) -> None:
        self._closed = True
        # The owner first closes the radio, which cancels/drains in-flight speech.
        async with self._lock:
            await self._stop()


class ChatterboxAnnouncementEngine:
    """TextToSpeechEngine facade using the session's exact same backend and voice."""

    def __init__(self, backend: ChatterboxSpeechBackend) -> None:
        self.backend = backend
        self._task: asyncio.Task[SpeechOutputResult] | None = None

    async def speak(self, utterance: Utterance, *, play_audio: bool = True) -> PlaybackResult:
        started = datetime.now(UTC)
        language = utterance.generator_metadata.get("language", "en")
        if language not in {"en", "tr"}:
            raise SpeechOutputError("tts_unsupported_language")
        reply = ConversationReply(
            text=utterance.text,
            language=cast(RadioLanguage, language),
            status="answered",
            session_id="automatic",
            source_sequence=0,
            mode="live",
        )
        before_playback = None
        deadline_value = utterance.generator_metadata.get("speech_deadline")
        if deadline_value is not None:
            try:
                if not isinstance(deadline_value, str):
                    raise ValueError("invalid speech deadline")
                deadline = datetime.fromisoformat(deadline_value)
                if deadline.tzinfo is None:
                    raise ValueError("naive speech deadline")
            except ValueError as error:
                raise SpeechOutputError("tts_invalid_deadline") from error

            def before_playback() -> bool:
                return datetime.now(UTC) < deadline

        task = asyncio.create_task(
            self.backend.speak(reply, play_audio=play_audio, before_playback=before_playback)
        )
        self._task = task
        try:
            await task
            status, reason = PlaybackStatus.COMPLETED, None
        except asyncio.CancelledError:
            current = asyncio.current_task()
            if current is not None and current.cancelling():
                raise
            status, reason = PlaybackStatus.CANCELLED, "interrupted"
        except SpeechOutputError as error:
            status = (
                PlaybackStatus.EXPIRED
                if str(error) == "tts_reply_expired"
                else PlaybackStatus.FAILED
            )
            reason = str(error)
        finally:
            if self._task is task:
                self._task = None
        return PlaybackResult(
            intent_id=utterance.intent_id,
            status=status,
            error_code=reason,
            started_at=started,
            finished_at=datetime.now(UTC),
        )

    async def cancel(self) -> None:
        task = self._task
        if task is not None and not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    async def aclose(self) -> None:
        await self.cancel()
        await self.backend.aclose()
