"""Offline Windows text-to-speech through the built-in SAPI COM interface."""

import asyncio
import base64
import logging
import os
import subprocess
from datetime import UTC, datetime

from race_engineer.config import TtsConfig
from race_engineer.core.contracts import PlaybackResult, Utterance
from race_engineer.core.enums import PlaybackStatus

_LOGGER = logging.getLogger(__name__)

_SPEAK_SCRIPT = r"""
$ErrorActionPreference = 'Stop'
$speaker = New-Object -ComObject SAPI.SpVoice
$speaker.Rate = [int]$env:RACE_ENGINEER_TTS_RATE
$speaker.Volume = [int]$env:RACE_ENGINEER_TTS_VOLUME
$voiceBytes = [Convert]::FromBase64String($env:RACE_ENGINEER_TTS_VOICE_B64)
$voiceName = [Text.Encoding]::UTF8.GetString($voiceBytes)
if ($voiceName) {
    $voice = @($speaker.GetVoices()) |
        Where-Object { $_.GetDescription() -eq $voiceName } |
        Select-Object -First 1
    if ($null -eq $voice) {
        throw "Requested SAPI voice is not installed"
    }
    $speaker.Voice = $voice
}
$textBytes = [Convert]::FromBase64String($env:RACE_ENGINEER_TTS_TEXT_B64)
$text = [Text.Encoding]::UTF8.GetString($textBytes)
[void]$speaker.Speak($text)
"""

_LIST_VOICES_SCRIPT = r"""
$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = [Text.UTF8Encoding]::new($false)
$speaker = New-Object -ComObject SAPI.SpVoice
$speaker.GetVoices() | ForEach-Object { $_.GetDescription() }
"""


def _encoded_command(script: str) -> str:
    return base64.b64encode(script.encode("utf-16-le")).decode("ascii")


def _text_base64(value: str) -> str:
    return base64.b64encode(value.encode("utf-8")).decode("ascii")


def _powershell_arguments(script: str) -> tuple[str, ...]:
    return (
        "powershell.exe",
        "-NoLogo",
        "-NoProfile",
        "-NonInteractive",
        "-ExecutionPolicy",
        "Bypass",
        "-EncodedCommand",
        _encoded_command(script),
    )


async def _stop_process(process: asyncio.subprocess.Process) -> None:
    if process.returncode is not None:
        return
    try:
        process.terminate()
    except ProcessLookupError:
        return
    try:
        await asyncio.wait_for(process.wait(), timeout=1.0)
    except TimeoutError:
        try:
            process.kill()
        except ProcessLookupError:
            return
        await process.wait()


class WindowsSapiTextToSpeechEngine:
    """Speaks locally without blocking the asyncio event loop or opening a window."""

    def __init__(self, config: TtsConfig) -> None:
        self._config = config
        self._speak_lock = asyncio.Lock()
        self._current_process: asyncio.subprocess.Process | None = None
        self._cancelled_process: asyncio.subprocess.Process | None = None

    @staticmethod
    async def installed_voices() -> tuple[str, ...]:
        process = await asyncio.create_subprocess_exec(
            *_powershell_arguments(_LIST_VOICES_SCRIPT),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        stdout, _ = await process.communicate()
        if process.returncode != 0:
            raise RuntimeError("could not enumerate installed Windows SAPI voices")
        return tuple(
            line.strip() for line in stdout.decode("utf-8-sig").splitlines() if line.strip()
        )

    async def speak(self, utterance: Utterance) -> PlaybackResult:
        async with self._speak_lock:
            started_at = datetime.now(UTC)
            environment = dict(os.environ)
            environment.update(
                {
                    "RACE_ENGINEER_TTS_TEXT_B64": _text_base64(utterance.text),
                    "RACE_ENGINEER_TTS_VOICE_B64": _text_base64(self._config.voice or ""),
                    "RACE_ENGINEER_TTS_RATE": str(self._config.rate),
                    "RACE_ENGINEER_TTS_VOLUME": str(self._config.volume),
                }
            )
            try:
                process = await asyncio.create_subprocess_exec(
                    *_powershell_arguments(_SPEAK_SCRIPT),
                    stdout=asyncio.subprocess.DEVNULL,
                    stderr=asyncio.subprocess.PIPE,
                    env=environment,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
            except OSError as error:
                _LOGGER.exception(
                    "Windows SAPI process could not start",
                    extra={
                        "event": "tts_failed",
                        "intent_id": utterance.intent_id,
                        "reason": type(error).__name__,
                    },
                )
                return PlaybackResult(
                    intent_id=utterance.intent_id,
                    status=PlaybackStatus.FAILED,
                    started_at=started_at,
                    finished_at=datetime.now(UTC),
                    error_code="process_start_failed",
                )

            self._current_process = process
            self._cancelled_process = None
            try:
                try:
                    await asyncio.wait_for(
                        process.communicate(),
                        timeout=self._config.playback_timeout_s,
                    )
                except TimeoutError:
                    await _stop_process(process)
                    _LOGGER.warning(
                        "Windows SAPI playback timed out",
                        extra={
                            "event": "tts_failed",
                            "intent_id": utterance.intent_id,
                            "reason": "timeout",
                        },
                    )
                    return PlaybackResult(
                        intent_id=utterance.intent_id,
                        status=PlaybackStatus.FAILED,
                        started_at=started_at,
                        finished_at=datetime.now(UTC),
                        error_code="playback_timeout",
                    )
                except asyncio.CancelledError:
                    await _stop_process(process)
                    raise

                if self._cancelled_process is process:
                    return PlaybackResult(
                        intent_id=utterance.intent_id,
                        status=PlaybackStatus.CANCELLED,
                        started_at=started_at,
                        finished_at=datetime.now(UTC),
                    )
                if process.returncode != 0:
                    _LOGGER.warning(
                        "Windows SAPI playback failed",
                        extra={
                            "event": "tts_failed",
                            "intent_id": utterance.intent_id,
                            "reason": "process_exit",
                            "return_code": process.returncode,
                        },
                    )
                    return PlaybackResult(
                        intent_id=utterance.intent_id,
                        status=PlaybackStatus.FAILED,
                        started_at=started_at,
                        finished_at=datetime.now(UTC),
                        error_code=f"process_exit_{process.returncode}",
                    )
                return PlaybackResult(
                    intent_id=utterance.intent_id,
                    status=PlaybackStatus.COMPLETED,
                    started_at=started_at,
                    finished_at=datetime.now(UTC),
                )
            finally:
                if self._current_process is process:
                    self._current_process = None

    async def cancel(self) -> None:
        process = self._current_process
        if process is None or process.returncode is not None:
            return
        self._cancelled_process = process
        await _stop_process(process)
