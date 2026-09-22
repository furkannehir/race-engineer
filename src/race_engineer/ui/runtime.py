"""Qt-free runtime for the panel, using the existing live radio pipeline."""

import asyncio
import http.client
import json
import math
import os
import struct
import subprocess
import time
from collections.abc import Awaitable
from contextlib import suppress
from pathlib import Path
from typing import Any

from race_engineer.application.control import LiveControl
from race_engineer.application.live_conversation import voice_iracing
from race_engineer.config import AppConfig
from race_engineer.core.conversation import ConversationReply, RadioLanguage
from race_engineer.memory import CommunicationPreferences
from race_engineer.processes import start_owned_process
from race_engineer.stt.capture import _sounddevice
from race_engineer.tts.piper import PiperConversationSpeaker
from race_engineer.ui.settings import PanelSettings


def model_ready(port: int, expected_model: str) -> bool:
    """Literal loopback only, no redirects, proxy, or remote requests."""
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=1)
    try:
        connection.request("GET", "/v1/models")
        response = connection.getresponse()
        if response.status != 200:
            return False
        payload = json.loads(response.read(65_536))
        return any(item.get("id") == expected_model for item in payload.get("data", []))
    except (OSError, ValueError, http.client.HTTPException, AttributeError, TypeError):
        return False
    finally:
        connection.close()


class ConversationServer:
    """Reuses an existing model server; only terminates a child it created."""

    def __init__(self, root: Path, config: AppConfig, control: LiveControl) -> None:
        self.root, self.config, self.control = root, config, control
        self.process: asyncio.subprocess.Process | None = None

    async def start(self) -> None:
        port, model = self.config.conversation.port, self.config.conversation.model
        if await asyncio.to_thread(model_ready, port, model):
            return
        assets = self.root / "data/conversation-prototype"
        executable = assets / "llama-b10964-cpu/llama-server.exe"
        weights = assets / "Qwen3-4B-Instruct-2507-Q4_K_M.gguf"
        if not executable.is_file() or not weights.is_file():
            raise RuntimeError("Local conversation model missing. See docs/conversation.md.")
        self.control.emit("phase", "loading_model")
        self.process = await start_owned_process(
            str(executable),
            "--model",
            str(weights),
            "--alias",
            model,
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--cors-origins",
            f"http://127.0.0.1:{port}",
            "--ctx-size",
            "8192",
            "--parallel",
            "1",
            "--threads",
            "8",
            "--n-gpu-layers",
            "0",
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            if self.process.returncode is not None:
                raise RuntimeError(
                    "Conversation model could not start. Check port and model files."
                )
            if await asyncio.to_thread(model_ready, port, model):
                return
            await asyncio.sleep(0.25)
        raise RuntimeError("Local conversation model startup timed out.")

    async def aclose(self) -> None:
        process, self.process = self.process, None
        if process is not None:
            if process.returncode is None:
                with suppress(ProcessLookupError):
                    process.terminate()
            try:
                await asyncio.wait_for(process.wait(), 5)
            except TimeoutError:
                with suppress(ProcessLookupError):
                    process.kill()
                await process.wait()


async def run_engineer(
    root: Path,
    config_path: Path,
    config: AppConfig,
    settings: PanelSettings,
    preferences: CommunicationPreferences,
    control: LiveControl,
) -> None:
    server = ConversationServer(root, config, control)
    try:
        await run_until_stopped(server.start(), control)
        if control.stop.is_set():
            return
        control.emit("phase", "loading_speech")
        await voice_iracing(
            config_path,
            app_config=settings.apply(config, preferences),
            control=control,
            reply_language=(
                None if preferences.reply_language == "auto" else preferences.reply_language
            ),
            persist_history=True,
            history_database_path=(
                config.paths.database_path
                if config.paths.database_path.is_absolute()
                else root / config.paths.database_path
            ),
        )
    finally:
        await server.aclose()


async def test_voice(
    config: AppConfig,
    settings: PanelSettings,
    preferences: CommunicationPreferences,
    control: LiveControl,
) -> None:
    language: RadioLanguage = "tr" if preferences.reply_language == "tr" else "en"
    speaker = PiperConversationSpeaker(settings.apply(config, preferences).radio_tts)
    try:
        control.emit("phase", "testing_voice")
        await speaker.start()
        await speaker.speak(
            ConversationReply(
                language=language,
                text="Telsiz kontrolü. Seni duyuyorum."
                if language == "tr"
                else "Radio check. Your race engineer is ready.",
                status="answered",
                session_id="radio-check",
                source_sequence=0,
                mode="live",
            )
        )
        control.emit("notice", "Voice test finished. Automatic calls use Windows default output.")
    finally:
        await speaker.aclose()


async def test_microphone(settings: PanelSettings, control: LiveControl) -> None:
    """Explicit user-initiated test; compute level only, never persist samples."""
    sd = _sounddevice()
    peak = [0.0]

    def level(data: Any, frames: object, times: object, status: object) -> None:
        del frames, times, status
        raw = bytes(data)
        samples = [sample[0] / 32768 for sample in struct.iter_unpack("<h", raw)]
        if samples:
            peak[0] = max(peak[0], math.sqrt(sum(x * x for x in samples) / len(samples)))

    control.emit("phase", "testing_mic")
    # PortAudio callbacks run separately; the stream is always closed on cancellation.
    with sd.RawInputStream(
        samplerate=16000,
        channels=1,
        dtype="int16",
        device=settings.input_device,
        callback=level,
    ):
        await asyncio.sleep(3)
    db = 20 * math.log10(max(peak[0], 1e-6))
    message = f"Microphone peak: {db:.0f} dBFS. "
    message += "Signal detected." if db > -42 else "Very quiet; check your input device."
    control.emit("notice", message)


async def run_until_stopped(job: Awaitable[None], control: LiveControl) -> None:
    """Stop works during model startup, tests, capture, inference and playback."""
    task = asyncio.ensure_future(job)
    try:
        while not task.done() and not control.stop.is_set():
            await asyncio.sleep(0.05)
        if not task.done():
            task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            if not control.stop.is_set():
                raise
    finally:
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
