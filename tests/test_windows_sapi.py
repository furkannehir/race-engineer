import asyncio
import base64

from race_engineer.config import TtsConfig
from race_engineer.core.contracts import Utterance
from race_engineer.core.enums import PlaybackStatus
from race_engineer.tts import WindowsSapiTextToSpeechEngine


class CompletedProcess:
    def __init__(self, *, returncode: int = 0, stdout: bytes = b"") -> None:
        self.returncode: int | None = returncode
        self.stdout = stdout
        self.terminated = False

    async def communicate(self):
        return self.stdout, b""

    async def wait(self):
        assert self.returncode is not None
        return self.returncode

    def terminate(self) -> None:
        self.terminated = True
        self.returncode = -1

    def kill(self) -> None:
        self.returncode = -9


def test_windows_sapi_passes_text_and_settings_without_command_interpolation(
    monkeypatch,
) -> None:
    captured = {}

    async def create_process(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return CompletedProcess()

    monkeypatch.setattr(
        "race_engineer.tts.windows_sapi.asyncio.create_subprocess_exec", create_process
    )
    engine = WindowsSapiTextToSpeechEngine(
        TtsConfig(voice="Microsoft David Desktop", rate=2, volume=75)
    )
    utterance = Utterance(intent_id="intent-1", text="Green flag; $not-a-command")

    result = asyncio.run(engine.speak(utterance))

    assert result.status is PlaybackStatus.COMPLETED
    assert captured["args"][0] == "powershell.exe"
    environment = captured["kwargs"]["env"]
    assert (
        base64.b64decode(environment["RACE_ENGINEER_TTS_TEXT_B64"]).decode("utf-8")
        == utterance.text
    )
    assert environment["RACE_ENGINEER_TTS_RATE"] == "2"
    assert environment["RACE_ENGINEER_TTS_VOLUME"] == "75"


def test_windows_sapi_reports_process_failure(monkeypatch) -> None:
    async def create_process(*args, **kwargs):
        del args, kwargs
        return CompletedProcess(returncode=7)

    monkeypatch.setattr(
        "race_engineer.tts.windows_sapi.asyncio.create_subprocess_exec", create_process
    )
    engine = WindowsSapiTextToSpeechEngine(TtsConfig())

    result = asyncio.run(engine.speak(Utterance(intent_id="intent-1", text="Green flag")))

    assert result.status is PlaybackStatus.FAILED
    assert result.error_code == "process_exit_7"


def test_windows_sapi_cancel_stops_current_playback(monkeypatch) -> None:
    async def scenario() -> PlaybackStatus:
        started = asyncio.Event()
        finished = asyncio.Event()

        class BlockingProcess(CompletedProcess):
            def __init__(self) -> None:
                super().__init__()
                self.returncode = None

            async def communicate(self):
                started.set()
                await finished.wait()
                return b"", b""

            async def wait(self):
                await finished.wait()
                assert self.returncode is not None
                return self.returncode

            def terminate(self) -> None:
                self.terminated = True
                self.returncode = -1
                finished.set()

        process = BlockingProcess()

        async def create_process(*args, **kwargs):
            del args, kwargs
            return process

        monkeypatch.setattr(
            "race_engineer.tts.windows_sapi.asyncio.create_subprocess_exec",
            create_process,
        )
        engine = WindowsSapiTextToSpeechEngine(TtsConfig())
        task = asyncio.create_task(
            engine.speak(Utterance(intent_id="intent-1", text="Caution, slow down"))
        )
        await started.wait()
        await engine.cancel()
        return (await task).status

    assert asyncio.run(scenario()) is PlaybackStatus.CANCELLED


def test_windows_sapi_lists_installed_voice_names(monkeypatch) -> None:
    async def create_process(*args, **kwargs):
        del args, kwargs
        return CompletedProcess(stdout=b"Microsoft David\r\nMicrosoft Zira\r\n")

    monkeypatch.setattr(
        "race_engineer.tts.windows_sapi.asyncio.create_subprocess_exec", create_process
    )

    voices = asyncio.run(WindowsSapiTextToSpeechEngine.installed_voices())

    assert voices == ("Microsoft David", "Microsoft Zira")
