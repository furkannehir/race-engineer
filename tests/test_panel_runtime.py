import asyncio
import json
from pathlib import Path

import pytest
from test_live_conversation import context
from test_live_radio import submit
from test_speech_playback_queue import RecordingEngine

from race_engineer.application.control import LiveControl
from race_engineer.application.live_conversation import LiveBridge, voice_iracing
from race_engineer.config import AppConfig, PttBindingConfig
from race_engineer.conversation.live import LiveRaceState
from race_engineer.memory import CommunicationPreferences
from race_engineer.stt.buttons import effective_binding, legacy_binding
from race_engineer.tts.live_radio import LiveRadio
from race_engineer.ui.runtime import ConversationServer, run_engineer, run_until_stopped
from race_engineer.ui.settings import (
    PanelSettings,
    load_settings,
    load_settings_state,
    save_settings,
)

ROOT = Path(__file__).parents[1]


def test_settings_round_trip_and_explicit_config_application(tmp_path):
    path = tmp_path / "panel.json"
    settings = PanelSettings(
        ptt_binding=legacy_binding("RCTRL"),
        volume=37,
        input_device=1,
        output_device=2,
    )
    preferences = CommunicationPreferences(
        profile_id="default",
        reply_language="tr",
        announce_position_changes=False,
        announce_pit_transitions=True,
    )
    save_settings(path, settings)
    assert load_settings(path, PanelSettings()) == settings
    base = AppConfig()
    applied = settings.apply(base, preferences)
    assert effective_binding(applied.stt).label == "RCTRL" and applied.stt.input_device == 1
    assert applied.radio_tts.output_device == 2 and applied.radio_tts.volume == 0.37
    assert applied.tts.volume == 37
    assert not applied.policy.strict.announce_position_changes
    assert applied.policy.strict.announce_pit_transitions
    assert applied.policy.strict.critical_cooldown_s == base.policy.strict.critical_cooldown_s
    assert base.stt.ptt_key == "F8" and base.stt.ptt_binding is None


def test_version_one_keyboard_setting_migrates_without_rewriting_file(tmp_path):
    path = tmp_path / "panel.json"
    original = '{"version": 1, "ptt_key": "RALT", "volume": 61}'
    path.write_text(original, encoding="utf-8")
    loaded = load_settings_state(path, PanelSettings())
    assert loaded.settings.version == 3
    assert loaded.settings.ptt_binding == legacy_binding("RALT")
    assert loaded.settings.volume == 61
    assert loaded.legacy_preferences == CommunicationPreferences(profile_id="default")
    assert path.read_text(encoding="utf-8") == original


def test_version_two_driver_preferences_are_extracted_from_machine_settings(tmp_path):
    path = tmp_path / "panel.json"
    original = {
        **PanelSettings().model_dump(mode="json"),
        "version": 2,
        "reply_language": "tr",
        "announce_position_changes": False,
        "announce_pit_transitions": True,
    }
    path.write_text(json.dumps(original), encoding="utf-8")
    loaded = load_settings_state(path, PanelSettings())
    assert loaded.settings == PanelSettings()
    assert loaded.legacy_preferences == CommunicationPreferences(
        profile_id="default",
        reply_language="tr",
        announce_position_changes=False,
        announce_pit_transitions=True,
    )


def test_device_qualified_wheel_binding_round_trips(tmp_path):
    path = tmp_path / "panel.json"
    binding = PttBindingConfig(
        kind="joystick",
        code=6,
        label="Test wheel · Button 7",
        device_guid="0123456789abcdef0123456789abcdef",
        device_name="Test wheel",
        device_index=2,
    )
    save_settings(path, PanelSettings(ptt_binding=binding))
    assert load_settings(path, PanelSettings()).ptt_binding == binding


def test_corrupt_settings_never_silently_overwritten(tmp_path):
    path = tmp_path / "panel.json"
    path.write_text('{"volume": 101}', encoding="utf-8")
    with pytest.raises(ValueError):
        load_settings(path, PanelSettings())
    assert path.read_text() == '{"volume": 101}'


def test_failed_atomic_save_preserves_previous_defaults(tmp_path, monkeypatch):
    path = tmp_path / "panel.json"
    save_settings(path, PanelSettings(volume=10))

    def fail(*args):
        raise OSError("locked")

    monkeypatch.setattr(Path, "replace", fail)
    with pytest.raises(OSError):
        save_settings(path, PanelSettings(volume=20))
    assert load_settings(path, PanelSettings()).volume == 10
    assert list(tmp_path.glob("*.tmp")) == []


def test_master_mute_interrupts_active_drops_pending_and_does_not_replay():
    async def run():
        engine = RecordingEngine()
        activity = []
        radio = LiveRadio(engine, activity=activity.append)
        entered, stopped = asyncio.Event(), asyncio.Event()

        async def playing():
            entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                stopped.set()

        answer = asyncio.create_task(radio.answer(playing, 0, 10))
        await entered.wait()
        await submit(radio, "queued")
        radio.set_muted(True)
        assert await answer == "interrupted"
        assert stopped.is_set()
        assert not await submit(radio, "muted-critical", 100)
        assert await radio.answer(playing, 0, 10) == "muted"
        radio.set_muted(False)
        await submit(radio, "new-call")
        for _ in range(10):
            await asyncio.sleep(0)
        assert [utterance.text for utterance in engine.utterances] == ["new-call"]
        assert activity == ["speaking", "idle", "speaking", "idle"]
        await radio.aclose()

    asyncio.run(asyncio.wait_for(run(), 2))


def test_mute_keeps_microphone_floor():
    async def run():
        radio = LiveRadio(None)
        radio.claim_capture()
        radio.set_muted(True)
        assert not asyncio.current_task().cancelling()
        radio.release_capture()
        await radio.aclose()

    asyncio.run(run())


def test_bridge_status_is_validated_telemetry_not_raw_connection():
    events = []
    state = LiveRaceState(3)
    bridge = LiveBridge(state, LiveRadio(None), LiveControl(lambda *event: events.append(event)))
    bridge.availability_changed(True)
    assert not events
    bridge.update(context())
    bridge.update(context(1))
    bridge.availability_changed(False)
    assert events == [("telemetry", "ready"), ("telemetry", "unavailable")]


def test_stop_while_starting_waits_for_cleanup():
    async def run():
        control = LiveControl()
        started, cleaned = asyncio.Event(), asyncio.Event()

        async def job():
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                await asyncio.sleep(0)
                cleaned.set()

        task = asyncio.create_task(run_until_stopped(job(), control))
        await started.wait()
        control.stop.set()
        await task
        assert cleaned.is_set()

    asyncio.run(asyncio.wait_for(run(), 2))


def test_existing_server_is_reused_and_never_terminated(monkeypatch):
    monkeypatch.setattr("race_engineer.conversation.runtime.model_ready", lambda *_: True)

    async def forbidden(*args, **kwargs):
        raise AssertionError("must not spawn when a matching model is already running")

    monkeypatch.setattr("race_engineer.conversation.runtime.start_owned_process", forbidden)

    async def run():
        server = ConversationServer(ROOT, AppConfig(), LiveControl())
        await server.start()
        assert server.process is None
        await server.aclose()

    asyncio.run(run())


def test_owned_server_stopped_after_cancel_during_startup(tmp_path, monkeypatch):
    assets = tmp_path / "data/conversation-prototype"
    (assets / "llama-b10964-cpu").mkdir(parents=True)
    (assets / "llama-b10964-cpu/llama-server.exe").touch()
    (assets / "Qwen3-4B-Instruct-2507-Q4_K_M.gguf").touch()
    events = []

    class Process:
        returncode = None

        def terminate(self):
            events.append("terminated")
            self.returncode = 0

        async def wait(self):
            return self.returncode

    monkeypatch.setattr("race_engineer.conversation.runtime.model_ready", lambda *_: False)

    async def run():
        control = LiveControl()

        async def spawn(*args, **kwargs):
            assert "127.0.0.1" in args
            control.stop.set()
            return Process()

        monkeypatch.setattr("race_engineer.conversation.runtime.start_owned_process", spawn)
        await run_until_stopped(
            run_engineer(
                tmp_path,
                ROOT / "config/default.toml",
                AppConfig(),
                PanelSettings(),
                CommunicationPreferences(profile_id="default"),
                control,
            ),
            control,
        )

    asyncio.run(asyncio.wait_for(run(), 2))
    assert events == ["terminated"]


def test_live_control_stops_all_components_and_passes_config(monkeypatch):
    closed = []
    config = PanelSettings().apply(
        AppConfig(),
        CommunicationPreferences(profile_id="default", announce_position_changes=False),
    )
    control = LiveControl()

    async def read(*args, **kwargs):
        assert kwargs["app_config"] == config
        try:
            control.stop.set()
            await asyncio.Event().wait()
        finally:
            closed.append("telemetry")

    async def dialogue(*args):
        try:
            await asyncio.Event().wait()
        finally:
            closed.append("dialogue")

    class Recognizer:
        def __init__(self, config):
            pass

        async def aclose(self):
            closed.append("recognizer")

    monkeypatch.setattr("race_engineer.cli._read_iracing", read)
    monkeypatch.setattr("race_engineer.application.live_conversation.live_dialogue", dialogue)
    monkeypatch.setattr(
        "race_engineer.application.live_conversation.QwenSpeechRecognizer", Recognizer
    )
    asyncio.run(
        voice_iracing(
            ROOT / "config/default.toml",
            app_config=config,
            text_only=True,
            control=control,
        )
    )
    assert set(closed) == {"telemetry", "dialogue", "recognizer"}


def test_engineer_stop_does_not_cancel_live_cleanup_a_second_time(monkeypatch):
    events = []
    monkeypatch.setattr("race_engineer.conversation.runtime.model_ready", lambda *_: True)

    async def live(*args, **kwargs):
        assert kwargs["persist_history"] is True
        assert kwargs["history_database_path"] == ROOT / "data/race_engineer.sqlite3"
        control = kwargs["control"]
        control.stop.set()
        # Simulate live mode's own cooperative shutdown; this must not be cancelled
        # by another generic stop watcher while resources are being closed.
        await asyncio.sleep(0.1)
        events.append("cleanup completed")

    monkeypatch.setattr("race_engineer.ui.runtime.voice_iracing", live)
    asyncio.run(
        run_engineer(
            ROOT,
            ROOT / "config/default.toml",
            AppConfig(),
            PanelSettings(),
            CommunicationPreferences(profile_id="default"),
            LiveControl(),
        )
    )
    assert events == ["cleanup completed"]
