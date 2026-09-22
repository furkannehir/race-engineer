import json
import os
from pathlib import Path

import pytest

os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ["QT_SCALE_FACTOR"] = "1"
pytest.importorskip("PySide6")
pytest.importorskip("qtawesome")

from PySide6.QtCore import QObject, Qt, QTimer, Signal
from PySide6.QtGui import QFontDatabase
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QCheckBox, QDialog, QDialogButtonBox

from race_engineer.application.control import LiveControl
from race_engineer.config import PttBindingConfig
from race_engineer.core.speech_input import SpeechInputError
from race_engineer.memory import SqliteDriverProfileRepository
from race_engineer.ui.app import BindingDialog, RadioDesk
from race_engineer.ui.settings import PanelSettings, load_settings, save_settings

ROOT = Path(__file__).parents[1]


@pytest.fixture(scope="module")
def app():
    app = QApplication.instance() or QApplication([])
    app.setStyle("Fusion")
    if os.name == "nt":
        fonts = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts"
        for name in ("segoeui.ttf", "segoeuib.ttf", "seguisb.ttf"):
            QFontDatabase.addApplicationFont(str(fonts / name))
    return app


@pytest.fixture
def panel(app, tmp_path, monkeypatch):
    inputs = [{"name": "Test microphone", "host_api": "Test", "index": 1}]
    outputs = [{"name": "Test headset", "host_api": "Test", "index": 2}]
    monkeypatch.setattr("race_engineer.ui.app.input_devices", lambda: inputs)
    monkeypatch.setattr("race_engineer.ui.app.output_devices", lambda: outputs)
    repository = SqliteDriverProfileRepository(tmp_path / "driver-profile.sqlite3")
    widget = RadioDesk(
        ROOT,
        ROOT / "config/default.toml",
        tmp_path / "panel.json",
        devices=(inputs, outputs),
        profile_repository=repository,
    )
    widget.show()
    app.processEvents()
    yield widget
    widget.worker = None
    widget.close()
    app.processEvents()


class FakeWorker(QObject):
    status = Signal(str, str)
    failed = Signal(str)
    finished = Signal()
    calls = 0

    def __init__(self, root, path, config, settings, preferences, mode, muted):
        super().__init__()
        self.control = LiveControl()
        self.settings, self.preferences, self.mode = settings, preferences, mode
        if muted:
            self.control.muted.set()

    def start(self):
        FakeWorker.calls += 1
        self.status.emit("models", "ready")
        self.status.emit("phase", "ready")


def test_opening_panel_never_claims_connected_or_loads_models(panel):
    assert panel.worker is None
    assert panel.heading.text() == "Ready to start"
    assert panel.telemetry.text.text() == "iRacing not connected"
    assert panel.start_button.text() == "Start engineer"


def test_inputs_persist_and_volume_applies_to_both_engines(panel):
    panel.microphone.setCurrentIndex(1)
    panel.output.setCurrentIndex(1)
    panel.volume.setValue(42)
    panel.language.setCurrentIndex(2)
    settings = load_settings(panel.settings_path, PanelSettings())
    preferences = panel.profile_repository.preferences(panel.profile.profile_id)
    assert settings.input_device == 1 and settings.output_device == 2
    assert settings.input_name == "Test microphone (Test)"
    assert preferences.reply_language == "tr" and settings.volume == 42
    assert not {
        "reply_language",
        "announce_position_changes",
        "announce_pit_transitions",
    } & json.loads(panel.settings_path.read_text(encoding="utf-8")).keys()
    assert panel.volume_label.text() == "42%"


def test_start_stop_locks_settings_and_keeps_mute_available(panel, monkeypatch):
    monkeypatch.setattr("race_engineer.ui.app.EngineWorker", FakeWorker)
    QTest.mouseClick(panel.start_button, Qt.MouseButton.LeftButton)
    worker = panel.worker
    assert worker is not None and not panel.microphone.isEnabled()
    assert panel.heading.text() == "Waiting for iRacing"
    worker.status.emit("telemetry", "ready")
    assert panel.heading.text() == "Radio ready"
    QTest.mouseClick(panel.mute_button, Qt.MouseButton.LeftButton)
    assert worker.control.muted.is_set() and panel.tray_mute.isChecked()
    assert panel.heading.text() == "Audio muted"
    QTest.mouseClick(panel.start_button, Qt.MouseButton.LeftButton)
    assert worker.control.stop.is_set()
    assert not panel.start_button.isEnabled()
    worker.finished.emit()
    assert panel.worker is None and panel.microphone.isEnabled()
    assert panel.heading.text() == "Ready to start"


def test_no_double_start_and_worker_errors_return_to_safe_stopped_state(panel, monkeypatch):
    FakeWorker.calls = 0
    monkeypatch.setattr("race_engineer.ui.app.EngineWorker", FakeWorker)
    panel._start("engineer")
    worker = panel.worker
    panel._start("engineer")
    assert FakeWorker.calls == 1
    worker.failed.emit("Model missing")
    worker.finished.emit()
    assert panel.heading.text() == "Needs attention"
    assert "Model missing" in panel.notice.text()
    assert panel.start_button.isEnabled()


def test_test_buttons_use_test_jobs_and_can_be_cancelled(panel, monkeypatch):
    monkeypatch.setattr("race_engineer.ui.app.EngineWorker", FakeWorker)
    QTest.mouseClick(panel.test_mic, Qt.MouseButton.LeftButton)
    worker = panel.worker
    assert worker.mode == "mic" and panel.start_button.text() == "Cancel test"
    QTest.mouseClick(panel.start_button, Qt.MouseButton.LeftButton)
    worker.finished.emit()
    QTest.mouseClick(panel.test_voice, Qt.MouseButton.LeftButton)
    assert panel.worker.mode == "voice"
    panel.mute_button.setChecked(True)
    assert panel.worker.control.stop.is_set()
    panel.worker.finished.emit()
    panel._start("voice")
    assert panel.worker is None and "Turn off" in panel.notice.text()


def test_binding_dialog_accepts_any_keyboard_key_and_escape(panel):
    dialog = BindingDialog(PanelSettings().ptt_binding, panel, detector_factory=None)
    dialog.show()
    QTest.keyClick(dialog, Qt.Key.Key_F9)
    assert dialog.binding.label == "F9"
    QTest.keyClick(dialog, Qt.Key.Key_A)
    assert dialog.binding.code == ord("A") and dialog.binding.label == "A"
    QTest.keyClick(dialog, Qt.Key.Key_Escape)
    assert dialog.binding.code == 0x1B
    buttons = dialog.findChild(QDialogButtonBox)
    QTest.mouseClick(
        buttons.button(QDialogButtonBox.StandardButton.Cancel), Qt.MouseButton.LeftButton
    )
    assert dialog.result() == QDialog.DialogCode.Rejected


def test_binding_dialog_accepts_mouse_and_wheel_buttons(panel):
    wheel = PttBindingConfig(
        kind="joystick",
        code=6,
        label="Test wheel · Button 7",
        device_guid="0123456789abcdef0123456789abcdef",
        device_name="Test wheel",
        device_index=2,
    )

    class Detector:
        def __init__(self):
            self.closed = False
            self.binding = wheel

        def poll(self):
            result, self.binding = self.binding, None
            return result

        def close(self):
            self.closed = True

    detector = Detector()
    dialog = BindingDialog(
        PanelSettings().ptt_binding, panel, detector_factory=lambda: detector
    )
    dialog.show()
    QTest.mouseClick(dialog.capture, Qt.MouseButton.BackButton)
    assert dialog.binding.kind == "mouse" and dialog.binding.code == 0x05
    dialog._poll_controller()
    assert dialog.binding == wheel
    dialog.reject()
    assert detector.closed


def test_preferences_dialog_saves_only_explicit_supported_toggles(panel, app):
    def edit():
        dialog = app.activeModalWidget()
        boxes = dialog.findChildren(QCheckBox)
        boxes[0].setChecked(False)
        boxes[1].setChecked(True)
        buttons = dialog.findChild(QDialogButtonBox)
        QTest.mouseClick(
            buttons.button(QDialogButtonBox.StandardButton.Save), Qt.MouseButton.LeftButton
        )

    QTimer.singleShot(0, edit)
    panel._preferences()
    preferences = panel.profile_repository.preferences(panel.profile.profile_id)
    assert not preferences.announce_position_changes
    assert preferences.announce_pit_transitions


def test_stale_device_identity_fails_closed(panel, monkeypatch):
    panel.microphone.setCurrentIndex(1)
    monkeypatch.setattr(
        "race_engineer.ui.app.input_devices",
        lambda: [{"name": "Different microphone", "host_api": "Test", "index": 1}],
    )
    panel._start("engineer")
    assert panel.worker is None and "devices changed" in panel.notice.text()


def test_missing_saved_wheel_fails_before_start_with_rebind_message(panel, monkeypatch):
    panel.settings = panel.settings.model_copy(
        update={
            "ptt_binding": PttBindingConfig(
                kind="joystick",
                code=0,
                label="Missing wheel · Button 1",
                device_guid="0123456789abcdef0123456789abcdef",
                device_name="Missing wheel",
                device_index=0,
            )
        }
    )

    class MissingController:
        def __init__(self, binding):
            raise SpeechInputError("push_to_talk_device_unavailable")

    monkeypatch.setattr("race_engineer.ui.app.JoystickButtonInput", MissingController)
    panel._start("engineer")
    assert panel.worker is None
    assert "controller is disconnected or changed" in panel.notice.text()


def test_device_reordering_is_resolved_by_name_at_load(app, tmp_path):
    path = tmp_path / "panel.json"
    save_settings(path, PanelSettings(input_device=1, input_name="Mic (Test)"))
    panel = RadioDesk(
        ROOT,
        ROOT / "config/default.toml",
        path,
        devices=([{"name": "Mic", "host_api": "Test", "index": 5}], []),
        profile_repository=SqliteDriverProfileRepository(tmp_path / "profile.sqlite3"),
    )
    assert panel.microphone.currentData() == 5
    panel.close()


def test_legacy_panel_preferences_are_imported_once_then_sqlite_wins(app, tmp_path):
    path = tmp_path / "panel.json"
    legacy = {
        **PanelSettings().model_dump(mode="json"),
        "version": 2,
        "reply_language": "tr",
        "announce_position_changes": False,
        "announce_pit_transitions": True,
    }
    path.write_text(json.dumps(legacy), encoding="utf-8")
    repository = SqliteDriverProfileRepository(tmp_path / "profile.sqlite3")
    panel = RadioDesk(
        ROOT,
        ROOT / "config/default.toml",
        path,
        devices=([], []),
        profile_repository=repository,
    )
    profile_id = panel.profile.profile_id
    imported = repository.preferences(profile_id)
    assert imported.reply_language == "tr"
    assert not imported.announce_position_changes and imported.announce_pit_transitions
    assert json.loads(path.read_text(encoding="utf-8"))["version"] == 3
    panel.close()

    conflicting = {**legacy, "reply_language": "en", "announce_position_changes": True}
    path.write_text(json.dumps(conflicting), encoding="utf-8")
    reopened = RadioDesk(
        ROOT,
        ROOT / "config/default.toml",
        path,
        devices=([], []),
        profile_repository=repository,
    )
    persisted = repository.preferences(profile_id)
    assert persisted.reply_language == "tr"
    assert not persisted.announce_position_changes and persisted.announce_pit_transitions
    assert reopened.language.currentData() == "tr"
    assert json.loads(path.read_text(encoding="utf-8"))["version"] == 3
    reopened.close()


def test_preview_never_starts_hardware_or_persists_settings(app, tmp_path):
    path = tmp_path / "preview.json"
    panel = RadioDesk(ROOT, ROOT / "config/default.toml", path, preview=True)
    panel._start("mic")
    panel.volume.setValue(12)
    assert panel.worker is None and not path.exists()
    assert "Design preview" in panel.notice.text()
    panel.close()
