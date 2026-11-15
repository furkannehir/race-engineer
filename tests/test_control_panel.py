import json
import logging
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
from PySide6.QtWidgets import QApplication, QCheckBox, QDialog, QDialogButtonBox, QMessageBox

from race_engineer.application.control import LiveControl
from race_engineer.config import PttBindingConfig
from race_engineer.core.speech_input import SpeechInputError
from race_engineer.installation import Installation
from race_engineer.memory import SqliteDriverProfileRepository
from race_engineer.ui import app as panel_app
from race_engineer.ui.app import BindingDialog, RadioDesk
from race_engineer.ui.branding import PRODUCT_NAME
from race_engineer.ui.settings import PanelSettings, load_settings, save_settings

ROOT = Path(__file__).parents[1]


def test_speech_status_distinguishes_preparation_playback_and_idle(panel):
    panel._status("phase", "ready")
    panel._status("speech", "preparing")
    assert panel.heading.text() == "Preparing speech…"
    panel._status("speech", "speaking")
    assert panel.heading.text() == "Speaking…"
    panel._status("speech", "idle")
    assert panel.heading.text() != "Speaking…"
    panel._status("speech", "preparing")
    panel._status("phase", "stopping")
    assert panel.heading.text() == "Stopping…"


def test_failed_voice_remains_visible_while_waiting_for_iracing_and_clears_on_retry(panel):
    panel._status("telemetry", "waiting")
    panel._status("models", "no_reply_voice")
    panel._status("phase", "ready")
    assert panel.heading.text() == "Speech unavailable"
    assert panel.models.text.text() == "Local speech unavailable"
    panel._status("telemetry", "ready")
    assert panel.heading.text() == "Speech unavailable"
    panel._status("models", "ready")
    assert panel.heading.text() == "Radio ready" and panel._models_ready


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
    assert panel.windowTitle() == PRODUCT_NAME
    assert not panel.windowIcon().isNull()
    assert panel.brand_logo.accessibleName() == PRODUCT_NAME
    assert panel.brand_logo.pixmap() is not None
    assert not panel.brand_logo.pixmap().isNull()
    assert panel.tray.toolTip() == f"{PRODUCT_NAME} — Ready to start"
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


def test_packaged_first_launch_requires_local_component_setup(app, tmp_path):
    installation = Installation.discover(
        environ={"LOCALAPPDATA": str(tmp_path / "local")},
        executable=tmp_path / "program/Pitward.exe",
        bundle_root=ROOT,
        frozen=True,
    )
    panel = RadioDesk(
        installation.install_root,
        ROOT / "config/default.toml",
        installation.settings_path,
        devices=([], []),
        installation=installation,
    )
    panel.show()
    app.processEvents()
    assert not panel.start_button.isEnabled()
    assert panel.setup_components.isVisible()
    assert "not installed" in panel.notice.text()
    panel.close()


@pytest.fixture
def component_setup(app, tmp_path, monkeypatch):
    installation = Installation.discover(
        environ={"PITWARD_DATA_DIR": str(tmp_path / "custom data")},
        executable=tmp_path / "program/Pitward.exe",
        bundle_root=ROOT,
        frozen=True,
    )
    installation.install_root.mkdir(parents=True)
    installation.component_installer_path.touch()
    widget = RadioDesk(
        installation.install_root,
        ROOT / "config/default.toml",
        installation.settings_path,
        devices=([], []),
        installation=installation,
    )
    widget.show()
    app.processEvents()

    class Download:
        returncode = None

        def poll(self):
            return self.returncode

    process = Download()
    calls = []

    def launch(*args, **kwargs):
        calls.append((args, kwargs))
        return process

    monkeypatch.setattr(panel_app.subprocess, "Popen", launch)
    monkeypatch.setattr(
        panel_app.QMessageBox, "question", lambda *args: QMessageBox.StandardButton.Yes
    )
    yield widget, process, calls
    widget.close()
    app.processEvents()


def complete_components(widget):
    for path in widget._missing_components:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()


def test_component_setup_waits_without_blocking_and_restarts_once_after_success(component_setup):
    widget, process, calls = component_setup
    completed = []
    widget.restart_ready.connect(lambda: completed.append(True))
    widget._setup_local_components()
    widget._setup_local_components()
    widget._poll_component_setup()
    assert len(calls) == 1 and not widget.restart_requested
    assert widget._component_setup_timer.isActive()
    assert not widget.setup_components.isEnabled() and not widget.test_mic.isEnabled()
    assert not widget.test_voice.isEnabled()
    assert "--root" in calls[0][0][0]
    assert calls[0][0][0][-1] == str(widget.installation.components_root)
    widget._start("mic")
    assert widget.worker is None
    widget.volume.setValue(42)
    widget.hide()  # Completion must also request exit when minimized to the tray.
    complete_components(widget)
    process.returncode = 0
    widget._poll_component_setup()
    widget._poll_component_setup()
    assert widget.restart_requested and completed == [True]
    assert not widget.isVisible() and not widget._component_setup_timer.isActive()
    assert load_settings(widget.settings_path, PanelSettings()).volume == 42


@pytest.mark.parametrize("exit_code", [1, 130])
def test_failed_or_cancelled_download_stays_open_and_can_retry(component_setup, exit_code):
    widget, process, calls = component_setup
    widget._setup_local_components()
    process.returncode = exit_code
    widget._poll_component_setup()
    assert not widget.restart_requested and widget.isVisible()
    assert not widget._component_setup_timer.isActive()
    assert widget.setup_components.isEnabled()
    assert "Retry setup" in widget.notice.text()
    process.returncode = None
    widget._setup_local_components()
    assert len(calls) == 2 and widget._component_setup_timer.isActive()


def test_successful_downloader_with_missing_files_does_not_restart(component_setup):
    widget, process, _ = component_setup
    widget._setup_local_components()
    process.returncode = 0
    widget._poll_component_setup()
    assert not widget.restart_requested and widget.isVisible()
    assert "still missing" in widget.notice.text()
    assert widget.setup_components.isEnabled() and not widget.start_button.isEnabled()


def test_closing_during_download_does_not_reopen_the_panel(component_setup):
    widget, process, _ = component_setup
    widget._setup_local_components()
    widget.close()
    complete_components(widget)
    process.returncode = 0
    widget._poll_component_setup()
    assert not widget.restart_requested and not widget.isVisible()
    assert not widget._component_setup_timer.isActive()


def test_completed_setup_stays_open_if_settings_cannot_be_saved(component_setup, monkeypatch):
    widget, process, _ = component_setup
    widget._setup_local_components()
    complete_components(widget)

    def fail(*args):
        raise OSError("settings are read-only")

    monkeypatch.setattr(panel_app, "save_settings", fail)
    process.returncode = 0
    widget._poll_component_setup()
    assert not widget.restart_requested and widget.isVisible()
    assert "Settings not saved" in widget.notice.text()


def test_setup_cannot_start_when_previewing_or_an_engineer_is_active(component_setup):
    widget, _, calls = component_setup
    widget.preview = True
    widget._setup_local_components()
    widget.preview = False
    widget.worker = object()
    widget._setup_local_components()
    widget.worker = None
    assert not calls


def test_setup_launch_failure_keeps_the_panel_available_for_retry(component_setup, monkeypatch):
    widget, _, _ = component_setup

    def fail(*args, **kwargs):
        raise OSError("cannot launch")

    monkeypatch.setattr(panel_app.subprocess, "Popen", fail)
    widget._setup_local_components()
    assert widget.setup_components.isEnabled() and widget.isVisible()
    assert widget._component_setup_process is None and not widget.restart_requested
    assert "could not be started" in widget.notice.text()


@pytest.mark.parametrize(
    "restart,launch_error,install_update",
    [(False, False, False), (True, False, False), (True, True, False),
     (False, False, True), (False, True, True)],
)
def test_restart_handoff_releases_lock_and_preserves_launch_paths(
    app, tmp_path, monkeypatch, restart, launch_error, install_update
):
    layout = Installation.discover(
        environ={"PITWARD_DATA_DIR": str(tmp_path / "custom data")},
        executable=tmp_path / "program/Pitward.exe",
        bundle_root=ROOT,
        frozen=True,
    )
    custom_config, settings = tmp_path / "my config.toml", tmp_path / "my settings.json"
    events, errors = [], []

    class Lock:
        def __init__(self, path):
            pass

        def tryLock(self, timeout):
            return True

        def unlock(self):
            events.append("unlock")

    class FinishedPanel(QObject):
        restart_ready = Signal()
        installer_ready = Signal()
        restart_requested = restart
        installer_requested = tmp_path / "verified update.exe" if install_update else None

        def __init__(self, *args, **kwargs):
            super().__init__()

        def show(self):
            pass

    def launch(command, **kwargs):
        assert events == ["unlock"]
        if install_update:
            assert command == panel_app.installer_command(
                FinishedPanel.installer_requested, layout.install_root, custom_config, settings
            )
            assert kwargs["cwd"] == tmp_path
        else:
            assert command == [
                str(layout.install_root / "Pitward.exe"),
                "--config", str(custom_config), "--settings", str(settings),
            ]
            assert kwargs["cwd"] == layout.install_root
        assert kwargs["env"]["PYINSTALLER_RESET_ENVIRONMENT"] == "1"
        assert kwargs["env"]["PITWARD_DATA_DIR"] == str(layout.data_root)
        events.append("launch")
        if launch_error:
            raise OSError("cannot relaunch")

    monkeypatch.setattr(
        panel_app.sys,
        "argv",
        ["panel", "--config", str(custom_config), "--settings", str(settings)],
    )
    monkeypatch.setattr(panel_app.sys, "executable", str(layout.install_root / "Pitward.exe"))
    monkeypatch.setenv("PITWARD_DATA_DIR", str(layout.data_root))
    monkeypatch.setattr(panel_app.Installation, "discover", lambda: layout)
    monkeypatch.setattr(panel_app, "QLockFile", Lock)
    monkeypatch.setattr(panel_app, "RadioDesk", FinishedPanel)
    monkeypatch.setattr(app, "exec", lambda: 0)
    monkeypatch.setattr(panel_app.subprocess, "Popen", launch)
    monkeypatch.setattr(panel_app.QMessageBox, "critical", lambda *args: errors.append(args[-1]))
    handler, logger = logging.NullHandler(), logging.getLogger()
    old_level = logger.level
    monkeypatch.setattr(panel_app, "RotatingFileHandler", lambda *args, **kwargs: handler)
    try:
        assert panel_app.main() == (1 if launch_error else 0)
        assert events == (["unlock", "launch"] if restart or install_update else ["unlock"])
        assert bool(errors) == launch_error
        if launch_error:
            message = (
                "installer could not start" if install_update else "Open Pitward again manually"
            )
            assert message in errors[0]
    finally:
        logger.removeHandler(handler)
        logger.setLevel(old_level)
