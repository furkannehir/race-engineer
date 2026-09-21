"""Native, optional Windows radio desk. No hardware is opened until a user starts it."""

import argparse
import asyncio
import logging
import os
import sys
from collections.abc import Callable
from logging.handlers import RotatingFileHandler
from pathlib import Path

from PySide6.QtCore import QLockFile, QSize, Qt, QThread, QTimer, Signal
from PySide6.QtGui import QAction, QCloseEvent, QFont, QKeyEvent, QKeySequence, QMouseEvent
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMenu,
    QMessageBox,
    QPushButton,
    QSlider,
    QSystemTrayIcon,
    QVBoxLayout,
    QWidget,
)

from race_engineer.application.control import LiveControl
from race_engineer.config import AppConfig, PttBindingConfig, load_config
from race_engineer.core.speech_input import SpeechInputError
from race_engineer.observability import JsonFormatter
from race_engineer.stt.buttons import (
    JoystickBindingDetector,
    JoystickButtonInput,
    compact_binding_label,
    mouse_binding,
)
from race_engineer.stt.capture import input_devices
from race_engineer.tts.devices import output_devices
from race_engineer.ui import icons as qta
from race_engineer.ui.runtime import (
    run_engineer,
    run_until_stopped,
    test_microphone,
    test_voice,
)
from race_engineer.ui.settings import PanelSettings, load_settings, save_settings

_ACCENT = "#2dc9c0"
_MUTED = "#a7b0bd"
_STYLE = """
QWidget { background: #1d242a; color: #f3f4f5; font-family: 'Segoe UI'; font-size: 16px; }
QLabel { background: transparent; }
QLabel[role='title'] { font-size: 36px; font-weight: 700; }
QLabel[role='muted'] { color: #a7b0bd; font-size: 14px; }
QLabel[role='eyebrow'] { color: #a7b0bd; font-size: 14px; font-weight: 600; }
QLabel[role='key'] { border: 1px solid #a7b0bd; border-radius: 6px;
    font-size: 38px; font-weight: 700; }
QLabel[role='instruction'] { font-size: 24px; font-weight: 600; }
QFrame[role='divider'] { background: #38434d; max-height: 1px; min-height: 1px; }
QFrame[role='vertical'] { background: #38434d; max-width: 1px; min-width: 1px; }
QFrame[role='capture'] { background: #20282f; border: 1px solid #64717e;
    border-radius: 6px; }
QFrame[role='capture']:focus { border: 2px solid #2dc9c0; }
QPushButton { background: transparent; border: 1px solid #64717e;
    border-radius: 4px; padding: 9px 15px; min-height: 20px; }
QPushButton:hover { background: #303b43; border-color: #a7b0bd; }
QPushButton:focus, QComboBox:focus { border: 2px solid #2dc9c0; }
QPushButton:disabled, QComboBox:disabled { color: #929aa5; border-color: #3c4852; }
QPushButton[role='primary'] { background: #2dc9c0; color: #081b1d; border: none;
    font-weight: 600; padding: 12px 20px; }
QPushButton[role='primary']:hover { background: #5cddd5; }
QPushButton[role='link'] { border: none; color: #80bfff; text-decoration: underline;
    padding: 0px; text-align: left; }
QPushButton[role='switch'] { border: none; padding: 0px; }
QComboBox { background: #20282f; border: 1px solid #53616e; border-radius: 4px;
    padding: 9px 10px; min-height: 20px; }
QComboBox QAbstractItemView { background: #253039; selection-background-color: #355c63;
    color: #f3f4f5; min-width: 220px; }
QSlider::groove:horizontal { height: 5px; background: #5b646e; border-radius: 2px; border: 0px; }
QSlider::sub-page:horizontal { background: #2dc9c0; border-radius: 2px; border: 0px; }
QSlider::add-page:horizontal { background: #5b646e; border-radius: 2px; border: 0px; }
QSlider::handle:horizontal { background: #2dc9c0; width: 18px; margin: -7px 0px;
    border-radius: 9px; }
QSlider { background: transparent; border: none; }
QSlider:focus { border: 1px solid #2dc9c0; }
QToolTip { background: #29343e; color: #ffffff; border: 1px solid #64717e; }
QMenu { background: #253039; padding: 6px; }
QMenu::item { padding: 8px 24px; }
QMenu::item:selected { background: #355c63; }
QCheckBox { spacing: 10px; padding: 6px; }
QCheckBox::indicator { width: 18px; height: 18px; }
QCheckBox::indicator:unchecked { background: #1d242a; border: 1px solid #83909c;
    border-radius: 2px; }
"""


def label(text: str, role: str = "") -> QLabel:
    widget = QLabel(text)
    if role:
        widget.setProperty("role", role)
    return widget


def divider(vertical: bool = False) -> QFrame:
    widget = QFrame()
    widget.setProperty("role", "vertical" if vertical else "divider")
    return widget


def _friendly_error(error: Exception) -> str:
    messages = {
        "push_to_talk_controller_dependency_missing": (
            "Controller support is not installed. Reinstall the desktop dependencies."
        ),
        "push_to_talk_device_unavailable": (
            "Push-to-talk controller is disconnected or changed. Bind it again."
        ),
        "push_to_talk_device_ambiguous": (
            "Push-to-talk controller identity is ambiguous. Bind the button again."
        ),
        "push_to_talk_button_unavailable": (
            "The saved push-to-talk control is no longer available. Bind it again."
        ),
        "push_to_talk_controller_open_failed": (
            "Push-to-talk controller could not be opened. Reconnect it and try again."
        ),
        "push_to_talk_controller_read_failed": (
            "Push-to-talk controller disconnected. Stop and bind it again."
        ),
    }
    return messages.get(str(error), str(error) or type(error).__name__)


class StatusRow(QWidget):
    def __init__(self, text: str) -> None:
        super().__init__()
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(14)
        self.icon = QLabel()
        self.text = label(text)
        layout.addWidget(self.icon)
        layout.addWidget(self.text, 1)
        self.update_status(text, False)

    def update_status(self, text: str, ready: bool) -> None:
        self.icon.setPixmap(
            qta.icon(
                "mdi6.check-circle" if ready else "mdi6.circle-outline",
                color=_ACCENT if ready else _MUTED,
            ).pixmap(32, 32)
        )
        self.text.setText(text)


def _keyboard_binding(event: QKeyEvent) -> PttBindingConfig | None:
    key = int(event.key())
    native = int(event.nativeVirtualKey())
    fallback = {
        int(Qt.Key.Key_Backspace): 0x08,
        int(Qt.Key.Key_Tab): 0x09,
        int(Qt.Key.Key_Return): 0x0D,
        int(Qt.Key.Key_Enter): 0x0D,
        int(Qt.Key.Key_Shift): 0x10,
        int(Qt.Key.Key_Control): 0x11,
        int(Qt.Key.Key_Alt): 0x12,
        int(Qt.Key.Key_Pause): 0x13,
        int(Qt.Key.Key_CapsLock): 0x14,
        int(Qt.Key.Key_Escape): 0x1B,
        int(Qt.Key.Key_Space): 0x20,
        int(Qt.Key.Key_PageUp): 0x21,
        int(Qt.Key.Key_PageDown): 0x22,
        int(Qt.Key.Key_End): 0x23,
        int(Qt.Key.Key_Home): 0x24,
        int(Qt.Key.Key_Left): 0x25,
        int(Qt.Key.Key_Up): 0x26,
        int(Qt.Key.Key_Right): 0x27,
        int(Qt.Key.Key_Down): 0x28,
        int(Qt.Key.Key_Insert): 0x2D,
        int(Qt.Key.Key_Delete): 0x2E,
    }
    if int(Qt.Key.Key_F1) <= key <= int(Qt.Key.Key_F24):
        fallback[key] = 0x70 + key - int(Qt.Key.Key_F1)
    if 0x30 <= key <= 0x5A:
        fallback[key] = key
    code = native or fallback.get(key)
    if code is None or not 1 <= code <= 0xFF:
        return None
    text = QKeySequence(key).toString(QKeySequence.SequenceFormat.NativeText).strip()
    aliases = {0x20: "SPACE", 0xA3: "RCTRL", 0xA5: "RALT"}
    label_text = aliases.get(code, text or event.text().strip() or f"Key {code:02X}")
    return PttBindingConfig(kind="keyboard", code=code, label=label_text)


class BindingCapture(QFrame):
    mouse_button_pressed = Signal(int)

    def __init__(self) -> None:
        super().__init__()
        self.setProperty("role", "capture")
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setAccessibleName("Push-to-talk button capture")
        self.setMinimumHeight(112)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 16, 18, 16)
        self.value = label("Waiting for input…", "instruction")
        self.value.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.value.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.detail = label("Press and hold the control you want to use.", "muted")
        self.detail.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.detail.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        layout.addWidget(self.value)
        layout.addWidget(self.detail)

    def show_binding(self, binding: PttBindingConfig) -> None:
        self.value.setText(binding.label)
        kinds = {"keyboard": "Keyboard", "mouse": "Mouse", "joystick": "Wheel / controller"}
        self.detail.setText(f"Selected · {kinds[binding.kind]}")

    def mousePressEvent(self, event: QMouseEvent) -> None:
        codes = {
            Qt.MouseButton.LeftButton: 0x01,
            Qt.MouseButton.RightButton: 0x02,
            Qt.MouseButton.MiddleButton: 0x04,
            Qt.MouseButton.BackButton: 0x05,
            Qt.MouseButton.ForwardButton: 0x06,
        }
        code = codes.get(event.button())
        if code is not None:
            self.mouse_button_pressed.emit(code)
            event.accept()
            return
        super().mousePressEvent(event)


class BindingDialog(QDialog):
    def __init__(
        self,
        current: PttBindingConfig,
        parent: QWidget,
        *,
        detector_factory: Callable[[], JoystickBindingDetector] | None = JoystickBindingDetector,
    ) -> None:
        super().__init__(parent)
        self.binding = current
        self._detector: JoystickBindingDetector | None = None
        self.setWindowTitle("Push-to-talk binding")
        self.setMinimumWidth(520)
        layout = QVBoxLayout(self)
        layout.setSpacing(18)
        layout.addWidget(label("Press any button", "instruction"))
        layout.addWidget(
            label("Keyboard, mouse, steering wheel, button box, joystick, or gamepad.")
        )
        self.capture = BindingCapture()
        self.capture.show_binding(current)
        self.capture.mouse_button_pressed.connect(
            lambda code: self._select(mouse_binding(code))
        )
        layout.addWidget(self.capture)
        help_text = label(
            "Hold-to-talk uses digital buttons only. Steering, pedals, and other analog axes "
            "are ignored. Choose a control not used in iRacing.",
            "muted",
        )
        help_text.setWordWrap(True)
        layout.addWidget(help_text)
        self.controller_status = label("Listening for wheel and controller buttons…", "muted")
        layout.addWidget(self.controller_status)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.capture.setFocus()
        self._timer = QTimer(self)
        self._timer.setInterval(20)
        self._timer.timeout.connect(self._poll_controller)
        if detector_factory is not None:
            try:
                self._detector = detector_factory()
                self._timer.start()
            except SpeechInputError:
                self.controller_status.setText(
                    "Wheel/controller input unavailable. Keyboard and mouse still work."
                )
        else:
            self.controller_status.setText("Wheel/controller detection disabled in preview.")

    def _select(self, binding: PttBindingConfig) -> None:
        self.binding = binding
        self.capture.show_binding(binding)
        self.capture.setFocus()

    def _poll_controller(self) -> None:
        assert self._detector is not None
        try:
            binding = self._detector.poll()
            if binding is not None:
                self._select(binding)
        except SpeechInputError:
            self._timer.stop()
            self.controller_status.setText(
                "Wheel/controller disconnected. Reconnect it and reopen this window."
            )

    def keyPressEvent(self, event: QKeyEvent) -> None:
        binding = _keyboard_binding(event)
        if binding is not None and not event.isAutoRepeat():
            self._select(binding)
            event.accept()
        else:
            super().keyPressEvent(event)

    def done(self, result: int) -> None:
        self._timer.stop()
        if self._detector is not None:
            self._detector.close()
            self._detector = None
        super().done(result)


class EngineWorker(QThread):
    status = Signal(str, str)
    failed = Signal(str)

    def __init__(
        self,
        root: Path,
        config_path: Path,
        config: AppConfig,
        settings: PanelSettings,
        mode: str,
        muted: bool,
    ) -> None:
        super().__init__()
        self.root, self.config_path, self.config = root, config_path, config
        self.settings, self.mode = settings, mode
        self.control = LiveControl(self.status.emit)
        if muted:
            self.control.muted.set()

    def run(self) -> None:
        async def work() -> None:
            if self.mode == "mic":
                job = test_microphone(self.settings, self.control)
            elif self.mode == "voice":
                job = test_voice(self.config, self.settings, self.control)
            else:
                job = run_engineer(
                    self.root, self.config_path, self.config, self.settings, self.control
                )
            if self.mode == "engineer":
                # Live mode owns graceful shutdown. Do not cancel it a second time
                # while it is closing microphone/model workers.
                await job
            else:
                await run_until_stopped(job, self.control)

        try:
            asyncio.run(work())
        except Exception as error:
            logging.getLogger(__name__).error(
                "panel worker stopped",
                extra={"event": "panel_error", "reason": type(error).__name__},
            )
            self.failed.emit(_friendly_error(error))


class RadioDesk(QWidget):
    def __init__(
        self,
        root: Path,
        config_path: Path,
        settings_path: Path,
        *,
        preview: bool = False,
        devices: tuple[list[dict[str, object]], list[dict[str, object]]] | None = None,
    ) -> None:
        super().__init__()
        self.root, self.config_path, self.settings_path = root, config_path, settings_path
        self.config = load_config(config_path)
        self.settings = PanelSettings.from_config(self.config)
        self._settings_error = ""
        try:
            self.settings = load_settings(settings_path, self.settings)
        except (OSError, ValueError):
            self._settings_error = (
                "Saved settings could not be read. Fix or move the settings file."
            )
        self.preview = preview
        self.worker: EngineWorker | None = None
        self._quitting = False
        self._telemetry_ready = False
        self._models_ready = False
        self._phase = "stopped"
        self._speaking = False
        self._mode = "engineer"
        self._failed = False
        self.setWindowTitle("Race Engineer" + (" — Design preview" if preview else ""))
        self.setWindowIcon(qta.icon("mdi6.headset", color=_ACCENT))
        self.resize(1040, 700)
        self.setMinimumSize(1000, 690)
        self.setStyleSheet(_STYLE)
        self._build()
        self._tray()
        self._populate_devices(devices)
        self._load_controls()
        self._connect()
        self._render()
        if self._settings_error:
            self.notice.setText(self._settings_error)
            self.start_button.setEnabled(False)
        if preview:
            self._telemetry_ready = self._models_ready = True
            self._phase = "ready"
            self._render()
            self.notice.setText("Design preview — no telemetry, microphone or speech active.")

    def _build(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        main = QHBoxLayout()
        main.setContentsMargins(32, 30, 28, 22)
        main.setSpacing(30)
        outer.addLayout(main, 1)
        left = QVBoxLayout()
        left.setSpacing(18)
        self.eyebrow = label("ENGINEER STOPPED", "eyebrow")
        eyebrow_row = QHBoxLayout()
        eyebrow_row.setSpacing(10)
        self.running_icon = QLabel()
        self.running_icon.setFixedWidth(18)
        eyebrow_row.addWidget(self.running_icon)
        eyebrow_row.addWidget(self.eyebrow, 1)
        left.addLayout(eyebrow_row)
        self.heading = label("Ready to start", "title")
        left.addWidget(self.heading)
        left.addSpacing(2)
        ptt = QHBoxLayout()
        ptt.setSpacing(16)
        self.keycap = label(compact_binding_label(self.settings.ptt_binding), "key")
        self.keycap.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.keycap.setFixedSize(92, 92)
        ptt.addWidget(self.keycap)
        ptt_text = QVBoxLayout()
        ptt_text.setSpacing(9)
        ptt_text.addWidget(label("Hold to talk", "instruction"))
        helper = label("Release to send your question.", "muted")
        helper.setWordWrap(True)
        ptt_text.addWidget(helper)
        ptt.addLayout(ptt_text, 1)
        left.addLayout(ptt)
        left.addSpacing(8)
        left.addWidget(divider())
        left.addSpacing(4)
        self.telemetry = StatusRow("iRacing not connected")
        self.models = StatusRow("Local speech not loaded")
        left.addWidget(self.telemetry)
        left.addWidget(self.models)
        left.addSpacing(8)
        left.addWidget(divider())
        mute = QHBoxLayout()
        mute.addWidget(label("Mute all audio"), 1)
        self.mute_button = QPushButton()
        self.mute_button.setCheckable(True)
        self.mute_button.setProperty("role", "switch")
        self.mute_button.setAccessibleName("Mute all audio")
        self.mute_button.setToolTip("Mutes automatic calls and replies, including critical calls.")
        self.mute_button.setFixedSize(54, 40)
        self.mute_button.setIconSize(QSize(48, 32))
        mute.addWidget(self.mute_button)
        left.addLayout(mute)
        left.addStretch(1)
        self.start_button = QPushButton("Start engineer")
        self.start_button.setMinimumWidth(190)
        left.addWidget(self.start_button, 0, Qt.AlignmentFlag.AlignLeft)
        main.addLayout(left, 4)
        main.addWidget(divider(True))
        right = QVBoxLayout()
        right.setSpacing(22)
        right.addWidget(label("Radio settings", "title"))
        right.addSpacing(10)
        form = QGridLayout()
        form.setHorizontalSpacing(16)
        form.setVerticalSpacing(24)
        form.setColumnStretch(1, 1)
        form.setColumnMinimumWidth(0, 138)
        self.binding = QPushButton(compact_binding_label(self.settings.ptt_binding))
        self.binding.setFixedWidth(66)
        self.binding.setAccessibleName("Change push-to-talk binding")
        self.change_binding = QPushButton("Change")
        key_row = QHBoxLayout()
        key_row.setSpacing(12)
        key_row.addWidget(self.binding)
        key_row.addWidget(self.change_binding)
        key_row.addStretch()
        form.addWidget(label("Push-to-talk"), 0, 0)
        form.addLayout(key_row, 0, 1, 1, 2)
        self.microphone = QComboBox()
        self.output = QComboBox()
        self.test_mic = QPushButton("Test mic")
        self.test_voice = QPushButton("Test voice")
        for row, caption, combo, button in (
            (1, "Microphone", self.microphone, self.test_mic),
            (2, "Output", self.output, self.test_voice),
        ):
            text = label(caption)
            text.setBuddy(combo)
            combo.setAccessibleName(caption)
            combo.setMinimumWidth(180)
            combo.setSizeAdjustPolicy(
                QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
            )
            combo.setMinimumContentsLength(16)
            form.addWidget(text, row, 0)
            form.addWidget(combo, row, 1)
            form.addWidget(button, row, 2)
        self.volume = QSlider(Qt.Orientation.Horizontal)
        self.volume.setFixedHeight(28)
        self.volume.setRange(0, 100)
        self.volume.setAccessibleName("Speech volume")
        self.volume_label = label("80%")
        form.addWidget(label("Volume"), 3, 0)
        form.addWidget(self.volume, 3, 1)
        form.addWidget(self.volume_label, 3, 2)
        self.language = QComboBox()
        self.language.setAccessibleName("Reply language")
        self.language.addItem("Auto · English / Turkish", "auto")
        self.language.addItem("English", "en")
        self.language.addItem("Turkish", "tr")
        form.addWidget(label("Reply language"), 4, 0)
        form.addWidget(self.language, 4, 1, 1, 2)
        right.addLayout(form)
        right.addSpacing(3)
        right.addWidget(divider())
        self.preferences = QPushButton("Engineer preferences")
        self.preferences.setProperty("role", "link")
        right.addWidget(self.preferences, 0, Qt.AlignmentFlag.AlignLeft)
        note = label(
            "Stop the engineer to change radio settings.\n"
            "Output selection is for replies; calls use Windows default.",
            "muted",
        )
        note.setWordWrap(True)
        right.addWidget(note)
        self.notice = label("", "muted")
        self.notice.setWordWrap(True)
        self.notice.setAccessibleName("Session status message")
        right.addWidget(self.notice)
        right.addStretch(1)
        main.addLayout(right, 6)
        outer.addWidget(divider())
        footer = QHBoxLayout()
        footer.setContentsMargins(32, 18, 28, 18)
        footer.addWidget(label("Runs locally on this PC", "muted"), 1)
        self.minimize_button = QPushButton("Minimize to tray")
        self.minimize_button.setProperty("role", "primary")
        footer.addWidget(self.minimize_button)
        outer.addLayout(footer)
        self._settings_controls = [
            self.binding,
            self.change_binding,
            self.microphone,
            self.output,
            self.test_mic,
            self.test_voice,
            self.volume,
            self.language,
            self.preferences,
        ]

    def _tray(self) -> None:
        self.tray = QSystemTrayIcon(self.windowIcon(), self)
        self.tray.setToolTip("Race Engineer")
        menu = QMenu(self)
        show_action = QAction("Show control panel", self)
        show_action.triggered.connect(self._restore)
        menu.addAction(show_action)
        self.tray_mute = QAction("Mute all audio", self)
        self.tray_mute.setCheckable(True)
        self.tray_mute.triggered.connect(self.mute_button.setChecked)
        menu.addAction(self.tray_mute)
        self.tray_stop = QAction("Stop engineer", self)
        self.tray_stop.triggered.connect(self._stop)
        menu.addAction(self.tray_stop)
        menu.addSeparator()
        quit_action = QAction("Quit", self)
        quit_action.triggered.connect(self.close)
        menu.addAction(quit_action)
        self.tray.setContextMenu(menu)
        self.tray.activated.connect(self._tray_activated)
        if QSystemTrayIcon.isSystemTrayAvailable() and not self.preview:
            self.tray.show()

    def _populate_devices(
        self,
        devices: tuple[list[dict[str, object]], list[dict[str, object]]] | None,
    ) -> None:
        try:
            inputs, outputs = (
                devices
                if devices is not None
                else (([], []) if self.preview else (input_devices(), output_devices()))
            )
        except Exception:
            inputs, outputs = [], []
            self.notice.setText("Audio devices could not be listed. Check your audio installation.")
        self._devices = (inputs, outputs)
        for combo, entries in ((self.microphone, inputs), (self.output, outputs)):
            combo.addItem("System default", None)
            for entry in entries:
                combo.addItem(f"{entry['name']} ({entry['host_api']})", entry["index"])

    def _load_controls(self) -> None:
        for combo, index, name in (
            (self.microphone, self.settings.input_device, self.settings.input_name),
            (self.output, self.settings.output_device, self.settings.output_name),
        ):
            found = combo.findData(index) if name is None else combo.findText(name)
            if found < 0:
                combo.addItem("Unavailable device — choose another", -1)
                found = combo.count() - 1
            combo.setCurrentIndex(found)
        self.volume.setValue(self.settings.volume)
        self.volume_label.setText(f"{self.settings.volume}%")
        self.language.setCurrentIndex(self.language.findData(self.settings.reply_language))
        self._show_binding()

    def _show_binding(self) -> None:
        compact = compact_binding_label(self.settings.ptt_binding)
        for widget in (self.binding, self.keycap):
            widget.setText(compact)
            widget.setToolTip(self.settings.ptt_binding.label)

    def _connect(self) -> None:
        self.start_button.clicked.connect(self._start_or_stop)
        self.minimize_button.clicked.connect(self._minimize)
        self.mute_button.toggled.connect(self._mute)
        self.binding.clicked.connect(self._change_binding)
        self.change_binding.clicked.connect(self._change_binding)
        self.preferences.clicked.connect(self._preferences)
        self.test_mic.clicked.connect(lambda: self._start("mic"))
        self.test_voice.clicked.connect(lambda: self._start("voice"))
        self.volume.valueChanged.connect(lambda value: self.volume_label.setText(f"{value}%"))
        self.volume.sliderReleased.connect(self._save)
        self.volume.valueChanged.connect(self._volume_changed)
        for combo in (self.microphone, self.output, self.language):
            combo.currentIndexChanged.connect(self._save)

    def _volume_changed(self, value: int) -> None:
        del value
        if not self.volume.isSliderDown():
            self._save()

    def _read_controls(self) -> PanelSettings:
        raw = self.settings.model_dump()
        raw.update(
            input_device=self.microphone.currentData(),
            output_device=self.output.currentData(),
            input_name=self.microphone.currentText()
            if self.microphone.currentData() is not None
            else None,
            output_name=self.output.currentText()
            if self.output.currentData() is not None
            else None,
            volume=self.volume.value(),
            reply_language=self.language.currentData(),
        )
        return PanelSettings.model_validate(raw)

    def _save(self) -> bool:
        if self.preview:
            return True
        if self._settings_error:
            self.notice.setText(self._settings_error)
            return False
        try:
            candidate = self._read_controls()
            save_settings(self.settings_path, candidate)
            self.settings = candidate
        except (OSError, ValueError):
            self.notice.setText("Settings not saved. Check device selection and file permissions.")
            return False
        return True

    def _change_binding(self) -> None:
        dialog = BindingDialog(
            self.settings.ptt_binding,
            self,
            detector_factory=None if self.preview else JoystickBindingDetector,
        )
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.settings = self.settings.model_copy(
                update={"ptt_binding": dialog.binding}
            )
            self._show_binding()
            self._save()

    def _preferences(self) -> None:
        dialog = QDialog(self)
        dialog.setWindowTitle("Engineer preferences")
        dialog.setMinimumWidth(460)
        layout = QVBoxLayout(dialog)
        layout.setSpacing(18)
        layout.addWidget(label("My defaults", "instruction"))
        position = QCheckBox("Announce position changes")
        position.setChecked(self.settings.announce_position_changes)
        pits = QCheckBox("Announce pit entry and exit")
        pits.setChecked(self.settings.announce_pit_transitions)
        layout.addWidget(position)
        layout.addWidget(pits)
        note = label(
            "Saved only on this PC, applied at the next Start.\n"
            "Flags and critical calls are unchanged.\n"
            "No automatic learning or voice-based settings yet.",
            "muted",
        )
        layout.addWidget(note)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.settings = self.settings.model_copy(
                update={
                    "announce_position_changes": position.isChecked(),
                    "announce_pit_transitions": pits.isChecked(),
                }
            )
            self._save()

    def _start_or_stop(self) -> None:
        if self.worker:
            self._stop()
        else:
            self._start("engineer")

    def _start(self, mode: str) -> None:
        if self.preview:
            self.notice.setText("Design preview only. Launch without --preview for a real session.")
            return
        if self.worker or not self._save():
            return
        if mode == "voice" and self.mute_button.isChecked():
            self.notice.setText("Turn off Mute all audio before testing the voice.")
            return
        # Re-enumerate before opening anything; stale PortAudio indexes must not silently move.
        try:
            for selected, name, entries in (
                (self.settings.input_device, self.settings.input_name, input_devices()),
                (self.settings.output_device, self.settings.output_name, output_devices()),
            ):
                if selected is not None and not any(
                    entry["index"] == selected and f"{entry['name']} ({entry['host_api']})" == name
                    for entry in entries
                ):
                    raise ValueError(
                        "Audio devices changed. Reopen the panel and select your device."
                    )
        except Exception as error:
            self.notice.setText(str(error))
            return
        if mode == "engineer" and self.settings.ptt_binding.kind == "joystick":
            try:
                button = JoystickButtonInput(self.settings.ptt_binding)
                button.close()
            except SpeechInputError as error:
                self.notice.setText(_friendly_error(error))
                return
        self._mode, self._failed = mode, False
        self._phase = "starting"
        self._telemetry_ready = self._models_ready = self._speaking = False
        self.notice.setText("Do not run another live engineer alongside this panel.")
        self.worker = EngineWorker(
            self.root,
            self.config_path,
            self.config,
            self.settings,
            mode,
            self.mute_button.isChecked(),
        )
        self.worker.status.connect(self._status)
        self.worker.failed.connect(self._error)
        self.worker.finished.connect(self._finished)
        self.worker.start()
        self._render()

    def _stop(self) -> None:
        if self.worker:
            self._phase = "stopping"
            self.worker.control.stop.set()
            self._render()

    def _mute(self, muted: bool) -> None:
        self.tray_mute.setChecked(muted)
        if self.worker:
            if muted:
                self.worker.control.muted.set()
                if self._mode == "voice":
                    self._stop()
            else:
                self.worker.control.muted.clear()
        self._render()

    def _status(self, topic: str, value: str) -> None:
        if topic == "telemetry":
            self._telemetry_ready = value == "ready"
        elif topic == "models":
            self._models_ready = value == "ready"
        elif topic == "phase" and self._phase != "stopping":
            self._phase = value
        elif topic == "speech":
            self._speaking = value == "speaking"
        elif topic == "notice":
            self.notice.setText(value)
        self._render()

    def _error(self, message: str) -> None:
        self._failed = True
        self.notice.setText(f"Could not continue: {message}")

    def _finished(self) -> None:
        worker, self.worker = self.worker, None
        if worker:
            worker.deleteLater()
        self._phase = "error" if self._failed else "stopped"
        self._telemetry_ready = self._models_ready = self._speaking = False
        self._render()
        if self._quitting:
            self.close()

    def _render(self) -> None:
        active = self.worker is not None or self.preview
        muted = self.mute_button.isChecked()
        self.mute_button.setIcon(
            qta.icon(
                "mdi6.toggle-switch" if muted else "mdi6.toggle-switch-off-outline",
                color=_ACCENT if muted else _MUTED,
                scale_factor=1.65,
            )
        )
        titles = {
            "stopped": "Ready to start",
            "error": "Needs attention",
            "starting": "Starting…",
            "loading_model": "Loading model…",
            "loading_speech": "Loading speech…",
            "listening": "Listening…",
            "transcribing": "Transcribing…",
            "thinking": "Thinking…",
            "testing_voice": "Testing voice…",
            "testing_mic": "Speak now…",
            "stopping": "Stopping…",
            "ready": "Radio ready" if self._telemetry_ready else "Waiting for iRacing",
        }
        heading = titles.get(self._phase, "Starting…")
        if self._speaking and self._phase != "stopping":
            heading = "Speaking…"
        if muted and self._phase == "ready":
            heading = "Audio muted"
        self.heading.setText(heading)
        self.heading.setStyleSheet(f"font-size: {40 if len(heading) <= 16 else 28}px;")
        self.running_icon.setPixmap(
            qta.icon(
                "mdi6.circle-medium",
                color="#29c66c" if active else _MUTED,
            ).pixmap(18, 18)
        )
        self.eyebrow.setText("ENGINEER RUNNING" if active else "ENGINEER STOPPED")
        if active and self._phase not in ("ready", "listening", "transcribing", "thinking"):
            self.eyebrow.setText("RADIO TEST" if self._mode != "engineer" else "ENGINEER STARTING")
        if self._phase == "stopping":
            self.eyebrow.setText("STOPPING")
        self.telemetry.update_status(
            "iRacing connected" if self._telemetry_ready else "iRacing not connected",
            self._telemetry_ready,
        )
        self.models.update_status(
            "Local speech ready" if self._models_ready else "Local speech not ready",
            self._models_ready,
        )
        self.start_button.setText(
            ("Stop engineer" if self._mode == "engineer" else "Cancel test")
            if active
            else "Start engineer"
        )
        self.start_button.setEnabled(self._phase != "stopping" and not self._settings_error)
        self.tray_stop.setEnabled(self.worker is not None)
        self.tray.setToolTip(f"Race Engineer — {heading}")
        for control in self._settings_controls:
            control.setEnabled(not active or self.preview)

    def _minimize(self) -> None:
        if self.tray.isVisible() and QSystemTrayIcon.isSystemTrayAvailable():
            self.hide()
        else:
            self.showMinimized()

    def _restore(self) -> None:
        self.showNormal()
        self.raise_()
        self.activateWindow()

    def _tray_activated(self, reason: QSystemTrayIcon.ActivationReason) -> None:
        if reason in (
            QSystemTrayIcon.ActivationReason.Trigger,
            QSystemTrayIcon.ActivationReason.DoubleClick,
        ):
            self._restore()

    def closeEvent(self, event: QCloseEvent) -> None:
        if self.worker:
            event.ignore()
            if (
                not self._quitting
                and QMessageBox.question(
                    self,
                    "Stop and quit?",
                    "Stop the engineer and close the control panel?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.No,
                )
                == QMessageBox.StandardButton.Yes
            ):
                self._quitting = True
                self._stop()
        else:
            self.tray.hide()
            event.accept()


def main() -> int:
    parser = argparse.ArgumentParser(description="Race Engineer desktop control panel")
    parser.add_argument("--config", type=Path, default=Path("config/default.toml"))
    parser.add_argument("--settings", type=Path, default=Path("data/control-panel.json"))
    parser.add_argument("--preview", action="store_true", help="design only; never opens hardware")
    args = parser.parse_args()
    # A pythonw launch has no console. Do not persist printed conversation text.
    if sys.stdout is None:
        sys.stdout = open(os.devnull, "w", encoding="utf-8")  # noqa: SIM115
    if sys.stderr is None:
        sys.stderr = open(os.devnull, "w", encoding="utf-8")  # noqa: SIM115
    app = QApplication.instance() or QApplication(sys.argv[:1])
    assert isinstance(app, QApplication)
    app.setApplicationName("Race Engineer")
    app.setStyle("Fusion")
    app.setFont(QFont("Segoe UI", 11))
    root = Path.cwd()
    lock: QLockFile | None = None
    if not args.preview:
        (root / "data").mkdir(exist_ok=True)
        lock = QLockFile(str(root / "data/control-panel.lock"))
        if not lock.tryLock(0):
            QMessageBox.information(None, "Race Engineer", "The control panel is already running.")
            return 1
        (root / "logs").mkdir(exist_ok=True)
        handler = RotatingFileHandler(
            root / "logs/control-panel.jsonl", maxBytes=2_000_000, backupCount=2, encoding="utf-8"
        )
        handler.setFormatter(JsonFormatter())
        logging.getLogger().addHandler(handler)
        logging.getLogger().setLevel(logging.INFO)
    try:
        panel = RadioDesk(
            root, args.config.resolve(), args.settings.resolve(), preview=args.preview
        )
    except (OSError, ValueError) as error:
        QMessageBox.critical(None, "Cannot open Race Engineer", str(error))
        return 1
    panel.show()
    result = app.exec()
    if lock:
        lock.unlock()
    return result


if __name__ == "__main__":
    raise SystemExit(main())
