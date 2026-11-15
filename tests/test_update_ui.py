from pathlib import Path

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("qtawesome")

from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import QMessageBox
from test_control_panel import app as app
from test_control_panel import panel as panel
from test_updates import available

from race_engineer.installation import Installation
from race_engineer.ui import app as panel_app


class UpdateJob(QObject):
    completed = Signal(object)
    failed = Signal(str)
    progress = Signal(int)
    finished = Signal()

    def __init__(self, release=None, directory=None):
        super().__init__()
        self.release, self.directory = release, directory
        self.cancelled, self.error = False, None
        self.started = False

    def start(self):
        self.started = True

    def requestInterruption(self):
        self.cancelled = True

    def isInterruptionRequested(self):
        return self.cancelled


@pytest.fixture
def update_panel(panel, tmp_path, monkeypatch):
    # The shared fixture opens no hardware. Simulated update workers open no network.
    layout = Installation.discover(
        environ={"PITWARD_DATA_DIR": str(tmp_path / "data")},
        executable=tmp_path / "program/Pitward.exe",
        frozen=True,
    )
    layout.install_root.mkdir(parents=True)
    (layout.install_root / "unins000.exe").touch()
    panel.installation = layout
    panel._missing_components = ()
    monkeypatch.setattr(panel_app, "UpdateWorker", UpdateJob)
    yield panel
    if panel._update_worker is not None:
        panel._update_worker.requestInterruption()
        panel._update_worker.finished.emit()


def mock_offer(monkeypatch, *, accept=False):
    shown = []
    monkeypatch.setattr(QMessageBox, "exec", lambda box: shown.append(box.text()))
    monkeypatch.setattr(
        QMessageBox,
        "clickedButton",
        lambda box: next(
            button
            for button in box.buttons()
            if button.text() == ("Update now" if accept else "Later")
        ),
    )
    return shown


def test_startup_check_can_be_disabled_and_is_not_duplicated(update_panel):
    update_panel.settings = update_panel.settings.model_copy(
        update={"check_updates_on_startup": False}
    )
    update_panel._automatic_update_check()
    assert update_panel._update_worker is None
    update_panel.settings = update_panel.settings.model_copy(
        update={"check_updates_on_startup": True}
    )
    update_panel._automatic_update_check()
    first = update_panel._update_worker
    update_panel._automatic_update_check()
    assert first is not None and first.started and update_panel._update_worker is first


def test_update_prompt_waits_until_the_engineer_is_stopped(update_panel, monkeypatch):
    shown = mock_offer(monkeypatch)
    update_panel._check_updates(manual=False)
    job = update_panel._update_worker
    update_panel.worker = object()
    job.completed.emit(available())
    job.finished.emit()
    assert shown == [] and update_panel._pending_update is not None
    update_panel.worker = None
    update_panel._maybe_offer_update()
    assert len(shown) == 1 and update_panel._update_worker is None
    assert update_panel.installer_requested is None and update_panel.isVisible()


def test_hidden_panel_defers_popup_until_restored(update_panel, monkeypatch):
    shown = mock_offer(monkeypatch)
    update_panel.hide()
    update_panel._pending_update = available()
    update_panel._maybe_offer_update()
    assert not shown
    update_panel.showMinimized()
    update_panel._maybe_offer_update()
    assert not shown
    update_panel._restore()
    assert len(shown) == 1


def test_offline_startup_is_quiet_and_manual_check_reports_the_failure(update_panel):
    update_panel.notice.setText("Ready")
    update_panel._check_updates(manual=False)
    job = update_panel._update_worker
    job.failed.emit("update_check_unavailable")
    job.finished.emit()
    assert update_panel.notice.text() == "Ready"
    update_panel._check_updates(manual=True)
    job = update_panel._update_worker
    job.failed.emit("update_check_unavailable")
    job.finished.emit()
    assert "Update unavailable" in update_panel.notice.text()
    assert not update_panel.restart_requested and update_panel.installer_requested is None


def test_acceptance_downloads_then_requests_the_installer_only_after_success(
    update_panel, monkeypatch, tmp_path
):
    shown = mock_offer(monkeypatch, accept=True)
    update_panel._pending_update = available()
    update_panel._maybe_offer_update()
    job = update_panel._update_worker
    assert shown and job.release == available() and job.started
    assert not update_panel.start_button.isEnabled()
    assert update_panel.installer_requested is None
    ready = []
    update_panel.installer_ready.connect(lambda: ready.append(True))
    installer = tmp_path / available().filename
    installer.touch()
    job.progress.emit(50)
    assert update_panel._update_progress.value() == 50
    job.completed.emit(installer)
    assert update_panel.installer_requested is None
    job.finished.emit()
    assert update_panel.installer_requested == installer and ready == [True]
    assert not update_panel.isVisible() and update_panel.settings_path.is_file()


def test_cancelled_or_failed_download_never_closes_the_panel(update_panel):
    update_panel._download_update(available())
    job = update_panel._update_worker
    update_panel._cancel_update()
    job.completed.emit(Path("must-not-install.exe"))
    job.finished.emit()
    assert update_panel.installer_requested is None and update_panel.isVisible()
    assert "cancelled" in update_panel.notice.text()
    update_panel._download_update(available())
    job = update_panel._update_worker
    job.failed.emit("update_checksum_mismatch")
    job.finished.emit()
    assert update_panel.installer_requested is None and update_panel.isVisible()
    assert "checksum" in update_panel.notice.text()


def test_quitting_waits_for_the_background_update_worker_to_stop(update_panel):
    update_panel._check_updates(manual=False)
    job = update_panel._update_worker
    update_panel.close()
    assert job.cancelled and update_panel.isVisible()
    job.finished.emit()
    assert not update_panel.isVisible() and update_panel._update_worker is None


def test_source_preview_and_first_model_setup_do_not_make_automatic_requests(update_panel):
    update_panel.preview = True
    update_panel._check_updates(manual=False)
    update_panel.preview = False
    update_panel._missing_components = (Path("missing.gguf"),)
    update_panel._automatic_update_check()
    assert update_panel._update_worker is None
