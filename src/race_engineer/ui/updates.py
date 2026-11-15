"""Background update I/O using the panel's existing Qt worker convention."""

import logging
from pathlib import Path

from PySide6.QtCore import QThread, Signal

from race_engineer.updates import (
    AvailableUpdate,
    UpdateCancelled,
    UpdateError,
    check_for_update,
    download_update,
)
from race_engineer.version import __version__


class UpdateWorker(QThread):
    completed = Signal(object)
    failed = Signal(str)
    progress = Signal(int)

    def __init__(
        self, release: AvailableUpdate | None = None, directory: Path | None = None
    ) -> None:
        super().__init__()
        self.release = release
        self.directory = directory
        self.error: str | None = None
        self.cancelled = False

    def run(self) -> None:
        try:
            result: AvailableUpdate | Path | None
            if self.release is None:
                result = check_for_update(__version__, cancelled=self.isInterruptionRequested)
            else:
                assert self.directory is not None
                result = download_update(
                    self.release,
                    self.directory,
                    cancelled=self.isInterruptionRequested,
                    progress=self.progress.emit,
                )
            if not self.cancelled and not self.isInterruptionRequested():
                self.completed.emit(result)
        except UpdateCancelled:
            self.cancelled = True
        except Exception as error:
            self.error = str(error) if isinstance(error, UpdateError) else "update_unavailable"
            logging.getLogger(__name__).warning(
                "update operation failed",
                extra={"event": "update_failed", "reason": self.error},
            )
            self.failed.emit(self.error)
