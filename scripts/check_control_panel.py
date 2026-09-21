"""Render the real Qt widgets without microphone, model, simulator, or playback access."""

import os
from pathlib import Path

os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ["QT_SCALE_FACTOR"] = "1"

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QFontDatabase, QImage, QPainter
from PySide6.QtWidgets import QApplication

from race_engineer.ui.app import BindingDialog, RadioDesk

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs/design/control-panel"


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    app = QApplication([])
    app.setStyle("Fusion")
    # The offscreen platform lacks Windows' font discovery; load the same installed
    # Segoe UI family the interactive Windows platform uses (no copying/installing).
    fonts = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts"
    for name in ("segoeui.ttf", "segoeuib.ttf", "seguisb.ttf"):
        if QFontDatabase.addApplicationFont(str(fonts / name)) < 0:
            raise RuntimeError(f"Could not load installed font: {name}")
    panel = RadioDesk(
        ROOT,
        ROOT / "config/default.toml",
        ROOT / "data/unused-preview.json",
        preview=True,
    )
    panel.show()
    app.processEvents()
    panel.grab().save(str(OUTPUT / "implementation.png"))
    source = QImage(str(OUTPUT / "selected-radio-desk.png"))
    if source.isNull():
        raise RuntimeError("Selected mockup missing")
    # Native OS chrome is intentionally excluded; normalize density, not app geometry.
    content = source.copy(2, 54, source.width() - 4, source.height() - 56)
    reference = content.scaledToWidth(panel.width(), Qt.TransformationMode.SmoothTransformation)
    actual = panel.grab().toImage()
    comparison = QImage(
        panel.width() * 2, max(reference.height(), actual.height()), QImage.Format.Format_RGB32
    )
    comparison.fill(Qt.GlobalColor.black)
    painter = QPainter(comparison)
    painter.drawImage(0, 0, reference)
    painter.drawImage(panel.width(), 0, actual)
    painter.end()
    comparison.save(str(OUTPUT / "comparison.png"))
    panel.preview = False
    panel._phase = "stopped"
    panel._telemetry_ready = panel._models_ready = False
    panel.notice.setText("")
    panel._render()
    app.processEvents()
    panel.grab().save(str(OUTPUT / "stopped.png"))
    panel.resize(1000, 690)
    panel.preview = True
    panel._telemetry_ready = panel._models_ready = True
    panel._status("phase", "thinking")
    app.processEvents()
    panel.grab().save(str(OUTPUT / "minimum-size.png"))
    panel._status("phase", "ready")
    panel.mute_button.setChecked(True)
    app.processEvents()
    panel.grab().save(str(OUTPUT / "muted.png"))

    binding = BindingDialog(panel.settings.ptt_binding, panel, detector_factory=None)
    binding.show()
    app.processEvents()
    binding.grab().save(str(OUTPUT / "ptt-binding.png"))
    binding.close()

    def capture_dialog() -> None:
        dialog = app.activeModalWidget()
        assert dialog is not None
        dialog.grab().save(str(OUTPUT / "preferences.png"))
        dialog.close()

    QTimer.singleShot(0, capture_dialog)
    panel._preferences()
    panel.close()
    print(f"Source: {source.width()}x{source.height()}; app: {actual.width()}x{actual.height()}")
    print(f"Native Qt screenshots saved to {OUTPUT}")


if __name__ == "__main__":
    main()
