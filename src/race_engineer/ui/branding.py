"""Pitward product identity used by the native control panel."""

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QIcon, QPixmap

PRODUCT_NAME = "Pitward"

_BRAND_ROOT = (
    Path(__file__).resolve().parents[1] / "assets" / "pitward-brand-pack"
)
_WINDOW_ICON = _BRAND_ROOT / "icons" / "pitward.ico"
_PANEL_LOGO = _BRAND_ROOT / "png" / "pitward-logo-horizontal-dark.png"


def application_icon() -> QIcon:
    """Return the multi-resolution Pitward icon for Windows and the system tray."""
    return QIcon(str(_WINDOW_ICON))


def panel_logo(width: int) -> QPixmap:
    """Return the reversed Pitward logo sized for the panel's dark surface."""
    logo = QPixmap(str(_PANEL_LOGO))
    if logo.isNull():
        return logo
    return logo.scaledToWidth(width, Qt.TransformationMode.SmoothTransformation)
