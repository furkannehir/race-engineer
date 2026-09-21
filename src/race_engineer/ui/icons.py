"""Material Design library icons loaded privately, without installing Windows fonts."""

from pathlib import Path
from typing import Any, cast

import qtawesome  # type: ignore[import-untyped]
from PySide6.QtGui import QIcon
from qtawesome.iconic_font import IconicFont  # type: ignore[import-untyped]

_font: Any = None


def icon(name: str, *, color: str, scale_factor: float = 1.0) -> QIcon:
    global _font
    if _font is None:
        fonts = Path(qtawesome.__file__).parent / "fonts"
        _font = IconicFont(
            (
                "mdi6",
                "materialdesignicons6-webfont-6.9.96.ttf",
                "materialdesignicons6-webfont-charmap-6.9.96.json",
                str(fonts),
            )
        )
    return cast(QIcon, _font.icon(name, color=color, scale_factor=scale_factor))
