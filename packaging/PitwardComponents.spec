# -*- mode: python ; coding: utf-8 -*-

import os
from pathlib import Path

ROOT = Path.cwd().resolve()
SRC = ROOT / "src"
ICON = SRC / "race_engineer/assets/pitward-brand-pack/icons/pitward.ico"
VERSION_FILE = os.environ["PITWARD_VERSION_FILE"]

analysis = Analysis(
    [str(ROOT / "packaging/components_entry.py")],
    pathex=[str(SRC)],
    binaries=[],
    datas=[
        (
            str(SRC / "race_engineer/assets/components.v1.json"),
            "race_engineer/assets",
        )
    ],
    hiddenimports=[],
    hookspath=[],
    runtime_hooks=[],
    excludes=["PySide6", "qtawesome", "torch", "transformers", "piper", "onnxruntime"],
    noarchive=False,
)
pyz = PYZ(analysis.pure)
exe = EXE(
    pyz,
    analysis.scripts,
    analysis.binaries,
    analysis.datas,
    [],
    name="PitwardComponents",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    icon=str(ICON),
    version=VERSION_FILE,
)
