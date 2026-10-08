# -*- mode: python ; coding: utf-8 -*-

import os
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files

ROOT = Path.cwd().resolve()
SRC = ROOT / "src"
ICON = SRC / "race_engineer/assets/pitward-brand-pack/icons/pitward.ico"
VERSION_FILE = os.environ["PITWARD_VERSION_FILE"]

datas = [
    (str(ROOT / "config/default.toml"), "config"),
    (
        str(SRC / "race_engineer/assets/pitward-brand-pack/icons/pitward.ico"),
        "race_engineer/assets/pitward-brand-pack/icons",
    ),
    (
        str(SRC / "race_engineer/assets/pitward-brand-pack/png/pitward-logo-horizontal-dark.png"),
        "race_engineer/assets/pitward-brand-pack/png",
    ),
    (str(SRC / "race_engineer/stt/assets/radio-open.wav"), "race_engineer/stt/assets"),
    (str(SRC / "race_engineer/stt/assets/radio-close.wav"), "race_engineer/stt/assets"),
]
datas += collect_data_files("qtawesome")

analysis = Analysis(
    [str(ROOT / "packaging/pitward_entry.py")],
    pathex=[str(SRC)],
    binaries=[],
    datas=datas,
    hiddenimports=["irsdk", "pygame", "sounddevice"],
    hookspath=[],
    runtime_hooks=[],
    excludes=["torch", "transformers", "piper", "onnxruntime", "librosa", "scipy"],
    noarchive=False,
)
pyz = PYZ(analysis.pure)
exe = EXE(
    pyz,
    analysis.scripts,
    [],
    exclude_binaries=True,
    name="Pitward",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    icon=str(ICON),
    version=VERSION_FILE,
)
bundle = COLLECT(
    exe,
    analysis.binaries,
    analysis.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="Pitward",
)
