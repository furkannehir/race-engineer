# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path

from PyInstaller.utils.hooks import collect_all

ROOT = Path.cwd().resolve()
SRC = ROOT / "src"

piper_data, piper_binaries, piper_hidden = collect_all("piper")

analysis = Analysis(
    [str(ROOT / "packaging/tts_worker_entry.py")],
    pathex=[str(SRC)],
    binaries=piper_binaries,
    datas=piper_data,
    hiddenimports=piper_hidden + ["onnxruntime", "sounddevice", "piper.config"],
    hookspath=[],
    runtime_hooks=[],
    excludes=["PySide6", "qtawesome", "pygame", "torch", "transformers"],
    noarchive=False,
)
pyz = PYZ(analysis.pure)
exe = EXE(
    pyz,
    analysis.scripts,
    [],
    exclude_binaries=True,
    name="PitwardTTSWorker",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
)
bundle = COLLECT(
    exe,
    analysis.binaries,
    analysis.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="PitwardTTSWorker",
)
