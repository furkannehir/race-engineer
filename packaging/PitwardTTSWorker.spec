# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path

from PyInstaller.utils.hooks import collect_all, collect_data_files, copy_metadata

ROOT = Path.cwd().resolve()
SRC = ROOT / "src"

speech_data, speech_binaries, speech_hidden = collect_all("chatterbox")
speech_data += collect_data_files("transformers") + copy_metadata("chatterbox-tts")
speech_data += copy_metadata("transformers") + copy_metadata("torch")
perth_data, perth_binaries, perth_hidden = collect_all("perth")

analysis = Analysis(
    [str(ROOT / "packaging/tts_worker_entry.py")],
    pathex=[str(SRC)],
    binaries=speech_binaries + perth_binaries,
    datas=speech_data + perth_data,
    hiddenimports=speech_hidden + perth_hidden + [
        "torch", "torchaudio", "sounddevice", "librosa", "scipy.signal", "safetensors.torch",
        "transformers", "chatterbox.mtl_tts", "s3tokenizer", "pykakasi", "spacy_pkuseg",
    ],
    hookspath=[],
    runtime_hooks=[],
    excludes=["PySide6", "qtawesome", "pygame"],
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
