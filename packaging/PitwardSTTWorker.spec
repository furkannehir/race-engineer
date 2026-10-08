# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files

ROOT = Path.cwd().resolve()
SRC = ROOT / "src"

analysis = Analysis(
    [str(ROOT / "packaging/stt_worker_entry.py")],
    pathex=[str(SRC)],
    binaries=[],
    datas=collect_data_files("transformers"),
    hiddenimports=[
        "numpy",
        "safetensors",
        "scipy.signal",
        "tokenizers",
        "torch",
        "transformers",
        "transformers.models.qwen3.configuration_qwen3",
        "transformers.models.qwen3.modeling_qwen3",
        "transformers.models.qwen3_asr.configuration_qwen3_asr",
        "transformers.models.qwen3_asr.feature_extraction_qwen3_asr",
        "transformers.models.qwen3_asr.modeling_qwen3_asr",
        "transformers.models.qwen3_asr.processing_qwen3_asr",
    ],
    hookspath=[],
    runtime_hooks=[],
    excludes=["PySide6", "qtawesome", "pygame", "piper", "onnxruntime"],
    noarchive=False,
)
pyz = PYZ(analysis.pure)
exe = EXE(
    pyz,
    analysis.scripts,
    [],
    exclude_binaries=True,
    name="PitwardSTTWorker",
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
    name="PitwardSTTWorker",
)
