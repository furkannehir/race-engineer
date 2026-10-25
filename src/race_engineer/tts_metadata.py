"""Pinned speech assets shared by setup, packaging and the offline worker."""

from pathlib import Path

CHATTERBOX_VERSION = "0.1.7"
MODEL_REVISION = "5bb1f6ee58e50c3b8d408bc82a6d3740c2db6e18"
MODEL_RELATIVE = Path("tts/chatterbox") / MODEL_REVISION
MODEL_FILES = (
    "ve.pt",
    "t3_mtl23ls_v2.safetensors",
    "s3gen.pt",
    "grapheme_mtl_merged_expanded_v1.json",
    "conds.pt",
)
