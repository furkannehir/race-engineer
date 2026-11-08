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

# V3 is an explicit source-runtime trial; the packaged/default component set stays V2.
V3_SOURCE_REVISION = "ab55923c27757c90cfcc407baa57dc25a7d25217"
V3_SOURCE_URL = f"https://codeload.github.com/resemble-ai/chatterbox/zip/{V3_SOURCE_REVISION}"
V3_CHECKPOINT = "t3_mtl23ls_v3.safetensors"
V3_CHECKPOINT_SHA256 = "5abca8321ede76f8e61f1cc0d19aea6c946b28871017ce8726f8a69203f05953"
V3_CHECKPOINT_SIZE = 2143989928


def model_files(variant: str = "v2") -> tuple[str, ...]:
    if variant not in {"v2", "v3"}:
        raise ValueError("unsupported Chatterbox model variant")
    return tuple(
        V3_CHECKPOINT if variant == "v3" and name == "t3_mtl23ls_v2.safetensors" else name
        for name in MODEL_FILES
    )
