"""Download pinned text-only comparison weights; never change the live configuration."""

import argparse
from dataclasses import dataclass
from pathlib import Path

if __package__:
    from .setup_qwen_gpu import download
else:
    from setup_qwen_gpu import download

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "data/dialogue-comparison"


@dataclass(frozen=True)
class ModelAsset:
    repo: str
    revision: str
    name: str
    size: int
    sha256: str

    @property
    def url(self):
        return f"https://huggingface.co/{self.repo}/resolve/{self.revision}/{self.name}"


# Public Hub blob metadata checked 2026-11-15. No vision projectors are needed.
MODELS = {
    "qwen35": ModelAsset(
        "unsloth/Qwen3.5-2B-GGUF", "f6d5376be1edb4d416d56da11e5397a961aca8ae",
        "Qwen3.5-2B-Q4_K_M.gguf", 1280835840,
        "aaf42c8b7c3cab2bf3d69c355048d4a0ee9973d48f16c731c0520ee914699223",
    ),
    "ministral": ModelAsset(
        "mistralai/Ministral-3-3B-Instruct-2512-GGUF",
        "eb599d408350ea2bb60452cb86be7c7b2fc28227",
        "Ministral-3-3B-Instruct-2512-Q4_K_M.gguf", 2147023008,
        "9ed150d4367e68df0ac8e1540f6ddc65b42d0ee26378329d1ecbca60f93fc5f8",
    ),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", action="append", choices=tuple(MODELS))
    args = parser.parse_args()
    for name in args.model or MODELS:
        download(MODELS[name], ASSETS)
    print("Comparison weights verified. Live configuration unchanged.", flush=True)


if __name__ == "__main__":
    main()
