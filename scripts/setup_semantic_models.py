"""Explicit download step only. Normal semantic workers are strictly offline.

Run with an interpreter containing huggingface_hub. The optional Laya package can
be installed into data/dialogue-prototype/packages without changing that interpreter.
"""

import argparse
import json
from pathlib import Path

MODELS = {
    "minilm": (
        "MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli",
        "0a71e92a985b6e1ad1828cf67ce9c459639c1dca",
        ["config.json", "model.safetensors", "*token*.json", "special_tokens_map.json"],
    ),
    "laya": (
        "convaiinnovations/laya-multilingual",
        "e4e9ddf21a7b1903b7acffd8814ad4307bf63a67",
        ["rl_agent_config.json", "model.safetensors", "encoder/*", "tokenizer/*"],
    ),
}


def main() -> None:
    from huggingface_hub import snapshot_download

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", choices=[*MODELS, "all"], default="all")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1] / "data/dialogue-prototype"
    for name, (repo, revision, patterns) in MODELS.items():
        if args.candidate not in {name, "all"}:
            continue
        destination = root / name
        snapshot_download(
            repo,
            revision=revision,
            local_dir=destination,
            allow_patterns=patterns,
            token=False,
            max_workers=2,
        )
        (destination / "provenance.json").write_text(
            json.dumps({"repo": repo, "revision": revision}, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"ready: {name} at pinned revision {revision}", flush=True)


if __name__ == "__main__":
    main()
