"""Download the pinned public mDeBERTa NLI checkpoint for the isolated CPU demo."""

import json
import os
from pathlib import Path

from race_engineer.evaluation.mdeberta_interpreter import MODEL_FILES, MODEL_REPO, MODEL_REVISION

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    destination = (ROOT / "data/dialogue-prototype/mdeberta").resolve()
    if not destination.is_relative_to(ROOT / "data"):
        raise ValueError("model destination must stay inside repository data")
    provenance = destination / "provenance.json"
    expected = {"repo": MODEL_REPO, "revision": MODEL_REVISION}
    if provenance.exists() and json.loads(provenance.read_text(encoding="utf-8")) != expected:
        raise ValueError("existing model provenance differs; not overwriting it")
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "0"
    os.environ["TRANSFORMERS_OFFLINE"] = "0"
    from huggingface_hub import snapshot_download  # type: ignore[import-not-found]

    print(f"Downloading {MODEL_REPO}@{MODEL_REVISION} into {destination}", flush=True)
    print("About 1.1 GB of weights plus tokenizer files. No Qwen or live settings changed.",
          flush=True)
    snapshot_download(repo_id=MODEL_REPO, revision=MODEL_REVISION,
                      local_dir=str(destination), allow_patterns=list(MODEL_FILES))
    if any(not (destination / name).is_file() for name in MODEL_FILES):
        raise ValueError("checkpoint download is incomplete; rerun setup to resume")
    provenance.write_text(json.dumps(expected, indent=2) + "\n", encoding="utf-8")
    print("Model ready. The terminal demo now works offline.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
