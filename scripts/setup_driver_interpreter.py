"""Download a pinned sentence encoder for the isolated local training pilot."""

import json
import os
from pathlib import Path

from race_engineer.evaluation.driver_training import MODEL_FILES, MODEL_REPO, MODEL_REVISION

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    path = (ROOT / "data/driver-interpreter-pilot/base").resolve()
    if not path.is_relative_to(ROOT / "data"):
        raise ValueError("pilot checkpoint must stay in repository data")
    provenance = {"repo": MODEL_REPO, "revision": MODEL_REVISION}
    if (path / "provenance.json").exists() and json.loads(
        (path / "provenance.json").read_text(encoding="utf-8")
    ) != provenance:
        raise ValueError("refusing to overwrite a different checkpoint")
    os.environ["HF_HUB_OFFLINE"] = "0"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    from huggingface_hub import snapshot_download  # type: ignore[import-not-found]

    print("Downloading pinned multilingual MiniLM sentence encoder (~470 MB)...", flush=True)
    snapshot_download(
        repo_id=MODEL_REPO,
        revision=MODEL_REVISION,
        local_dir=str(path),
        allow_patterns=list(MODEL_FILES),
    )
    if any(not (path / filename).is_file() for filename in MODEL_FILES):
        raise ValueError("incomplete model download")
    (path / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print("Checkpoint ready; training and inference will use local files only.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
