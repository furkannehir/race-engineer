"""Fail public release publication while recorded distribution reviews are incomplete."""

import json
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    path = root / "distribution/release-clearance.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    required = ("radio_cue_source_cleared", "turkish_voice_distribution_reviewed")
    missing = [name for name in required if raw.get(name) is not True]
    if missing:
        print("Public release blocked: " + ", ".join(missing))
        return 1
    print("Distribution clearance gates passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
