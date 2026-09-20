"""Explicit online setup of the isolated Piper runtime and pinned bilingual voices."""

import hashlib
import subprocess
import sys
import urllib.request
import venv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "data/tts-prototype"
REVISION = "c10ece1aade47bb51c153c893d14e5bf8e5b7117"
BASE_URL = f"https://huggingface.co/rhasspy/piper-voices/resolve/{REVISION}"
VOICES = (
    (
        "en/en_US/ljspeech/high",
        "en_US-ljspeech-high",
        "5d4f08ba6a2a48c44592eed3ce56bf85e9de3dd4e20df90541ae68a8310c029a",
    ),
    (
        "tr/tr_TR/dfki/medium",
        "tr_TR-dfki-medium",
        "2844717f524ab965d3fe86e60562cbb601d3e456836efcc2196cc3a14112a8fb",
    ),
)


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def download(url: str, destination: Path, expected: str | None = None) -> None:
    if destination.exists():
        if expected is not None and digest(destination) != expected:
            raise RuntimeError(f"Existing {destination.name} failed checksum; not overwriting")
        print(f"Keeping {destination.name}", flush=True)
        return
    print(f"Downloading {destination.name}...", flush=True)
    temporary = destination.with_name(destination.name + ".partial")
    with urllib.request.urlopen(url, timeout=60) as response, temporary.open("xb") as stream:
        while chunk := response.read(1024 * 1024):
            stream.write(chunk)
    if expected is not None and digest(temporary) != expected:
        raise RuntimeError(f"Checksum failed for {destination.name}; retained .partial")
    temporary.rename(destination)


def main() -> None:
    print("Piper is GPL-3.0; Turkish dfki's model card lists CC BY-NC-SA 4.0 dataset terms.")
    print("These are prototype voices; see docs/conversational-speech.md before distribution.")
    ASSETS.mkdir(parents=True, exist_ok=True)
    environment = ASSETS / "runtime"
    python = environment / "Scripts/python.exe"
    if not python.is_file():
        print("Creating isolated Piper environment...", flush=True)
        venv.create(environment, with_pip=True)
    subprocess.run([str(python), "-m", "pip", "install", "-e", f"{ROOT}[radio]"], check=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-e", f"{ROOT}[audio]"], check=True)
    voice_dir = ASSETS / "voices"
    voice_dir.mkdir(exist_ok=True)
    for folder, voice, checksum in VOICES:
        for suffix, expected in ((".onnx", checksum), (".onnx.json", None)):
            name = voice + suffix
            download(f"{BASE_URL}/{folder}/{name}", voice_dir / name, expected)
        download(f"{BASE_URL}/{folder}/MODEL_CARD", voice_dir / f"{voice}.MODEL_CARD")
    print("Local conversational speech ready. Voice model checksums verified.", flush=True)


if __name__ == "__main__":
    main()
