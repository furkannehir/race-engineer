"""Explicit online setup; speech recognition itself never downloads anything."""

import hashlib
import subprocess
import sys
import urllib.request
import venv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "data/stt-prototype"
REVISION = "7f1569a48a89f3e3f4dc3a5c9d28bddd903bc76c"
MODEL_URL = f"https://huggingface.co/Qwen/Qwen3-ASR-0.6B-hf/resolve/{REVISION}"
FILES = {
    "config.json": None,
    "generation_config.json": None,
    "processor_config.json": None,
    "tokenizer_config.json": None,
    "chat_template.jinja": None,
    "tokenizer.json": "fe1fad59be22a41ee293363fcf95fdedbc7c93f3b49270b1d2e18bd1399a7a05",
    "model.safetensors": "d3f212dd20abecd315d830bc54ae3865e56ebfc3276484e57b771288ba27fd35",
}


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    environment = ASSETS / "runtime"
    python = environment / "Scripts/python.exe"
    if not python.is_file():
        print("Creating isolated ASR environment...", flush=True)
        venv.create(environment, with_pip=True)
    subprocess.run(
        [
            str(python),
            "-m",
            "pip",
            "install",
            "torch==2.14.0",
            "--index-url",
            "https://download.pytorch.org/whl/cpu",
        ],
        check=True,
    )
    subprocess.run(
        [str(python), "-m", "pip", "install", "-e", f"{ROOT}[asr]"],
        check=True,
    )
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-e", f"{ROOT}[audio]"],
        check=True,
    )
    model_dir = ASSETS / "Qwen3-ASR-0.6B-hf"
    model_dir.mkdir(exist_ok=True)
    for name, expected in FILES.items():
        destination = model_dir / name
        if destination.exists():
            if expected is not None and digest(destination) != expected:
                raise RuntimeError(f"Existing {name} failed checksum; refusing to overwrite it")
            print(f"Keeping {name}", flush=True)
            continue
        print(f"Downloading {name} from pinned official Qwen revision...", flush=True)
        temporary = model_dir / f"{name}.partial"
        # Exclusive creation protects a previous partial download from being overwritten.
        with (
            urllib.request.urlopen(f"{MODEL_URL}/{name}", timeout=60) as response,
            temporary.open("xb") as stream,
        ):
            while chunk := response.read(1024 * 1024):
                stream.write(chunk)
        if expected is not None and digest(temporary) != expected:
            raise RuntimeError(
                f"Downloaded {name} failed checksum; retained .partial for inspection"
            )
        temporary.rename(destination)
    print("Local STT setup complete. Model weights and tokenizer checksums verified.", flush=True)


if __name__ == "__main__":
    main()
