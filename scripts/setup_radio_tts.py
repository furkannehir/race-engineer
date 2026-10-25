"""Provision an isolated Windows Chatterbox runtime and checksum-pinned local models."""

import argparse
import shutil
import subprocess
import sys
import venv
from pathlib import Path

from race_engineer.components import install, load_manifest

ROOT = Path(__file__).resolve().parents[1]


def automatic_backend() -> str:
    executable = shutil.which("nvidia-smi")
    if executable:
        result = subprocess.run(
            [executable, "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            timeout=10,
            check=False,
        )
        if result.returncode == 0 and result.stdout.strip():
            return "cuda"
    return "cpu"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--runtime-dir", type=Path, default=ROOT / "data/tts-chatterbox/runtime")
    parser.add_argument(
        "--runtime-only", action="store_true", help="skip model downloads (e.g. CI)"
    )
    args = parser.parse_args()
    if sys.version_info[:2] != (3, 12):
        parser.error("Use Python 3.12 for the pinned Windows TTS runtime: py -3.12 ...")
    runtime = args.runtime_dir.resolve()
    if not runtime.is_relative_to(ROOT):
        parser.error("runtime-dir must stay inside this repository")
    python = runtime / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    if not python.is_file():
        venv.create(runtime, with_pip=True)
    backend = automatic_backend() if args.backend == "auto" else args.backend
    # CPU matches upstream metadata. CUDA 2.7.1 is an explicit compatibility override
    # for Blackwell/RTX 50xx; 2.6.0 CUDA wheels cannot execute those architectures.
    torch_version, index = ("2.7.1", "cu128") if backend == "cuda" else ("2.6.0", "cpu")
    commands = [
        [
            str(python),
            "-m",
            "pip",
            "install",
            f"torch=={torch_version}",
            f"torchaudio=={torch_version}",
            "--index-url",
            f"https://download.pytorch.org/whl/{index}",
        ],
        [
            str(python),
            "-m",
            "pip",
            "install",
            "-r",
            str(ROOT / "requirements/chatterbox-runtime.txt"),
        ],
        # Dependencies above are explicit: upstream's torch==2.6.0 pin must not
        # downgrade Blackwell CUDA. This applies ONLY to the Chatterbox wheel.
        [str(python), "-m", "pip", "install", "--no-deps", "chatterbox-tts==0.1.7"],
        [str(python), "-m", "pip", "install", "-e", f"{ROOT}[radio]"],
    ]
    print(f"Setting up Chatterbox 0.1.7 ({backend}). No reference recording is needed.", flush=True)
    if backend == "cuda":
        print(
            "CUDA uses PyTorch 2.7.1 compatibility override; see docs/text-to-speech.md.",
            flush=True,
        )
    else:
        print("CPU speech may not meet racing latency/deadline requirements.", flush=True)
    for command in commands:
        subprocess.run(command, check=True)
    if not args.runtime_only:
        _, artifacts = load_manifest()
        install(
            ROOT / "data/components",
            tuple(a for a in artifacts if a.artifact_id.startswith("chatterbox-")),
        )
    print("Speech setup complete. Live inference is offline; cached models are reused.", flush=True)


if __name__ == "__main__":
    main()
