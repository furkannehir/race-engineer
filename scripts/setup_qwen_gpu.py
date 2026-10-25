"""Explicit source-mode setup of a pinned CUDA runtime and optional CPU/GPU trial profiles."""

import argparse
import hashlib
import json
import shutil
import stat
import tempfile
import tomllib
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

from race_engineer.config import AppConfig
from race_engineer.conversation.runtime import probe_runtime

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "data/conversation-prototype"
REVISION = "b10964"
RUNTIME_NAME = "llama-b10964-cuda-13.3"
RUNTIME = ASSETS / RUNTIME_NAME
PROFILE_PORT = 8088


@dataclass(frozen=True)
class ReleaseAsset:
    name: str
    size: int
    sha256: str

    @property
    def url(self) -> str:
        return f"https://github.com/ggml-org/llama.cpp/releases/download/{REVISION}/{self.name}"


# GitHub's b10964 release asset size/digest metadata, checked 2026-10-11.
RELEASE_ASSETS = (
    ReleaseAsset(
        "llama-b10964-bin-win-cuda-13.3-x64.zip",
        149703589,
        "cd63ae76ad78a1540aa0f30f6c6284bab14c146d99a58f70c3f0a38cb9c62351",
    ),
    ReleaseAsset(
        "cudart-llama-bin-win-cuda-13.3-x64.zip",
        390970417,
        "1462a050eb4c684921ba51dcc4cc488a036674c3e73e9945ee705b854808d03e",
    ),
)


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def download(asset: ReleaseAsset, directory: Path) -> Path:
    target = directory / asset.name

    def valid(path: Path) -> bool:
        return path.stat().st_size == asset.size and digest(path) == asset.sha256

    if target.exists():
        if not target.is_file() or not valid(target):
            raise ValueError(f"Existing {target.name} failed verification; not overwriting")
        print(f"Keeping verified {target.name}", flush=True)
        return target
    directory.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".partial")
    if partial.exists():
        if not partial.is_file() or not valid(partial):
            raise ValueError(f"Incomplete {partial.name}; retained, not overwriting")
        partial.rename(target)
        return target
    print(f"Downloading {asset.name} ({asset.size / 1_000_000:.1f} MB)...", flush=True)
    request = urllib.request.Request(asset.url, headers={"User-Agent": "Pitward GPU trial setup"})
    with urllib.request.urlopen(request, timeout=60) as response, partial.open("xb") as output:
        received = 0
        next_progress = 32 * 1024 * 1024
        while chunk := response.read(1024 * 1024):
            output.write(chunk)
            received += len(chunk)
            if received >= next_progress:
                print(f"  {received / asset.size:.0%}", flush=True)
                next_progress += 32 * 1024 * 1024
    if not valid(partial):
        raise ValueError(f"Checksum/size failed for {asset.name}; .partial retained")
    partial.rename(target)
    return target


def extract_into(archive_path: Path, staging: Path) -> None:
    root = staging.resolve()
    with zipfile.ZipFile(archive_path) as archive:
        for item in archive.infolist():
            name = item.filename.replace("\\", "/")
            target = (root / name).resolve()
            if ":" in name or not target.is_relative_to(root):
                raise ValueError("Unsafe archive path")
            if stat.S_ISLNK(item.external_attr >> 16):
                raise ValueError("Archive symlinks are not allowed")
            if item.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                # The executable and dependency archives may contain a shared file.
                with archive.open(item) as stream:
                    incoming = hashlib.file_digest(stream, "sha256").hexdigest()
                if digest(target) != incoming:
                    raise ValueError("Conflicting archive files; refusing to overwrite")
                continue
            with archive.open(item) as source, target.open("xb") as output:
                shutil.copyfileobj(source, output)


def install_runtime() -> Path:
    ASSETS.mkdir(parents=True, exist_ok=True)
    if RUNTIME.exists():
        marker = json.loads((RUNTIME / ".pitward-gpu-runtime.json").read_text(encoding="utf-8"))
        expected = {asset.name: asset.sha256 for asset in RELEASE_ASSETS}
        if (
            not isinstance(marker, dict)
            or marker.get("revision") != REVISION
            or marker.get("archives") != expected
        ):
            raise ValueError("Existing GPU runtime has unexpected provenance; not overwriting")
        file_digests = marker.get("files")
        if not isinstance(file_digests, dict) or not {"llama-server.exe", "ggml-cuda.dll"}.issubset(
            file_digests
        ):
            raise ValueError("Existing GPU runtime is incomplete; not overwriting")
        for name, expected_digest in file_digests.items():
            path = (RUNTIME / name).resolve()
            if not path.is_relative_to(RUNTIME.resolve()) or digest(path) != expected_digest:
                raise ValueError("Existing GPU runtime failed verification; not overwriting")
        if not (RUNTIME / "llama-server.exe").is_file():
            raise ValueError("Existing GPU runtime is incomplete; not overwriting")
        print("Keeping verified CUDA runtime", flush=True)
        return RUNTIME / "llama-server.exe"
    archives = [download(asset, ASSETS / ".gpu-downloads") for asset in RELEASE_ASSETS]
    staging = Path(tempfile.mkdtemp(prefix=".cuda-stage-", dir=ASSETS))
    # Preserve a failed stage for diagnosis; never delete a pre-existing user directory.
    for archive in archives:
        extract_into(archive, staging)
    if not all((staging / name).is_file() for name in ("llama-server.exe", "ggml-cuda.dll")):
        raise ValueError(f"Required CUDA runtime files are missing; stage retained at {staging}")
    marker = {
        "revision": REVISION,
        "archives": {asset.name: asset.sha256 for asset in RELEASE_ASSETS},
        "files": {
            str(path.relative_to(staging)): digest(path)
            for path in staging.rglob("*")
            if path.is_file()
        },
    }
    (staging / ".pitward-gpu-runtime.json").write_text(
        json.dumps(marker, indent=2), encoding="utf-8"
    )
    staging.rename(RUNTIME)
    return RUNTIME / "llama-server.exe"


def replace_table(source: str, table: str, updates: dict[str, object]) -> str:
    lines = source.splitlines()
    header = f"[{table}]"
    start = next(index for index, line in enumerate(lines) if line.strip() == header)
    end = next(
        (index for index in range(start + 1, len(lines)) if lines[index].strip().startswith("[")),
        len(lines),
    )
    retained = [
        line for line in lines[start + 1 : end] if line.split("=", 1)[0].strip() not in updates
    ]
    rendered = [
        f"{key} = {json.dumps(value, ensure_ascii=False)}" for key, value in updates.items()
    ]
    return "\n".join([*lines[: start + 1], *rendered, *retained, *lines[end:]]) + "\n"


def trial_profile(source: str, mode: str) -> str:
    original = AppConfig.model_validate(tomllib.loads(source))
    if mode not in {"cpu", "gpu"} or original.conversation.runtime.runtime_revision != REVISION:
        raise ValueError("Trial requires the pinned b10964 baseline and cpu/gpu mode")
    if original.stt.device != "cpu":
        raise ValueError("This isolated Qwen trial requires CPU speech recognition")
    profile = replace_table(source, "conversation", {"port": PROFILE_PORT})
    profile = replace_table(
        profile,
        "conversation.runtime",
        {
            "mode": mode,
            "accelerated_executable_path": (
                f"data/conversation-prototype/{RUNTIME_NAME}/llama-server.exe"
            ),
            "accelerated_backend": "cuda",
            "device": "CUDA0",
            "gpu_layers": -1,
            "fallback_to_cpu": False,
            "reuse_external_server": False,
        },
    )
    parsed = AppConfig.model_validate(tomllib.loads(profile))
    if (parsed.stt, parsed.tts) != (original.stt, original.tts):
        raise ValueError("Trial must not change speech or voice settings")
    return (
        "# Generated Qwen-only trial; default.toml is unchanged. No silent CPU/external fallback.\n"
        + profile
    )


def write_profiles(source_path: Path, destination: Path | None = None) -> None:
    source = source_path.read_text(encoding="utf-8")
    directory = ASSETS if destination is None else destination
    # Validate both and check ALL destinations before creating either file.
    profiles = {
        directory / f"{mode}-trial.toml": trial_profile(source, mode) for mode in ("cpu", "gpu")
    }
    for path, contents in profiles.items():
        if path.exists() and path.read_text(encoding="utf-8") != contents:
            raise ValueError(f"Existing {path.name} differs; not overwriting")
    directory.mkdir(parents=True, exist_ok=True)
    for path, contents in profiles.items():
        if not path.exists():
            with path.open("x", encoding="utf-8", newline="\n") as stream:
                stream.write(contents)
        print(f"Trial profile: {path.relative_to(ROOT)}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config/default.toml")
    parser.add_argument(
        "--runtime-only", action="store_true",
        help="Install/verify CUDA without creating or touching comparison profiles",
    )
    parser.add_argument(
        "--profiles-dir", type=Path,
        help="Create comparison profiles in a new repository directory; never overwrite",
    )
    args = parser.parse_args()
    if not args.config.resolve().is_relative_to(ROOT):
        parser.error("source config must be inside the repository")
    if args.runtime_only and args.profiles_dir is not None:
        parser.error("--runtime-only cannot be combined with --profiles-dir")
    if args.profiles_dir is not None and not args.profiles_dir.resolve().is_relative_to(ROOT):
        parser.error("profiles directory must be inside the repository")
    print(
        "Source-mode NVIDIA/CUDA setup; no driver, system toolkit or Python changes.",
        flush=True,
    )
    print("llama.cpp is MIT; CUDA DLL redistribution terms remain NVIDIA's.", flush=True)
    try:
        # Reject an incompatible source before any download or installation.
        trial_profile(args.config.read_text(encoding="utf-8"), "gpu")
        executable = install_runtime()
        probe = probe_runtime(executable, 20)
        if "CUDA0" not in probe.devices:
            raise ValueError(f"CUDA0 could not be detected ({probe.reason or 'device missing'})")
        print(f"Available CUDA devices: {', '.join(probe.devices)}", flush=True)
        if not args.runtime_only:
            write_profiles(args.config, args.profiles_dir)
    except (OSError, ValueError, RuntimeError, zipfile.BadZipFile) as error:
        parser.error(str(error))
    print("Ready. Close the existing panel, then run .\\start-panel.cmd", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
