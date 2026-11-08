"""Install pinned LFM3 assets and optional comparison profiles; never rewrite config."""

import argparse
import json
import tempfile
import tomllib
from dataclasses import dataclass
from pathlib import Path

from race_engineer.config import AppConfig

if __package__:
    from .setup_qwen_gpu import digest, download, extract_into, replace_table
else:
    from setup_qwen_gpu import digest, download, extract_into, replace_table

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "data/lfm3-prototype"
RUNTIME_REVISION = "b12088"
MODEL_REVISION = "cd0c74c575b5c416f3011b5b4c298077f26ca8c2"
MODEL_NAME = "LFM3-1.7B-Q4_K_M.gguf"
MODEL_SHA256 = "3793e1e6e2266919d6e015287221b41100175a98eddaba92e111aed3b05c93c4"
MODEL_ID = "LiquidAI/LFM3-1.7B"
PORT = 8091


@dataclass(frozen=True)
class Asset:
    name: str
    size: int
    sha256: str
    url: str


def release(name, size, sha256):
    return Asset(
        name,
        size,
        sha256,
        f"https://github.com/ggml-org/llama.cpp/releases/download/{RUNTIME_REVISION}/{name}",
    )


CPU = (
    release(
        "llama-b12088-bin-win-cpu-x64.zip",
        24629485,
        "d67dabc287170961d6517ad96b4a8b7fe3e830c1663ad096cd7b5f932484fc9b",
    ),
)
CUDA = (
    release(
        "llama-b12088-bin-win-cuda-13.4-x64.zip",
        169091292,
        "aaf9f9e5ca45465b8aa45fb388d44e48e97126688cdf7d3a289bf54089f119f4",
    ),
    release(
        "cudart-llama-bin-win-cuda-13.4-x64.zip",
        423535356,
        "738f8c251ac22b70c3ae6f83a10cf222725df0395246a2cf58f32bdb85fbe668",
    ),
)
MODEL = Asset(
    MODEL_NAME,
    1099627808,
    MODEL_SHA256,
    f"https://huggingface.co/LiquidAI/LFM3-1.7B-GGUF/resolve/{MODEL_REVISION}/{MODEL_NAME}",
)


def install_runtime(mode):
    assets = CUDA if mode == "gpu" else CPU
    name = f"llama-{RUNTIME_REVISION}-{'cuda-13.4' if mode == 'gpu' else 'cpu'}"
    destination = ASSETS / name
    expected = {asset.name: asset.sha256 for asset in assets}
    if destination.exists():
        marker = json.loads((destination / ".pitward-lfm-trial.json").read_text(encoding="utf-8"))
        if marker.get("archives") != expected or not marker.get("files"):
            raise ValueError("Existing LFM runtime provenance differs; refusing to overwrite")
        for filename, sha256 in marker["files"].items():
            target = (destination / filename).resolve()
            if not target.is_relative_to(destination.resolve()) or digest(target) != sha256:
                raise ValueError("Existing LFM runtime failed verification")
        return destination / "llama-server.exe"
    archives = [download(asset, ASSETS / ".downloads") for asset in assets]
    stage = Path(tempfile.mkdtemp(prefix=".stage-", dir=ASSETS))
    for archive in archives:
        extract_into(archive, stage)
    required = ("llama-server.exe", "ggml-cuda.dll") if mode == "gpu" else ("llama-server.exe",)
    if not all((stage / filename).is_file() for filename in required):
        raise ValueError(f"Incomplete runtime; diagnostic stage retained at {stage}")
    marker = {
        "revision": RUNTIME_REVISION,
        "archives": expected,
        "files": {str(p.relative_to(stage)): digest(p) for p in stage.rglob("*") if p.is_file()},
    }
    (stage / ".pitward-lfm-trial.json").write_text(json.dumps(marker, indent=2), encoding="utf-8")
    if not all(path.resolve().is_relative_to(ASSETS.resolve()) for path in (stage, destination)):
        raise ValueError("Runtime stage/destination escapes trial assets")
    stage.rename(destination)
    return destination / "llama-server.exe"


def trial_profile(source, mode, *, native=False):
    if mode not in {"cpu", "gpu"}:
        raise ValueError("Unknown trial mode")
    original = AppConfig.model_validate(tomllib.loads(source))
    profile = replace_table(
        source,
        "conversation",
        {
            "model": MODEL_ID,
            "port": 8092 if native else PORT,
            "context_planner": "lfm3-native-v1" if native else "qwen-context-v8",
            "default_language": "en",
        },
    )
    profile = replace_table(
        profile,
        "conversation.runtime",
        {
            "mode": mode,
            "cpu_executable_path": (
                f"data/lfm3-prototype/llama-{RUNTIME_REVISION}-cpu/llama-server.exe"
            ),
            "accelerated_executable_path": (
                f"data/lfm3-prototype/llama-{RUNTIME_REVISION}-cuda-13.4/llama-server.exe"
            ),
            "accelerated_backend": "cuda",
            "device": "CUDA0",
            "gpu_layers": -1,
            "model_path": f"data/lfm3-prototype/{MODEL_NAME}",
            "runtime_revision": RUNTIME_REVISION,
            "model_revision": MODEL_REVISION,
            "quantization": "Q4_K_M",
            "context_size": 8192,
            "parallel": 1,
            "enable_thinking": False,
            "fallback_to_cpu": False,
            "reuse_external_server": False,
        },
    )
    parsed = AppConfig.model_validate(tomllib.loads(profile))
    if (parsed.stt, parsed.tts, parsed.policy) != (original.stt, original.tts, original.policy):
        raise ValueError("Trial must preserve speech and telemetry policy settings")
    return (
        "# Generated English-first LFM3 comparison profile. Source configuration is unchanged.\n"
        + (
            "# lfm3-native-v1 selects native tool calls with existing evidence guards.\n"
            if native
            else "# qwen-context-v8 identifies the existing prompt policy, not the model weights.\n"
        )
        + profile
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config/default.toml")
    parser.add_argument("--backend", choices=("cpu", "gpu", "both"), default="both")
    parser.add_argument(
        "--runtime-only",
        action="store_true",
        help="Install/verify model and runtime assets without creating comparison profiles",
    )
    parser.add_argument(
        "--native-tools",
        action="store_true",
        help="Create separate native-adapter profiles on port 8092",
    )
    args = parser.parse_args()
    if not args.config.resolve().is_relative_to(ROOT):
        parser.error("Config must be inside this repository")
    source = args.config.read_text(encoding="utf-8")
    modes = ("cpu", "gpu") if args.backend == "both" else (args.backend,)
    profile_name = "lfm3-native" if args.native_tools else "lfm3"
    profiles = (
        {}
        if args.runtime_only
        else {
            ROOT
            / f"data/trials/{profile_name}-{mode}.toml": trial_profile(
                source, mode, native=args.native_tools
            )
            for mode in modes
        }
    )
    for path, contents in profiles.items():
        if path.exists() and path.read_text(encoding="utf-8") != contents:
            raise ValueError(f"Existing {path.name} differs; refusing to overwrite")
    ASSETS.mkdir(parents=True, exist_ok=True)
    for mode in modes:
        print(f"Installing isolated {mode} runtime...", flush=True)
        install_runtime(mode)
    download(MODEL, ASSETS)
    for path, contents in profiles.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            with path.open("x", encoding="utf-8", newline="\n") as stream:
                stream.write(contents)
        print(f"Trial config: {path.relative_to(ROOT)}", flush=True)
    print("LFM3 assets installed. English MVP; no Turkish training performed.")


if __name__ == "__main__":
    main()
