import json
import re
from pathlib import Path

from race_engineer.build_identity import read_build_identity
from race_engineer.components import load_manifest
from race_engineer.config import load_config
from race_engineer.installation import Installation
from race_engineer.version import __version__

ROOT = Path(__file__).parents[1]


def test_frozen_installation_keeps_program_and_user_data_separate(tmp_path: Path) -> None:
    install = tmp_path / "Program Files/Pitward"
    bundle = install / "_internal"
    local = tmp_path / "LocalAppData"
    layout = Installation.discover(
        environ={"LOCALAPPDATA": str(local)},
        executable=install / "Pitward.exe",
        bundle_root=bundle,
        frozen=True,
    )
    config = layout.apply_to(load_config(ROOT / "config/default.toml", environ={}))

    assert layout.config_path == bundle / "config/default.toml"
    assert layout.data_root == local / "Pitward"
    assert config.paths.database_path == local / "Pitward/race_engineer.sqlite3"
    assert config.conversation.runtime.model_path == (
        local / "Pitward/components/conversation/Qwen3-4B-Instruct-2507-Q4_K_M.gguf"
    )
    assert config.stt.worker_path == install / "workers/PitwardSTTWorker/PitwardSTTWorker.exe"
    assert config.radio_tts.worker_path == (
        install / "workers/PitwardTTSWorker/PitwardTTSWorker.exe"
    )
    missing = layout.missing_required_components(config)
    assert config.conversation.runtime.model_path in missing
    assert config.stt.model_path / "model.safetensors" in missing
    assert config.radio_tts.turkish_model_path in missing


def test_source_installation_preserves_repository_layout(tmp_path: Path) -> None:
    layout = Installation.discover(environ={}, cwd=tmp_path, frozen=False)
    config = load_config(ROOT / "config/default.toml", environ={})
    assert layout.data_root == tmp_path / "data"
    assert layout.logs_root == tmp_path / "logs"
    assert layout.apply_to(config) is config
    assert layout.missing_required_components(config) == ()


def test_build_identity_requires_matching_version_and_valid_commit(tmp_path: Path) -> None:
    commit = "a" * 40
    (tmp_path / "build-info.json").write_text(
        json.dumps({"version": __version__, "commit": commit}), encoding="utf-8"
    )
    assert read_build_identity(tmp_path).commit == "a" * 12
    (tmp_path / "build-info.json").write_text(
        json.dumps({"version": "999.0.0", "commit": commit}), encoding="utf-8"
    )
    assert read_build_identity(tmp_path).commit == "development"


def test_component_manifest_is_unique_pinned_and_confined() -> None:
    component_set, artifacts = load_manifest()
    ids = [artifact.artifact_id for artifact in artifacts]
    assert component_set == "1" and len(ids) == len(set(ids))
    assert sum(artifact.size for artifact in artifacts) > 4_000_000_000
    for artifact in artifacts:
        assert artifact.url.startswith("https://")
        assert re.fullmatch(r"[0-9a-f]{64}", artifact.sha256)
        assert artifact.size > 0 and not artifact.destination.is_absolute()
        assert ".." not in artifact.destination.parts
