import io
import json
import stat
import subprocess
import zipfile
from argparse import Namespace
from pathlib import Path

import pytest

from race_engineer.config import AppConfig, load_config
from scripts import benchmark_intelligence, setup_qwen_gpu

ROOT = Path(__file__).parents[1]


def test_trial_profiles_change_only_qwen_compute_and_isolate_the_port(tmp_path):
    import tomllib

    source = (ROOT / "config/default.toml").read_text(encoding="utf-8")
    original = AppConfig.model_validate(tomllib.loads(source))
    profiles = [
        AppConfig.model_validate(tomllib.loads(setup_qwen_gpu.trial_profile(source, mode)))
        for mode in ("cpu", "gpu")
    ]
    for profile in profiles:
        assert profile.stt == original.stt and profile.stt.device == "cpu"
        assert profile.tts == original.tts and profile.radio_tts == original.radio_tts
        assert profile.conversation.model == original.conversation.model
        assert profile.conversation.timeout_s == original.conversation.timeout_s
        assert profile.conversation.port == 8088
        assert not profile.conversation.runtime.reuse_external_server
        assert not profile.conversation.runtime.fallback_to_cpu
        assert profile.conversation.runtime.device == "CUDA0"
    cpu = profiles[0].model_dump()
    gpu = profiles[1].model_dump()
    cpu["conversation"]["runtime"]["mode"] = "gpu"
    assert cpu == gpu


def test_profile_generation_refuses_to_change_source_or_existing_trial_files(tmp_path, monkeypatch):
    source = (ROOT / "config/default.toml").read_text(encoding="utf-8")
    source_path = tmp_path / "default.toml"
    source_path.write_text(source, encoding="utf-8")
    monkeypatch.setattr(setup_qwen_gpu, "ROOT", tmp_path)
    monkeypatch.setattr(setup_qwen_gpu, "ASSETS", tmp_path)
    existing = tmp_path / "gpu-trial.toml"
    existing.write_text("user-edited profile", encoding="utf-8")
    with pytest.raises(ValueError, match="not overwriting"):
        setup_qwen_gpu.write_profiles(source_path)
    assert existing.read_text(encoding="utf-8") == "user-edited profile"
    assert source_path.read_text(encoding="utf-8") == source
    assert not (tmp_path / "cpu-trial.toml").exists()


def test_successful_profile_generation_is_idempotent_and_does_not_touch_default(
    tmp_path, monkeypatch
):
    source = (ROOT / "config/default.toml").read_text(encoding="utf-8")
    source_path = tmp_path / "default.toml"
    source_path.write_text(source, encoding="utf-8")
    monkeypatch.setattr(setup_qwen_gpu, "ROOT", tmp_path)
    monkeypatch.setattr(setup_qwen_gpu, "ASSETS", tmp_path)
    setup_qwen_gpu.write_profiles(source_path)
    gpu = tmp_path / "gpu-trial.toml"
    modified = gpu.stat().st_mtime_ns
    setup_qwen_gpu.write_profiles(source_path)
    assert gpu.stat().st_mtime_ns == modified
    assert load_config(gpu, environ={}).conversation.runtime.mode == "gpu"
    assert source_path.read_text(encoding="utf-8") == source


def test_profile_refuses_mismatched_runtime_revision_or_non_cpu_asr():
    source = (ROOT / "config/default.toml").read_text(encoding="utf-8")
    for changed in (
        source.replace('runtime_revision = "b10964"', 'runtime_revision = "other"'),
        source.replace('device = "cpu"', 'device = "cuda"'),
    ):
        with pytest.raises(ValueError):
            setup_qwen_gpu.trial_profile(changed, "gpu")


def make_zip(path, entries):
    with zipfile.ZipFile(path, "w") as archive:
        for name, contents in entries:
            archive.writestr(name, contents)


@pytest.mark.parametrize(
    "name",
    ["../escape.exe", "/escape.exe", "C:/escape.exe", "nested/../../escape.exe", "..\\escape.exe"],
)
def test_gpu_archive_extraction_rejects_escaping_targets(tmp_path, name):
    archive, staging = tmp_path / "download.zip", tmp_path / "staging"
    staging.mkdir()
    make_zip(archive, [(name, b"bad")])
    with pytest.raises(ValueError, match="Unsafe archive path"):
        setup_qwen_gpu.extract_into(archive, staging)
    assert not (tmp_path / "escape.exe").exists()


def test_gpu_archive_extraction_rejects_symlinks(tmp_path):
    archive, staging = tmp_path / "download.zip", tmp_path / "staging"
    staging.mkdir()
    info = zipfile.ZipInfo("link")
    info.create_system = 3
    info.external_attr = (stat.S_IFLNK | 0o777) << 16
    with zipfile.ZipFile(archive, "w") as stream:
        stream.writestr(info, "../outside")
    with pytest.raises(ValueError, match="symlinks"):
        setup_qwen_gpu.extract_into(archive, staging)


def test_dependency_archives_can_share_identical_files_but_never_overwrite_conflicts(tmp_path):
    archive, staging = tmp_path / "download.zip", tmp_path / "staging"
    staging.mkdir()
    make_zip(archive, [("shared.dll", b"original")])
    setup_qwen_gpu.extract_into(archive, staging)
    setup_qwen_gpu.extract_into(archive, staging)
    make_zip(archive, [("shared.dll", b"different")])
    with pytest.raises(ValueError, match="Conflicting"):
        setup_qwen_gpu.extract_into(archive, staging)
    assert (staging / "shared.dll").read_bytes() == b"original"


def test_download_verifies_size_and_checksum_before_installing(tmp_path, monkeypatch):
    payload = b"pinned archive"
    monkeypatch.setattr(
        setup_qwen_gpu.urllib.request, "urlopen", lambda *args, **kwargs: io.BytesIO(payload)
    )
    asset = setup_qwen_gpu.ReleaseAsset("test.zip", len(payload), "0" * 64)
    with pytest.raises(ValueError, match="Checksum/size failed"):
        setup_qwen_gpu.download(asset, tmp_path)
    assert not (tmp_path / "test.zip").exists()
    assert (tmp_path / "test.zip.partial").read_bytes() == payload


def test_download_never_replaces_existing_bad_file(tmp_path, monkeypatch):
    path = tmp_path / "test.zip"
    path.write_bytes(b"user file")
    monkeypatch.setattr(
        setup_qwen_gpu.urllib.request,
        "urlopen",
        lambda *args, **kwargs: pytest.fail("must not download"),
    )
    with pytest.raises(ValueError, match="not overwriting"):
        setup_qwen_gpu.download(setup_qwen_gpu.ReleaseAsset("test.zip", 10, "0" * 64), tmp_path)
    assert path.read_bytes() == b"user file"


def test_existing_gpu_runtime_requires_complete_provenance_and_unchanged_files(
    tmp_path, monkeypatch
):
    runtime = tmp_path / "cuda"
    runtime.mkdir()
    monkeypatch.setattr(setup_qwen_gpu, "ASSETS", tmp_path)
    monkeypatch.setattr(setup_qwen_gpu, "RUNTIME", runtime)
    original = {name: b"pinned file" for name in ("llama-server.exe", "ggml-cuda.dll")}
    for name, value in original.items():
        (runtime / name).write_bytes(value)
    marker = {
        "revision": setup_qwen_gpu.REVISION,
        "archives": {asset.name: asset.sha256 for asset in setup_qwen_gpu.RELEASE_ASSETS},
        "files": {name: setup_qwen_gpu.digest(runtime / name) for name in original},
    }
    marker_path = runtime / ".pitward-gpu-runtime.json"
    marker_path.write_text(json.dumps(marker), encoding="utf-8")
    assert setup_qwen_gpu.install_runtime() == runtime / "llama-server.exe"
    (runtime / "ggml-cuda.dll").write_bytes(b"changed file")
    with pytest.raises(ValueError, match="failed verification"):
        setup_qwen_gpu.install_runtime()
    marker["files"] = {}
    marker_path.write_text(json.dumps(marker), encoding="utf-8")
    with pytest.raises(ValueError, match="incomplete"):
        setup_qwen_gpu.install_runtime()


def test_controlled_benchmark_rejects_unknown_external_backend():
    class Server:
        process = effective = None

    with pytest.raises(ValueError, match="external server"):
        benchmark_intelligence.verify_trial_runtime(Server(), Namespace(require_owned_runtime=True))
    with pytest.raises(ValueError, match="Expected owned cuda"):
        benchmark_intelligence.verify_trial_runtime(Server(), Namespace(expect_backend="cuda"))


def test_gpu_memory_report_is_whole_device_and_handles_wddm_without_claiming_zero(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(benchmark_intelligence.shutil, "which", lambda _: "nvidia-smi")

    def run(*args, **kwargs):
        return subprocess.CompletedProcess(
            args[0], 0, stdout="0, NVIDIA GeForce RTX 5080, 7200, 16303\n"
        )

    monkeypatch.setattr(benchmark_intelligence.subprocess, "run", run)
    report = benchmark_intelligence.gpu_memory_snapshot()
    assert report == {
        "available": True,
        "devices": [
            {"index": 0, "name": "NVIDIA GeForce RTX 5080", "used_mib": 7200, "total_mib": 16303},
        ],
    }


def test_gpu_memory_unknown_measurement_is_unavailable_not_zero(monkeypatch):
    monkeypatch.setattr(benchmark_intelligence.shutil, "which", lambda _: None)
    assert benchmark_intelligence.gpu_memory_snapshot()["available"] is False


def test_gpu_launcher_uses_main_default_and_regular_launcher_forwards_options():
    gpu = (ROOT / "start-panel-gpu.cmd").read_text(encoding="utf-8")
    regular = (ROOT / "start-panel.cmd").read_text(encoding="utf-8")
    assert 'start-panel.cmd" %*' in gpu
    assert "gpu-trial.toml" not in gpu
    assert "-m race_engineer.ui %*" in regular


def test_runtime_only_setup_never_touches_existing_profiles(monkeypatch):
    from race_engineer.conversation.runtime import RuntimeProbe

    monkeypatch.setattr("sys.argv", ["setup_qwen_gpu.py", "--runtime-only"])
    monkeypatch.setattr(setup_qwen_gpu, "install_runtime", lambda: Path("cuda/llama-server.exe"))
    monkeypatch.setattr(setup_qwen_gpu, "probe_runtime", lambda *_: RuntimeProbe(("CUDA0",), None))
    monkeypatch.setattr(
        setup_qwen_gpu, "write_profiles",
        lambda *_: pytest.fail("runtime-only setup must not create or overwrite profiles"),
    )
    assert setup_qwen_gpu.main() == 0


def test_profile_generation_accepts_new_directory_without_touching_legacy_profiles(
    tmp_path, monkeypatch
):
    source = (ROOT / "config/default.toml").read_text(encoding="utf-8")
    source_path = tmp_path / "default.toml"
    source_path.write_text(source, encoding="utf-8")
    monkeypatch.setattr(setup_qwen_gpu, "ROOT", tmp_path)
    monkeypatch.setattr(setup_qwen_gpu, "ASSETS", tmp_path)
    legacy = tmp_path / "gpu-trial.toml"
    legacy.write_text("legacy interpreter profile", encoding="utf-8")
    directory = tmp_path / "new-comparison"
    setup_qwen_gpu.write_profiles(source_path, directory)
    assert load_config(directory / "cpu-trial.toml", environ={}).conversation.runtime.mode == "cpu"
    assert load_config(directory / "gpu-trial.toml", environ={}).conversation.runtime.mode == "gpu"
    assert legacy.read_text(encoding="utf-8") == "legacy interpreter profile"
