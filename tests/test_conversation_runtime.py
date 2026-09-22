import asyncio
import subprocess
from pathlib import Path

import pytest

from race_engineer.config import ConversationConfig, ConversationRuntimeConfig
from race_engineer.conversation.runtime import (
    ConversationRuntimeError,
    ConversationServer,
    RuntimeCandidate,
    RuntimeProbe,
    probe_runtime,
    runtime_candidates,
    server_command,
)


def runtime_config(tmp_path: Path, **changes: object) -> ConversationRuntimeConfig:
    values: dict[str, object] = {
        "mode": "gpu",
        "cpu_executable_path": tmp_path / "cpu-server.exe",
        "accelerated_executable_path": tmp_path / "gpu-server.exe",
        "accelerated_backend": "cuda",
        "model_path": tmp_path / "model.gguf",
        "probe_timeout_s": 0.1,
        "startup_timeout_s": 0.1,
        "shutdown_timeout_s": 0.1,
    }
    values.update(changes)
    return ConversationRuntimeConfig.model_validate(values)


def test_runtime_probe_reads_only_reported_device_section(tmp_path, monkeypatch):
    executable = tmp_path / "llama-server.exe"
    executable.touch()

    def run(*args, **kwargs):
        return subprocess.CompletedProcess(
            args[0],
            0,
            stdout=b"loader: harmless log\nAvailable devices:\n  CUDA0: NVIDIA Test GPU\n",
            stderr=b"",
        )

    monkeypatch.setattr("race_engineer.conversation.runtime.subprocess.run", run)
    assert probe_runtime(executable, 1) == RuntimeProbe(("CUDA0",))


def test_runtime_probe_timeout_is_bounded(tmp_path, monkeypatch):
    executable = tmp_path / "llama-server.exe"
    executable.touch()

    def run(*args, **kwargs):
        raise subprocess.TimeoutExpired(args[0], kwargs["timeout"])

    monkeypatch.setattr("race_engineer.conversation.runtime.subprocess.run", run)
    assert probe_runtime(executable, 0.1) == RuntimeProbe((), "runtime_probe_timeout")


def test_accelerator_probe_failure_falls_back_to_cpu(tmp_path, monkeypatch):
    config = runtime_config(tmp_path)
    monkeypatch.setattr(
        "race_engineer.conversation.runtime.probe_runtime",
        lambda *_: RuntimeProbe((), "no_gpu_device"),
    )
    candidates, reason = asyncio.run(runtime_candidates(tmp_path, config))
    assert [candidate.mode for candidate in candidates] == ["cpu"]
    assert reason == "no_gpu_device"


def test_required_accelerator_fails_closed(tmp_path, monkeypatch):
    config = runtime_config(tmp_path, fallback_to_cpu=False)
    monkeypatch.setattr(
        "race_engineer.conversation.runtime.probe_runtime",
        lambda *_: RuntimeProbe((), "no_gpu_device"),
    )
    with pytest.raises(ConversationRuntimeError, match="no_gpu_device"):
        asyncio.run(runtime_candidates(tmp_path, config))


def test_unknown_configured_device_falls_back_to_cpu(tmp_path, monkeypatch):
    config = runtime_config(tmp_path, device="CUDA1")
    monkeypatch.setattr(
        "race_engineer.conversation.runtime.probe_runtime",
        lambda *_: RuntimeProbe(("CUDA0",)),
    )
    candidates, reason = asyncio.run(runtime_candidates(tmp_path, config))
    assert [candidate.mode for candidate in candidates] == ["cpu"]
    assert reason == "configured_device_unavailable"


def test_server_command_keeps_cpu_baseline_and_allows_explicit_gpu_device(tmp_path):
    runtime = runtime_config(tmp_path, device="CUDA0", gpu_layers=24)
    config = ConversationConfig(model="local-test", port=9000, runtime=runtime)
    cpu = RuntimeCandidate(
        "cpu", tmp_path / "cpu.exe", tmp_path / "model.gguf", "cpu", None, 0
    )
    gpu = RuntimeCandidate(
        "gpu", tmp_path / "gpu.exe", tmp_path / "model.gguf", "cuda", "CUDA0", 24
    )
    cpu_command = server_command(cpu, config)
    gpu_command = server_command(gpu, config)
    assert cpu_command[cpu_command.index("--n-gpu-layers") + 1] == "0"
    assert "--device" not in cpu_command
    assert gpu_command[gpu_command.index("--n-gpu-layers") + 1] == "24"
    assert gpu_command[gpu_command.index("--device") + 1] == "CUDA0"
    assert gpu_command[gpu_command.index("--alias") + 1] == "local-test"


def test_failed_accelerated_start_retries_cpu_once(tmp_path, monkeypatch):
    for name in ("cpu-server.exe", "gpu-server.exe", "model.gguf"):
        (tmp_path / name).touch()
    config = ConversationConfig(runtime=runtime_config(tmp_path))
    events: list[tuple[str, str]] = []
    spawned: list[str] = []

    class Process:
        def __init__(self, returncode: int | None) -> None:
            self.returncode = returncode

        def terminate(self) -> None:
            self.returncode = 0

        def kill(self) -> None:
            self.returncode = 0

        async def wait(self) -> int:
            return self.returncode or 0

    monkeypatch.setattr(
        "race_engineer.conversation.runtime.probe_runtime",
        lambda *_: RuntimeProbe(("CUDA0",)),
    )
    readiness = iter((False, True))
    monkeypatch.setattr(
        "race_engineer.conversation.runtime.model_ready", lambda *_: next(readiness)
    )

    async def spawn(executable, *args, **kwargs):
        del args, kwargs
        spawned.append(Path(executable).name)
        return Process(1 if "gpu" in Path(executable).name else None)

    monkeypatch.setattr("race_engineer.conversation.runtime.start_owned_process", spawn)

    async def exercise() -> None:
        server = ConversationServer(tmp_path, config, notify=lambda *event: events.append(event))
        await server.start()
        assert server.effective is not None and server.effective.mode == "cpu"
        await server.aclose()

    asyncio.run(exercise())
    assert spawned == ["gpu-server.exe", "cpu-server.exe"]
    assert ("compute", "cpu:model_startup_failed") in events
    assert events[-1] == ("compute", "cpu:cpu")
