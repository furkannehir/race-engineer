"""Bounded, hardware-neutral lifecycle for a local llama.cpp server."""

import asyncio
import http.client
import json
import logging
import os
import re
import subprocess
import time
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from race_engineer.config import ConversationConfig, ConversationRuntimeConfig
from race_engineer.processes import start_owned_process

_LOGGER = logging.getLogger(__name__)
_DEVICE_LINE = re.compile(r"^\s*([^\s:]+):\s+(.+?)\s*$")


class ConversationRuntimeError(RuntimeError):
    """Sanitized local-runtime failure code."""


@dataclass(frozen=True)
class RuntimeProbe:
    devices: tuple[str, ...]
    reason: str | None = None

    @property
    def available(self) -> bool:
        return bool(self.devices) and self.reason is None


@dataclass(frozen=True)
class RuntimeCandidate:
    mode: Literal["cpu", "gpu"]
    executable: Path
    model_path: Path
    backend: str
    device: str | None
    gpu_layers: int


def _resolve(root: Path, configured: Path) -> Path:
    return configured if configured.is_absolute() else root / configured


def model_ready(port: int, expected_model: str, *, timeout_s: float = 1) -> bool:
    """Check a literal loopback endpoint without redirects, proxies, or credentials."""
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout_s)
    try:
        connection.request("GET", "/v1/models")
        response = connection.getresponse()
        if response.status != 200:
            return False
        payload = json.loads(response.read(65_536))
        return any(item.get("id") == expected_model for item in payload.get("data", []))
    except (OSError, ValueError, http.client.HTTPException, AttributeError, TypeError):
        return False
    finally:
        connection.close()


def probe_runtime(executable: Path, timeout_s: float) -> RuntimeProbe:
    """Ask one installed runtime which accelerator devices it can actually see."""
    if not executable.is_file():
        return RuntimeProbe((), "accelerated_runtime_missing")
    try:
        result = subprocess.run(
            [str(executable), "--list-devices"],
            check=False,
            capture_output=True,
            timeout=timeout_s,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
    except subprocess.TimeoutExpired:
        return RuntimeProbe((), "runtime_probe_timeout")
    except OSError:
        return RuntimeProbe((), "runtime_probe_failed")
    if result.returncode != 0:
        return RuntimeProbe((), "runtime_probe_failed")
    output = (result.stdout + b"\n" + result.stderr).decode("utf-8", errors="replace")
    in_device_section = False
    found: list[str] = []
    for line in output.splitlines():
        if line.strip() == "Available devices:":
            in_device_section = True
            continue
        if not in_device_section:
            continue
        match = _DEVICE_LINE.match(line)
        if match is not None:
            found.append(match.group(1))
        elif found and not line.strip():
            break
    devices = tuple(found)
    return RuntimeProbe(tuple(dict.fromkeys(devices)), None if devices else "no_gpu_device")


def _cpu_candidate(root: Path, config: ConversationRuntimeConfig) -> RuntimeCandidate:
    return RuntimeCandidate(
        mode="cpu",
        executable=_resolve(root, config.cpu_executable_path),
        model_path=_resolve(root, config.model_path),
        backend="cpu",
        device=None,
        gpu_layers=0,
    )


async def runtime_candidates(
    root: Path, config: ConversationRuntimeConfig
) -> tuple[tuple[RuntimeCandidate, ...], str | None]:
    """Return ordered launch attempts and any reason acceleration was skipped."""
    cpu = _cpu_candidate(root, config)
    if config.mode == "cpu":
        return (cpu,), None

    accelerated_path = config.accelerated_executable_path
    if accelerated_path is None:
        if not config.fallback_to_cpu:
            raise ConversationRuntimeError("accelerated_runtime_not_configured")
        return (cpu,), "accelerated_runtime_not_configured"

    executable = _resolve(root, accelerated_path)
    probe = await asyncio.to_thread(probe_runtime, executable, config.probe_timeout_s)
    if not probe.available:
        if not config.fallback_to_cpu:
            raise ConversationRuntimeError(probe.reason or "accelerated_runtime_unavailable")
        return (cpu,), probe.reason or "accelerated_runtime_unavailable"
    if config.device is not None and config.device not in probe.devices:
        if not config.fallback_to_cpu:
            raise ConversationRuntimeError("configured_device_unavailable")
        return (cpu,), "configured_device_unavailable"

    accelerated = RuntimeCandidate(
        mode="gpu",
        executable=executable,
        model_path=_resolve(root, config.model_path),
        backend=config.accelerated_backend or "gpu",
        device=config.device,
        gpu_layers=config.gpu_layers,
    )
    if config.fallback_to_cpu:
        return (accelerated, cpu), None
    return (accelerated,), None


def server_command(candidate: RuntimeCandidate, config: ConversationConfig) -> tuple[str, ...]:
    arguments = [
        str(candidate.executable),
        "--model",
        str(candidate.model_path),
        "--alias",
        config.model,
        "--host",
        "127.0.0.1",
        "--port",
        str(config.port),
        "--cors-origins",
        f"http://127.0.0.1:{config.port}",
        "--ctx-size",
        str(config.runtime.context_size),
        "--parallel",
        str(config.runtime.parallel),
        "--threads",
        str(config.runtime.threads),
        "--n-gpu-layers",
        str(candidate.gpu_layers),
    ]
    if candidate.device is not None:
        arguments.extend(("--device", candidate.device))
    return tuple(arguments)


class ConversationServer:
    """Reuse an external server or own one local child with one bounded CPU fallback."""

    def __init__(
        self,
        root: Path,
        config: ConversationConfig,
        *,
        notify: Callable[[str, str], None] = lambda *_: None,
        quiet: bool = True,
    ) -> None:
        self.root = root
        self.config = config
        self.notify = notify
        self.quiet = quiet
        self.process: asyncio.subprocess.Process | None = None
        self.effective: RuntimeCandidate | None = None

    async def _stop_process(self) -> None:
        process, self.process = self.process, None
        if process is None:
            return
        if process.returncode is None:
            with suppress(ProcessLookupError):
                process.terminate()
        try:
            await asyncio.wait_for(process.wait(), self.config.runtime.shutdown_timeout_s)
        except TimeoutError:
            with suppress(ProcessLookupError):
                process.kill()
            await process.wait()

    async def _start_candidate(self, candidate: RuntimeCandidate) -> None:
        if not candidate.executable.is_file():
            raise ConversationRuntimeError(
                "cpu_runtime_missing" if candidate.mode == "cpu" else "accelerated_runtime_missing"
            )
        if not candidate.model_path.is_file():
            raise ConversationRuntimeError("model_weights_missing")
        output = asyncio.subprocess.DEVNULL if self.quiet else None
        try:
            self.process = await start_owned_process(
                *server_command(candidate, self.config),
                stdout=output,
                stderr=output,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
        except OSError as error:
            raise ConversationRuntimeError("model_startup_failed") from error
        deadline = time.monotonic() + self.config.runtime.startup_timeout_s
        while time.monotonic() < deadline:
            if self.process.returncode is not None:
                raise ConversationRuntimeError("model_startup_failed")
            if await asyncio.to_thread(model_ready, self.config.port, self.config.model):
                self.effective = candidate
                self.notify("compute", f"{candidate.mode}:{candidate.backend}")
                return
            await asyncio.sleep(0.25)
        raise ConversationRuntimeError("model_startup_timeout")

    async def start(self) -> None:
        if await asyncio.to_thread(model_ready, self.config.port, self.config.model):
            self.notify("compute", "external:unmanaged")
            return
        self.notify("phase", "loading_model")
        candidates, skipped_reason = await runtime_candidates(self.root, self.config.runtime)
        if skipped_reason is not None:
            self.notify("compute", f"cpu:{skipped_reason}")
            _LOGGER.warning(
                "conversation acceleration unavailable; using CPU",
                extra={"event": "conversation_compute_fallback", "reason": skipped_reason},
            )

        last_error: ConversationRuntimeError | None = None
        for index, candidate in enumerate(candidates):
            try:
                await self._start_candidate(candidate)
                return
            except asyncio.CancelledError:
                await self._stop_process()
                raise
            except ConversationRuntimeError as error:
                last_error = error
                await self._stop_process()
                if index + 1 < len(candidates) and str(error) != "model_weights_missing":
                    self.notify("compute", f"cpu:{error}")
                    _LOGGER.warning(
                        "conversation accelerator failed; retrying on CPU",
                        extra={"event": "conversation_compute_fallback", "reason": str(error)},
                    )
                    continue
                break
        raise last_error or ConversationRuntimeError("model_startup_failed")

    async def aclose(self) -> None:
        await self._stop_process()


async def run_server_until_exit(server: ConversationServer) -> int:
    """Foreground entry point used by the standalone launcher."""
    try:
        await server.start()
        if server.process is None:
            return 0
        return await server.process.wait()
    finally:
        await server.aclose()
