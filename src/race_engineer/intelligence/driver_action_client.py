"""Async ownership of a persistent, optional CPU action-classifier worker."""

import asyncio
import json
import os
import subprocess
from contextlib import suppress
from pathlib import Path

from race_engineer.config import DriverInterpreterConfig
from race_engineer.intelligence.driver_actions import (
    ActionModelMetadata,
    ActionPrediction,
    ActionRequest,
    load_action_metadata,
)
from race_engineer.processes import start_owned_process


class DriverInterpreterError(Exception):
    """Sanitized optional-runtime failure; the caller must fall back to Qwen."""


class LocalDriverActionClassifier:
    def __init__(self, config: DriverInterpreterConfig) -> None:
        self._config = config
        self._process: asyncio.subprocess.Process | None = None
        self._metadata: ActionModelMetadata | None = None
        self._lock = asyncio.Lock()

    async def _stop(self) -> None:
        process, self._process = self._process, None
        self._metadata = None
        if process is not None:
            if process.returncode is None:
                with suppress(ProcessLookupError):
                    process.kill()
            await process.wait()

    async def _read(self) -> dict[str, object]:
        assert self._process is not None and self._process.stdout is not None
        line = await self._process.stdout.readline()
        if not line:
            raise DriverInterpreterError("driver_interpreter_worker_exited")
        result = json.loads(line)
        if not isinstance(result, dict):
            raise DriverInterpreterError("driver_interpreter_protocol_error")
        return result

    async def _start(self) -> None:
        if self._process is not None and self._process.returncode is None:
            return
        config = self._config
        paths = (
            config.python_path.resolve(),
            config.base_model_path.resolve(),
            config.model_path.resolve(),
        )
        data_root = (Path.cwd() / "data").resolve()
        if any(not path.is_relative_to(data_root) for path in paths):
            raise DriverInterpreterError("driver_interpreter_paths_outside_data")
        python, base, model = paths
        if not python.is_file() or not base.is_dir() or not model.is_dir():
            raise DriverInterpreterError("driver_interpreter_installation_missing")
        metadata = load_action_metadata(model)
        environment = dict(os.environ)
        environment.update(
            HF_HUB_OFFLINE="1",
            TRANSFORMERS_OFFLINE="1",
            HF_HUB_DISABLE_TELEMETRY="1",
            PYTHONIOENCODING="utf-8",
        )
        self._process = await start_owned_process(
            str(python),
            "-u",
            "-m",
            "race_engineer.intelligence.driver_action_worker",
            "--base",
            str(base),
            "--model",
            str(model),
            "--threads",
            str(config.threads),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
            env=environment,
            limit=65536,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        ready = await self._read()
        if ready.get("ready") is not True or ready.get("metadata") != metadata.model_dump():
            raise DriverInterpreterError("driver_interpreter_model_load_failed")
        self._metadata = metadata

    async def _start_locked(self) -> None:
        try:
            await asyncio.wait_for(self._start(), self._config.startup_timeout_s)
        except BaseException as error:
            await self._stop()
            if isinstance(error, TimeoutError):
                raise DriverInterpreterError("driver_interpreter_startup_timeout") from error
            if isinstance(error, (OSError, ValueError)):
                raise DriverInterpreterError("driver_interpreter_start_failed") from error
            raise

    async def start(self) -> None:
        async with self._lock:
            await self._start_locked()

    async def predict(self, request: ActionRequest) -> ActionPrediction:
        async with self._lock:
            await self._start_locked()

            async def exchange() -> ActionPrediction:
                assert self._process is not None and self._process.stdin is not None
                self._process.stdin.write(request.model_dump_json().encode("utf-8") + b"\n")
                await self._process.stdin.drain()
                reply = await self._read()
                if reply.get("request_id") != request.request_id:
                    raise DriverInterpreterError("driver_interpreter_protocol_error")
                result = ActionPrediction.model_validate(reply.get("result"))
                if self._metadata is None or result.threshold != self._metadata.threshold:
                    raise DriverInterpreterError("driver_interpreter_protocol_error")
                return result

            try:
                return await asyncio.wait_for(exchange(), self._config.timeout_s)
            except BaseException as error:
                await self._stop()
                if isinstance(error, TimeoutError):
                    raise DriverInterpreterError("driver_interpreter_timeout") from error
                if isinstance(error, (OSError, ValueError)):
                    raise DriverInterpreterError("driver_interpreter_protocol_error") from error
                raise

    async def aclose(self) -> None:
        async with self._lock:
            await self._stop()
