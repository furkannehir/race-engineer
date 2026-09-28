"""Offline evaluation adapters. Importing this module loads no model packages."""

import asyncio
import json
import os
import subprocess
import time
from contextlib import suppress
from pathlib import Path
from typing import Any

from race_engineer.config import ConversationConfig
from race_engineer.conversation.local_model import _unique_object
from race_engineer.conversation.semantic import PROMPT, compose_scores, hypotheses, semantic_text
from race_engineer.core.dialogue import DialogueTurnInput, SemanticJudgeError, SemanticProposal
from race_engineer.processes import start_owned_process
from race_engineer.worker_lifecycle import stop_owned_worker


def proposal_schema() -> dict[str, Any]:
    schema = SemanticProposal.model_json_schema()
    for field in ("schema_version", "model_id", "calibration_id"):
        schema["properties"].pop(field)
    schema["required"] = list(schema["properties"])
    return schema


class QwenSemanticJudge:
    """Full-context v2 boundary, independent of the existing live v1 planner."""

    def __init__(self, config: ConversationConfig) -> None:
        self.config = config
        self.calls = 0

    async def _request_json(
        self,
        *,
        prompt: str,
        content: str,
        schema: dict[str, Any],
        max_tokens: int,
    ) -> dict[str, Any]:
        self.calls += 1
        writer: asyncio.StreamWriter | None = None
        try:
            async with asyncio.timeout(self.config.timeout_s):
                reader, writer = await asyncio.open_connection("127.0.0.1", self.config.port)
                body = json.dumps(
                    {
                        "model": self.config.model,
                        "messages": [
                            {"role": "system", "content": prompt},
                            {"role": "user", "content": content},
                        ],
                        "temperature": 0,
                        "max_tokens": max_tokens,
                        "stream": False,
                        "response_format": {"type": "json_object", "schema": schema},
                    },
                    ensure_ascii=False,
                ).encode("utf-8")
                header = (
                    "POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1\r\n"
                    "Content-Type: application/json\r\nConnection: close\r\n"
                    f"Content-Length: {len(body)}\r\n\r\n"
                ).encode("ascii")
                writer.write(header + body)
                await writer.drain()
                headers = await reader.readuntil(b"\r\n\r\n")
                if len(headers) > 8192 or headers.split(b"\r\n")[0].split()[1] != b"200":
                    raise SemanticJudgeError("judge_http_error")
                fields = dict(
                    line.lower().split(b":", 1)
                    for line in headers.split(b"\r\n")[1:]
                    if b":" in line
                )
                length = int(fields[b"content-length"])
                if not 0 < length <= 65536 or b"transfer-encoding" in fields:
                    raise SemanticJudgeError("judge_http_size")
                payload = json.loads(
                    await reader.readexactly(length), object_pairs_hook=_unique_object
                )
                choice = payload["choices"][0]
                if choice["finish_reason"] != "stop":
                    raise SemanticJudgeError("judge_response_incomplete")
                result = json.loads(choice["message"]["content"], object_pairs_hook=_unique_object)
                if not isinstance(result, dict) or not set(result) <= set(schema["properties"]):
                    raise SemanticJudgeError("judge_response_invalid")
                return result
        except (
            OSError,
            ValueError,
            KeyError,
            IndexError,
            TypeError,
            asyncio.IncompleteReadError,
            asyncio.LimitOverrunError,
        ) as error:
            raise SemanticJudgeError("judge_request_failed") from error
        finally:
            if writer is not None:
                writer.close()
                with suppress(OSError):
                    await writer.wait_closed()

    async def judge(self, request: DialogueTurnInput) -> SemanticProposal:
        result = await self._request_json(
            prompt=PROMPT,
            content=semantic_text(request),
            schema=proposal_schema(),
            max_tokens=512,
        )
        return SemanticProposal.model_validate({**result, "model_id": self.config.model})


class EncoderSemanticJudge:
    """One persistent CPU worker, no queue, killed on timeout or cancellation.

    Thresholds require a separately identified calibration run; uncalibrated
    candidates are usable for evaluation but never automatically promoted.
    """

    def __init__(
        self,
        *,
        python: Path,
        worker: Path,
        model_path: Path,
        candidate: str,
        package_path: Path | None = None,
        threads: int = 8,
        timeout_s: float = 30,
        startup_timeout_s: float = 120,
        threshold: float = 0.65,
        margin: float = 0.10,
        calibration_id: str | None = None,
    ) -> None:
        if candidate not in {"minilm", "laya"} or not 1 <= threads <= 32:
            raise ValueError("invalid encoder configuration")
        if not 0 < timeout_s <= 120 or not 0 < startup_timeout_s <= 300:
            raise ValueError("invalid encoder timeouts")
        if not 0.5 <= threshold <= 1 or not 0 <= margin <= 1:
            raise ValueError("invalid encoder thresholds")
        self.python, self.worker, self.model_path = python, worker, model_path
        self.candidate, self.package_path, self.threads = candidate, package_path, threads
        self.timeout_s, self.startup_timeout_s = timeout_s, startup_timeout_s
        self.threshold, self.margin, self.calibration_id = threshold, margin, calibration_id
        self.process: asyncio.subprocess.Process | None = None
        self.metadata: dict[str, Any] = {}
        self.last_scores: dict[str, float] = {}
        self._busy = False
        self.startup_ms: float | None = None

    async def _receive(self) -> dict[str, Any]:
        if self.process is None or self.process.stdout is None:
            raise SemanticJudgeError("judge_not_started")
        line = await self.process.stdout.readline()
        if not line or len(line) > 65536:
            raise SemanticJudgeError("judge_protocol_error")
        payload = json.loads(line, object_pairs_hook=_unique_object)
        if not isinstance(payload, dict):
            raise SemanticJudgeError("judge_worker_error")
        return payload

    async def start(self) -> None:
        if self.process is not None:
            raise SemanticJudgeError("judge_already_started")
        if not all(path.exists() for path in (self.python, self.worker, self.model_path)):
            raise SemanticJudgeError("judge_artifacts_missing")
        env = {
            **os.environ,
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "PYTHONUTF8": "1",
            "TOKENIZERS_PARALLELISM": "false",
        }
        # Never inherit a user's arbitrary module search path into the worker.
        env.pop("PYTHONPATH", None)
        if self.package_path is not None:
            env["PYTHONPATH"] = str(self.package_path.resolve())
        start = time.perf_counter()
        try:
            async with asyncio.timeout(self.startup_timeout_s):
                self.process = await start_owned_process(
                    str(self.python.resolve()),
                    str(self.worker.resolve()),
                    "--candidate",
                    self.candidate,
                    "--model",
                    str(self.model_path.resolve()),
                    "--threads",
                    str(self.threads),
                    env=env,
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.DEVNULL,
                    limit=65537,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
                )
                self.metadata = await self._receive()
                if self.metadata.get("ready") != self.candidate:
                    raise SemanticJudgeError("judge_protocol_error")
            self.startup_ms = (time.perf_counter() - start) * 1000
        except BaseException:
            await self.aclose()
            raise

    async def judge(self, request: DialogueTurnInput) -> SemanticProposal:
        if self._busy:
            raise SemanticJudgeError("judge_busy")
        if self.process is None or self.process.stdin is None:
            raise SemanticJudgeError("judge_not_started")
        self._busy = True
        self.last_scores = {}
        try:
            async with asyncio.timeout(self.timeout_s):
                message = {
                    "id": request.turn_id,
                    "state": semantic_text(request),
                    "hypotheses": hypotheses(),
                }
                self.process.stdin.write(json.dumps(message, ensure_ascii=False).encode() + b"\n")
                await self.process.stdin.drain()
                payload = await self._receive()
                if payload.get("id") != request.turn_id:
                    raise SemanticJudgeError("judge_protocol_error")
                if payload.get("error") == "context_limit":
                    raise SemanticJudgeError("judge_context_limit")
                scores = payload["scores"]
                if not isinstance(scores, dict):
                    raise SemanticJudgeError("judge_scores_invalid")
                self.last_scores = scores
                return compose_scores(
                    request,
                    scores,
                    model_id=self.candidate,
                    threshold=self.threshold,
                    margin=self.margin,
                    calibration_id=self.calibration_id,
                )
        except asyncio.CancelledError:
            await self.aclose()
            raise
        except (OSError, ValueError, TypeError, KeyError, SemanticJudgeError) as error:
            if isinstance(error, SemanticJudgeError) and str(error) == "judge_context_limit":
                raise
            await self.aclose()
            raise SemanticJudgeError("judge_inference_failed") from error
        finally:
            self._busy = False

    async def aclose(self) -> None:
        process, self.process = self.process, None
        if process is None:
            return
        await stop_owned_worker(process)
