"""Small-model Context Engineer candidate over the dynamic evidence catalog."""

import asyncio
import json
from collections.abc import Sequence
from contextlib import suppress
from pathlib import Path
from typing import Protocol

from race_engineer.core.intelligence import (
    CapabilityDescriptor,
    CapabilityRequest,
    ContextPlan,
    DriverTurn,
    EvidenceQuery,
    SignalDescriptor,
)
from race_engineer.intelligence.local_model import LocalIntelligenceError

_MAX_HYPOTHESES = 32


class SemanticScorer(Protocol):
    async def score(
        self,
        text: str,
        hypotheses: Sequence[str],
        *,
        language: str | None,
    ) -> tuple[float, ...]: ...


class LocalSemanticWorker:
    """Keep one optional PyTorch candidate resident behind a JSON-lines subprocess."""

    def __init__(
        self,
        root: Path,
        *,
        python_path: Path,
        backend: str,
        model_path: Path,
        package_path: Path | None = None,
        model_revision: str,
        threads: int = 8,
        startup_timeout_s: float = 60.0,
        request_timeout_s: float = 10.0,
    ) -> None:
        self.root = root.resolve()
        self.python_path = self._file(python_path)
        self.model_path = self._directory(model_path)
        self.package_path = (
            self._directory(package_path) if package_path is not None else None
        )
        if backend not in {"minilm", "laya"}:
            raise ValueError("unsupported semantic worker backend")
        if not model_revision:
            raise ValueError("semantic worker model revision is required")
        if not 1 <= threads <= 64:
            raise ValueError("semantic worker threads must be between one and sixty-four")
        self.backend = backend
        self.model_revision = model_revision
        self.threads = threads
        self.startup_timeout_s = startup_timeout_s
        self.request_timeout_s = request_timeout_s
        self._process: asyncio.subprocess.Process | None = None
        self._lock = asyncio.Lock()
        self._request_id = 0
        self.runtime_metadata: dict[str, object] | None = None

    def _file(self, path: Path) -> Path:
        resolved = (path if path.is_absolute() else self.root / path).resolve()
        if not resolved.is_relative_to(self.root) or not resolved.is_file():
            raise ValueError("semantic worker executable must be inside the repository")
        return resolved

    def _directory(self, path: Path) -> Path:
        resolved = (path if path.is_absolute() else self.root / path).resolve()
        if not resolved.is_relative_to(self.root) or not resolved.is_dir():
            raise ValueError("semantic worker directory must be inside the repository")
        return resolved

    async def _start(self) -> None:
        arguments = [
            str(self.python_path),
            "-m",
            "race_engineer.intelligence.semantic_worker_main",
            "--backend",
            self.backend,
            "--model",
            str(self.model_path),
            "--threads",
            str(self.threads),
        ]
        if self.package_path is not None:
            arguments.extend(("--package-path", str(self.package_path)))
        self._process = await asyncio.create_subprocess_exec(
            *arguments,
            cwd=str(self.root),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        assert self._process.stdout is not None
        try:
            raw = await asyncio.wait_for(
                self._process.stdout.readline(),
                timeout=self.startup_timeout_s,
            )
            ready = json.loads(raw)
        except (TimeoutError, json.JSONDecodeError, UnicodeDecodeError) as error:
            await self.aclose()
            raise LocalIntelligenceError("semantic_worker_startup_failed") from error
        if (
            not isinstance(ready, dict)
            or ready.get("status") != "ready"
            or ready.get("backend") != self.backend
            or not isinstance(ready.get("runtime"), dict)
        ):
            await self.aclose()
            raise LocalIntelligenceError("semantic_worker_startup_failed")
        self.runtime_metadata = dict(ready["runtime"])

    async def score(
        self,
        text: str,
        hypotheses: Sequence[str],
        *,
        language: str | None,
    ) -> tuple[float, ...]:
        if not hypotheses or len(hypotheses) > _MAX_HYPOTHESES:
            raise ValueError("semantic score request has an invalid hypothesis count")
        async with self._lock:
            if self._process is None:
                await self._start()
            process = self._process
            assert process is not None
            if process.returncode is not None or process.stdin is None or process.stdout is None:
                raise LocalIntelligenceError("semantic_worker_unavailable")
            self._request_id += 1
            request_id = self._request_id
            payload = json.dumps(
                {
                    "id": request_id,
                    "text": text,
                    "hypotheses": list(hypotheses),
                    "language": language,
                },
                ensure_ascii=True,
                separators=(",", ":"),
            ).encode("utf-8")
            if len(payload) > 65_536:
                raise LocalIntelligenceError("semantic_worker_request_too_large")
            try:
                process.stdin.write(payload + b"\n")
                await process.stdin.drain()
                raw = await asyncio.wait_for(
                    process.stdout.readline(),
                    timeout=self.request_timeout_s,
                )
                response = json.loads(raw)
                scores = response["scores"]
                if response.get("id") != request_id or len(scores) != len(hypotheses):
                    raise ValueError("semantic worker response mismatch")
                values = tuple(float(score) for score in scores)
                if any(not 0 <= score <= 1 for score in values):
                    raise ValueError("semantic worker score is outside zero and one")
                return values
            except TimeoutError as error:
                raise LocalIntelligenceError("semantic_worker_timeout") from error
            except (BrokenPipeError, OSError) as error:
                raise LocalIntelligenceError("semantic_worker_unavailable") from error
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
                raise LocalIntelligenceError("semantic_worker_response_invalid") from error

    async def aclose(self) -> None:
        process, self._process = self._process, None
        if process is None:
            return
        if process.returncode is None:
            with suppress(ProcessLookupError):
                process.terminate()
        try:
            await asyncio.wait_for(process.wait(), timeout=5.0)
        except TimeoutError:
            with suppress(ProcessLookupError):
                process.kill()
            await process.wait()

class CatalogSemanticPlanner:
    """Screen a single-topic small-model plan against dynamic typed catalog entries."""

    def __init__(
        self,
        scorer: SemanticScorer,
        *,
        planner_id: str,
        acceptance_threshold: float = 0.0,
    ) -> None:
        if not planner_id:
            raise ValueError("semantic planner ID is required")
        if not 0 <= acceptance_threshold <= 1:
            raise ValueError("semantic planner threshold must be between zero and one")
        self._scorer = scorer
        self._planner_id = planner_id
        self._acceptance_threshold = acceptance_threshold

    @staticmethod
    def _signal_hypothesis(signal: SignalDescriptor) -> str:
        selector = signal.selector
        subject = (
            selector.subject_id
            if selector.source == "opponent"
            else selector.source
        )
        name = selector.signal.replace("_", " ")
        return f"The driver asks for the current {subject} telemetry value: {name}."

    @staticmethod
    def _signal_priority(signal: SignalDescriptor) -> tuple[int, str, str]:
        order = {"player": 0, "context": 1, "field": 2, "opponent": 3}
        selector = signal.selector
        return (
            order[selector.source],
            selector.subject_id or "",
            selector.signal,
        )

    async def plan(
        self,
        turn: DriverTurn,
        signals: Sequence[SignalDescriptor],
        capabilities: Sequence[CapabilityDescriptor],
    ) -> ContextPlan:
        capability_options = tuple(capabilities[: _MAX_HYPOTHESES - 1])
        remaining = _MAX_HYPOTHESES - len(capability_options) - 1
        signal_options = tuple(sorted(signals, key=self._signal_priority)[:remaining])
        hypotheses = tuple(
            f"The driver asks for this race-engineering calculation: {item.description}"
            for item in capability_options
        ) + tuple(self._signal_hypothesis(item) for item in signal_options) + (
            "The driver makes a social or emotional remark that needs no telemetry evidence.",
        )
        scores = await self._scorer.score(
            turn.transcript,
            hypotheses,
            language=turn.asr_language or turn.reply_language,
        )
        best = max(range(len(scores)), key=scores.__getitem__)
        if scores[best] < self._acceptance_threshold:
            return ContextPlan(
                turn_id=turn.turn_id,
                planner_id=self._planner_id,
                unknowns=("semantic_candidate_abstained",),
            )
        if best < len(capability_options):
            capability = capability_options[best]
            return ContextPlan(
                turn_id=turn.turn_id,
                planner_id=self._planner_id,
                temporal_scope=capability.temporal_scope,
                capability_requests=(
                    CapabilityRequest(
                        request_id="c1",
                        capability_id=capability.capability_id,
                    ),
                ),
            )
        signal_index = best - len(capability_options)
        if signal_index < len(signal_options):
            signal = signal_options[signal_index]
            return ContextPlan(
                turn_id=turn.turn_id,
                planner_id=self._planner_id,
                queries=(
                    EvidenceQuery(
                        query_id="e1",
                        selector=signal.selector,
                        operation=(
                            "maximum" if signal.selector.source == "field" else "latest"
                        ),
                    ),
                ),
            )
        return ContextPlan(
            turn_id=turn.turn_id,
            planner_id=self._planner_id,
            temporal_scope="social",
        )

    async def aclose(self) -> None:
        close = getattr(self._scorer, "aclose", None)
        if close is not None:
            await close()

    @property
    def runtime_metadata(self) -> dict[str, object] | None:
        value = getattr(self._scorer, "runtime_metadata", None)
        return dict(value) if isinstance(value, dict) else None
