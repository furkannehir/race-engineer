"""Correlated local radio diagnostics, with explicit text and telemetry privacy gates."""

import asyncio
import logging
import re
import time
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass, field

from race_engineer.core.intelligence import ContextPacket

_LOGGER = logging.getLogger(__name__)
_TRACE: ContextVar["RadioTrace | None"] = ContextVar("radio_trace", default=None)
_STAGE: ContextVar[str | None] = ContextVar("radio_stage", default=None)


def error_code(error: BaseException) -> str:
    reason = str(error)
    return reason if re.fullmatch(r"[a-z][a-z0-9_]{0,159}", reason) else type(error).__name__


@dataclass
class RadioTrace:
    run_id: str
    trace_id: str
    record_text: bool = False
    record_values: bool = False
    scope: dict[str, object] = field(default_factory=dict)
    started: float = field(default_factory=time.perf_counter)
    finished: bool = False

    @contextmanager
    def bind(self) -> Iterator[None]:
        token = self.activate()
        try:
            yield
        finally:
            self.deactivate(token)

    def activate(self) -> Token["RadioTrace | None"]:
        return _TRACE.set(self)

    @staticmethod
    def deactivate(token: Token["RadioTrace | None"]) -> None:
        _TRACE.reset(token)

    def event(self, event: str, **fields: object) -> None:
        _LOGGER.info(
            event,
            extra={
                **self.scope,
                **fields,
                "event": event,
                "run_id": self.run_id,
                "trace_id": self.trace_id,
                "elapsed_ms": round((time.perf_counter() - self.started) * 1000, 3),
            },
        )

    def text(self, event: str, text: str | None, **fields: object) -> None:
        if self.record_text:
            fields["text"] = text[:1500] if text is not None else None
        self.event(event, text_recorded=self.record_text, **fields)

    def finish(self, outcome: str, **fields: object) -> None:
        if not self.finished:
            self.finished = True
            self.event("radio_turn_finished", outcome=outcome, **fields)


def radio_event(event: str, **fields: object) -> None:
    trace = _TRACE.get()
    if trace is not None:
        trace.event(event, **fields)


def radio_text(event: str, text: str | None, **fields: object) -> None:
    trace = _TRACE.get()
    if trace is not None:
        trace.text(event, text, **fields)


@contextmanager
def radio_stage(stage: str) -> Iterator[None]:
    started = time.perf_counter()
    token = _STAGE.set(stage)
    radio_event("radio_stage_started", stage=stage)
    try:
        yield
    except BaseException as error:
        radio_event(
            "radio_stage_finished",
            stage=stage,
            outcome="cancelled" if isinstance(error, asyncio.CancelledError) else "failed",
            reason=error_code(error),
            duration_ms=round((time.perf_counter() - started) * 1000, 3),
        )
        raise
    else:
        radio_event(
            "radio_stage_finished",
            stage=stage,
            outcome="completed",
            duration_ms=round((time.perf_counter() - started) * 1000, 3),
        )
    finally:
        _STAGE.reset(token)


def model_stage() -> str | None:
    return _STAGE.get()


def radio_capture_scope(generation: int) -> None:
    trace = _TRACE.get()
    if trace is not None:
        trace.started = time.perf_counter()
        trace.scope["generation"] = generation


def radio_evidence(event: str, packet: ContextPacket) -> None:
    trace = _TRACE.get()
    if trace is None:
        return
    evidence = []
    for item in packet.evidence:
        summary: dict[str, object] = {
            "id": item.evidence_id,
            "kind": item.kind,
            "subject": item.subject,
            "metric": item.metric,
            "unit": item.unit,
            "source_sequence": item.source_sequence,
            "observed_at": item.observed_at.isoformat(),
        }
        if trace.record_values:
            summary["value"] = item.value
        evidence.append(summary)
    trace.event(
        event,
        source_sequence=packet.source_sequence,
        situation=packet.situation,
        unknowns=packet.unknowns,
        evidence=evidence,
        values_recorded=trace.record_values,
    )
