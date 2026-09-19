"""Reconnect-safe async telemetry adapter for the local iRacing simulator."""

import asyncio
import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from math import floor
from time import perf_counter

from pydantic import ValidationError

from race_engineer.config import IracingTelemetryConfig
from race_engineer.core.contracts import TelemetryFrame
from race_engineer.telemetry.iracing.normalizer import normalize_sample
from race_engineer.telemetry.iracing.source import (
    IracingReadMetrics,
    IracingSource,
    MonotonicClock,
    source_factory,
)

SourceFactory = Callable[[], IracingSource]
Sleeper = Callable[[float], Awaitable[None]]


class IracingTelemetryAdapter:
    def __init__(
        self,
        config: IracingTelemetryConfig,
        source_provider: SourceFactory = source_factory,
        sleeper: Sleeper = asyncio.sleep,
        monotonic: MonotonicClock = perf_counter,
    ) -> None:
        self._config = config
        self._source_provider = source_provider
        self._sleep = sleeper
        self._monotonic = monotonic
        self._last_session_id: str | None = None
        self._last_sequence: int | None = None
        self._last_accepted_frame: TelemetryFrame | None = None
        self._last_accepted_started_at: float | None = None
        self._last_accepted_metrics: IracingReadMetrics | None = None
        self._logger = logging.getLogger(__name__)

    def _accept_order(self, frame: TelemetryFrame) -> bool:
        if frame.session_id != self._last_session_id:
            self._logger.info(
                "iRacing session changed",
                extra={"event": "session_changed", "session_id": frame.session_id},
            )
            self._last_session_id = frame.session_id
            self._last_sequence = None
        if self._last_sequence is not None and frame.sequence <= self._last_sequence:
            self._logger.debug(
                "stale iRacing frame dropped",
                extra={
                    "event": "frame_dropped",
                    "reason": "stale_or_duplicate",
                    "session_id": frame.session_id,
                    "sequence": frame.sequence,
                },
            )
            return False
        self._last_sequence = frame.sequence
        return True

    async def _wait_for_sample_slot(self, target: float, interval_s: float) -> float:
        now = self._monotonic()
        if now < target:
            await self._sleep(target - now)
            now = self._monotonic()

        lateness_s = max(0.0, now - target)
        missed_deadlines = floor(lateness_s / interval_s)
        if missed_deadlines:
            self._logger.warning(
                "iRacing sampling deadline missed",
                extra={
                    "event": "sampling_deadline_missed",
                    "late_by_ms": round(lateness_s * 1000, 3),
                    "missed_deadlines": missed_deadlines,
                },
            )
            target += missed_deadlines * interval_s
        return target

    def _log_read_metrics(
        self,
        metrics: IracingReadMetrics,
        normalization_ms: float,
    ) -> None:
        extra = {
            "event": "telemetry_read_timing",
            "buffer_wait_ms": round(metrics.buffer_wait_ms, 3),
            "variable_read_ms": round(metrics.variable_read_ms, 3),
            "metadata_refresh_ms": round(metrics.metadata_refresh_ms, 3),
            "normalization_ms": round(normalization_ms, 3),
            "total_read_ms": round(metrics.total_ms, 3),
            "metadata_refreshed": metrics.metadata_refreshed,
        }
        if metrics.total_ms >= self._config.slow_read_warning_s * 1000:
            self._logger.warning("slow iRacing telemetry read", extra=extra)
        elif metrics.metadata_refreshed:
            self._logger.info("iRacing session metadata refreshed", extra=extra)
        else:
            self._logger.debug("iRacing telemetry read timing", extra=extra)

    def _observe_sample_gap(
        self,
        frame: TelemetryFrame,
        sample_started_at: float,
        metrics: IracingReadMetrics,
    ) -> None:
        previous_frame = self._last_accepted_frame
        previous_started_at = self._last_accepted_started_at
        previous_metrics = self._last_accepted_metrics
        if (
            previous_frame is not None
            and previous_started_at is not None
            and previous_frame.session_id == frame.session_id
        ):
            gap_s = sample_started_at - previous_started_at
            if gap_s >= self._config.sample_gap_warning_s:
                self._logger.warning(
                    "iRacing telemetry sample gap",
                    extra={
                        "event": "telemetry_sample_gap",
                        "session_id": frame.session_id,
                        "gap_ms": round(gap_s * 1000, 3),
                        "sequence_delta": frame.sequence - previous_frame.sequence,
                        "previous_metadata_refreshed": (
                            previous_metrics.metadata_refreshed
                            if previous_metrics is not None
                            else False
                        ),
                        "previous_metadata_refresh_ms": (
                            round(previous_metrics.metadata_refresh_ms, 3)
                            if previous_metrics is not None
                            else 0.0
                        ),
                    },
                )
        self._last_accepted_frame = frame
        self._last_accepted_started_at = sample_started_at
        self._last_accepted_metrics = metrics

    async def stream(self) -> AsyncIterator[TelemetryFrame]:
        interval_s = 1.0 / self._config.sample_rate_hz
        while True:
            source = self._source_provider()
            try:
                if not source.connect():
                    self._logger.info(
                        "waiting for iRacing",
                        extra={"event": "connection_wait", "reason": "simulator_unavailable"},
                    )
                    await self._sleep(self._config.reconnect_delay_s)
                    continue

                self._logger.info("connected to iRacing", extra={"event": "connected"})
                next_sample_at = self._monotonic()
                while source.connected:
                    sample_slot = await self._wait_for_sample_slot(next_sample_at, interval_s)
                    sample_started_at = self._monotonic()
                    try:
                        read_result = source.read()
                        normalization_started_at = self._monotonic()
                        frame = normalize_sample(read_result.sample)
                        normalization_ms = (self._monotonic() - normalization_started_at) * 1000
                    except (OSError, RuntimeError, ValidationError, ValueError) as error:
                        self._logger.warning(
                            "iRacing sample failed",
                            extra={
                                "event": "sample_failed",
                                "reason": type(error).__name__,
                            },
                        )
                        break
                    finally:
                        next_sample_at = sample_slot + interval_s

                    self._log_read_metrics(read_result.metrics, normalization_ms)

                    if frame is None:
                        self._logger.debug(
                            "incomplete iRacing sample dropped",
                            extra={"event": "frame_dropped", "reason": "incomplete_identity"},
                        )
                    elif frame.is_replay and not self._config.include_replay:
                        self._logger.debug(
                            "iRacing replay frame dropped",
                            extra={"event": "frame_dropped", "reason": "replay_disabled"},
                        )
                    elif self._accept_order(frame):
                        self._observe_sample_gap(
                            frame,
                            sample_started_at,
                            read_result.metrics,
                        )
                        yield frame

                self._logger.info("iRacing disconnected", extra={"event": "disconnected"})
            finally:
                source.close()

            await self._sleep(self._config.reconnect_delay_s)
