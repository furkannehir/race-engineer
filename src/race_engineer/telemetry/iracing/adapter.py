"""Reconnect-safe async telemetry adapter for the local iRacing simulator."""

import asyncio
import logging
from collections.abc import AsyncIterator, Awaitable, Callable

from pydantic import ValidationError

from race_engineer.config import IracingTelemetryConfig
from race_engineer.core.contracts import TelemetryFrame
from race_engineer.telemetry.iracing.normalizer import normalize_sample
from race_engineer.telemetry.iracing.source import IracingSource, source_factory

SourceFactory = Callable[[], IracingSource]
Sleeper = Callable[[float], Awaitable[None]]


class IracingTelemetryAdapter:
    def __init__(
        self,
        config: IracingTelemetryConfig,
        source_provider: SourceFactory = source_factory,
        sleeper: Sleeper = asyncio.sleep,
    ) -> None:
        self._config = config
        self._source_provider = source_provider
        self._sleep = sleeper
        self._last_session_id: str | None = None
        self._last_sequence: int | None = None
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
                while source.connected:
                    try:
                        frame = normalize_sample(source.read())
                    except (OSError, RuntimeError, ValidationError, ValueError) as error:
                        self._logger.warning(
                            "iRacing sample failed",
                            extra={
                                "event": "sample_failed",
                                "reason": type(error).__name__,
                            },
                        )
                        break

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
                        yield frame
                    await self._sleep(interval_s)

                self._logger.info("iRacing disconnected", extra={"event": "disconnected"})
            finally:
                source.close()

            await self._sleep(self._config.reconnect_delay_s)
