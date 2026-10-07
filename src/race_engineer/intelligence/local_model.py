"""Reusable loopback-only JSON client for local intelligence adapters."""

import asyncio
import http.client
import json
import math
from collections.abc import Callable
from typing import Protocol

from race_engineer.config import ConversationConfig


class LocalIntelligenceError(Exception):
    """Sanitized local-model failure; prompts and provider bodies are never exposed."""


class JsonModelClient(Protocol):
    """Replaceable schema-constrained local inference boundary."""

    async def request(
        self,
        *,
        system_prompt: str,
        content: str,
        schema: dict[str, object],
        max_tokens: int = 1024,
    ) -> object: ...


type ModelRequestMetrics = dict[str, float | int | None]


def _request_metrics(payload: dict[str, object]) -> ModelRequestMetrics:
    """Whitelist optional server timings; never copy response text into diagnostics."""

    def number(container: object, key: str) -> float | int | None:
        value = container.get(key) if isinstance(container, dict) else None
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        return value if 0 <= value <= 1e12 and math.isfinite(value) else None

    usage = payload.get("usage")
    details = usage.get("prompt_tokens_details") if isinstance(usage, dict) else None
    timing = payload.get("timings")
    return {
        "prompt_ms": number(timing, "prompt_ms"),
        "generation_ms": number(timing, "predicted_ms"),
        "prompt_tokens": number(usage, "prompt_tokens"),
        "cached_prompt_tokens": number(details, "cached_tokens"),
        "completion_tokens": number(usage, "completion_tokens"),
    }


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


class LocalJsonModel:
    """Issue one schema-constrained request to a literal loopback llama.cpp server."""

    def __init__(
        self,
        config: ConversationConfig,
        *,
        on_metrics: Callable[[ModelRequestMetrics], None] | None = None,
    ) -> None:
        self._config = config
        self._on_metrics = on_metrics

    def _request(
        self,
        *,
        system_prompt: str,
        content: str,
        schema: dict[str, object],
        max_tokens: int,
    ) -> tuple[object, ModelRequestMetrics]:
        body = json.dumps(
            {
                "model": self._config.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": content},
                ],
                "temperature": 0,
                "max_tokens": max_tokens,
                "stream": False,
                "response_format": {
                    "type": "json_object",
                    "schema": schema,
                },
            },
            ensure_ascii=False,
        ).encode("utf-8")
        if len(body) > 65_536:
            raise LocalIntelligenceError("model_request_too_large")
        connection = http.client.HTTPConnection(
            "127.0.0.1",
            self._config.port,
            timeout=self._config.timeout_s,
        )
        try:
            connection.request(
                "POST",
                "/v1/chat/completions",
                body,
                {"Content-Type": "application/json"},
            )
            response = connection.getresponse()
            if response.status != 200:
                raise LocalIntelligenceError(f"model_http_{response.status}")
            raw = response.read(65_537)
            if len(raw) > 65_536:
                raise LocalIntelligenceError("model_response_too_large")
            payload = json.loads(raw, object_pairs_hook=_unique_object)
            choice = payload["choices"][0]
            if choice.get("finish_reason") != "stop":
                raise LocalIntelligenceError("model_response_incomplete")
            content = json.loads(choice["message"]["content"], object_pairs_hook=_unique_object)
            return content, _request_metrics(payload)
        except TimeoutError as error:
            raise LocalIntelligenceError("model_timeout") from error
        except (OSError, http.client.HTTPException) as error:
            raise LocalIntelligenceError("model_unreachable") from error
        except (ValueError, TypeError, KeyError, IndexError, AttributeError) as error:
            raise LocalIntelligenceError("model_response_invalid") from error
        finally:
            connection.close()

    async def request(
        self,
        *,
        system_prompt: str,
        content: str,
        schema: dict[str, object],
        max_tokens: int = 1024,
    ) -> object:
        try:
            result, metrics = await asyncio.wait_for(
                asyncio.to_thread(
                    self._request,
                    system_prompt=system_prompt,
                    content=content,
                    schema=schema,
                    max_tokens=max_tokens,
                ),
                timeout=self._config.timeout_s,
            )
        except TimeoutError as error:
            raise LocalIntelligenceError("model_timeout") from error
        # Notify only after this awaited request completes. A timed-out background thread
        # must not attach late metrics to the next evaluation turn.
        if self._on_metrics is not None:
            self._on_metrics(metrics)
        return result
