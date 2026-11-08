"""Bounded loopback client for native model tool calls, separate from JSON drafts."""

import asyncio
import http.client
import json
import time
from collections.abc import Callable
from typing import Any, Protocol

from pydantic import Field, JsonValue

from race_engineer.config import ConversationConfig
from race_engineer.core.contracts import ContractModel
from race_engineer.intelligence.local_model import (
    LocalIntelligenceError,
    ModelRequestMetrics,
    _request_metrics,
    _unique_object,
)
from race_engineer.radio_diagnostics import error_code, model_stage, radio_event


class NativeToolCall(ContractModel):
    call_id: str = Field(min_length=1, max_length=160)
    name: str = Field(pattern=r"^[A-Za-z_][A-Za-z0-9_]*$", max_length=100)
    arguments: dict[str, JsonValue] = Field(max_length=16)


class NativeToolReply(ContractModel):
    calls: tuple[NativeToolCall, ...] = Field(max_length=32)


class NativeToolModel(Protocol):
    async def request_tools(
        self,
        *,
        messages: list[dict[str, object]],
        tools: list[dict[str, object]],
        max_tokens: int = 768,
    ) -> NativeToolReply: ...


class LocalNativeModel:
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
        messages: list[dict[str, object]],
        tools: list[dict[str, object]],
        max_tokens: int,
    ) -> tuple[NativeToolReply, ModelRequestMetrics]:
        body = json.dumps(
            {
                "model": self._config.model,
                "messages": messages,
                "tools": tools,
                "tool_choice": "required",
                "parallel_tool_calls": True,
                "temperature": 0,
                "max_tokens": max_tokens,
                "stream": False,
            },
            ensure_ascii=False,
        ).encode("utf-8")
        if len(body) > 65_536:
            raise LocalIntelligenceError("model_request_too_large")
        connection = http.client.HTTPConnection(
            "127.0.0.1", self._config.port, timeout=self._config.timeout_s
        )
        try:
            connection.request(
                "POST", "/v1/chat/completions", body, {"Content-Type": "application/json"}
            )
            response = connection.getresponse()
            if response.status != 200:
                raise LocalIntelligenceError(f"model_http_{response.status}")
            raw = response.read(65_537)
            if len(raw) > 65_536:
                raise LocalIntelligenceError("model_response_too_large")
            payload = json.loads(raw, object_pairs_hook=_unique_object)
            choice = payload["choices"][0]
            if choice.get("finish_reason") not in {"stop", "tool_calls"}:
                raise LocalIntelligenceError("model_response_incomplete")
            entries = choice["message"].get("tool_calls")
            if not isinstance(entries, list) or not 1 <= len(entries) <= 32:
                raise LocalIntelligenceError("model_native_calls_missing")
            calls = []
            for entry in entries:
                if entry.get("type") != "function":
                    raise ValueError("unsupported tool type")
                function = entry["function"]
                arguments = function["arguments"]
                if isinstance(arguments, str):
                    arguments = json.loads(arguments, object_pairs_hook=_unique_object)
                calls.append(
                    NativeToolCall(call_id=entry["id"], name=function["name"], arguments=arguments)
                )
            if len({call.call_id for call in calls}) != len(calls):
                raise ValueError("duplicate tool call ID")
            return NativeToolReply(calls=tuple(calls)), _request_metrics(payload)
        except TimeoutError as error:
            raise LocalIntelligenceError("model_timeout") from error
        except (OSError, http.client.HTTPException) as error:
            raise LocalIntelligenceError("model_unreachable") from error
        except (ValueError, TypeError, KeyError, IndexError, AttributeError) as error:
            raise LocalIntelligenceError("model_response_invalid") from error
        finally:
            connection.close()

    async def request_tools(
        self,
        *,
        messages: list[dict[str, object]],
        tools: list[dict[str, object]],
        max_tokens: int = 768,
    ) -> NativeToolReply:
        started = time.perf_counter()
        stage = model_stage()
        radio_event(
            "radio_model_started",
            stage=stage,
            max_tokens=max_tokens,
            timeout_s=self._config.timeout_s,
            interface="native_tools",
        )
        try:
            result, metrics = await asyncio.wait_for(
                asyncio.to_thread(self._request, messages, tools, max_tokens),
                timeout=self._config.timeout_s,
            )
        except BaseException as error:
            reason = "model_timeout" if isinstance(error, TimeoutError) else error_code(error)
            radio_event(
                "radio_model_finished",
                stage=stage,
                outcome="cancelled" if isinstance(error, asyncio.CancelledError) else "failed",
                reason=reason,
                duration_ms=round((time.perf_counter() - started) * 1000, 3),
            )
            if isinstance(error, TimeoutError):
                raise LocalIntelligenceError("model_timeout") from error
            raise
        if self._on_metrics is not None:
            self._on_metrics(metrics)
        radio_event(
            "radio_model_finished",
            stage=stage,
            outcome="completed",
            **metrics,
            duration_ms=round((time.perf_counter() - started) * 1000, 3),
        )
        return result


def function_tool(
    name: str,
    description: str,
    properties: dict[str, Any],
    required: list[str],
    *,
    definitions: dict[str, Any] | None = None,
) -> dict[str, object]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
                "additionalProperties": False,
                **({"$defs": definitions} if definitions else {}),
            },
        },
    }
