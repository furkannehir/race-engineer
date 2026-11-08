import asyncio
import json
import threading

import pytest
from test_intelligence_local_model import FakeConnection, install

from race_engineer.config import ConversationConfig
from race_engineer.intelligence.local_model import LocalIntelligenceError
from race_engineer.intelligence.native_model import (
    LocalNativeModel,
    NativeToolReply,
)


def response(arguments="{}", *, name="get_current_race_position", finish="tool_calls"):
    return json.dumps(
        {
            "choices": [
                {
                    "finish_reason": finish,
                    "message": {
                        "content": "untrusted text is not executable",
                        "tool_calls": [
                            {
                                "id": "call-one",
                                "type": "function",
                                "function": {"name": name, "arguments": arguments},
                            }
                        ],
                    },
                }
            ]
        }
    ).encode()


def request(client):
    return asyncio.run(
        client.request_tools(
            messages=[{"role": "user", "content": "Position?"}], tools=[{"type": "function"}]
        )
    )


@pytest.mark.parametrize("arguments", ["{}", {}])
def test_native_transport_uses_loopback_tool_api_and_parses_arguments(monkeypatch, arguments):
    connection = FakeConnection(raw=response(arguments))
    install(monkeypatch, connection)
    monkeypatch.setenv("HTTP_PROXY", "http://example.invalid")
    actual = request(LocalNativeModel(ConversationConfig()))
    assert actual.calls[0].name == "get_current_race_position"
    assert actual.calls[0].arguments == {}
    _, path, body, headers = connection.requests[0]
    assert path == "/v1/chat/completions" and body["tool_choice"] == "required"
    assert "response_format" not in body and body["parallel_tool_calls"]
    assert "Authorization" not in headers and connection.closed


@pytest.mark.parametrize(
    "raw,reason",
    [
        (response('{"x":1,"x":2}'), "model_response_invalid"),
        (response("[]"), "model_response_invalid"),
        (response(finish="length"), "model_response_incomplete"),
        (
            b'{"choices":[{"finish_reason":"stop","message":{"content":"P5"}}]}',
            "model_native_calls_missing",
        ),
        (b"not-json", "model_response_invalid"),
        (b"x" * 70000, "model_response_too_large"),
    ],
    ids=["duplicate-key", "array-arguments", "truncated", "no-calls", "bad-json", "oversize"],
)
def test_invalid_native_responses_are_rejected_not_treated_as_spoken_answers(
    monkeypatch, raw, reason
):
    connection = FakeConnection(raw=raw)
    install(monkeypatch, connection)
    with pytest.raises(LocalIntelligenceError, match=reason):
        request(LocalNativeModel(ConversationConfig()))
    assert connection.closed


@pytest.mark.parametrize("cancel", [False, True], ids=["timeout", "cancel"])
def test_abandoned_native_request_cannot_publish_late_metrics(monkeypatch, cancel):
    observed = []
    started, release = threading.Event(), threading.Event()
    client = LocalNativeModel(
        ConversationConfig().model_copy(update={"timeout_s": 0.02}),
        on_metrics=observed.append,
    )

    def slow_request(*_):
        started.set()
        release.wait(2)
        return NativeToolReply(calls=()), {"prompt_ms": 1}

    monkeypatch.setattr(client, "_request", slow_request)

    async def run():
        try:
            task = asyncio.create_task(client.request_tools(messages=[], tools=[]))
            while not started.is_set():
                await asyncio.sleep(0.001)
            if cancel:
                task.cancel()
            with pytest.raises(asyncio.CancelledError if cancel else LocalIntelligenceError):
                await task
        finally:
            release.set()
        await asyncio.sleep(0.01)
        assert observed == []

    asyncio.run(run())


@pytest.mark.parametrize(
    "connection,reason",
    [
        (FakeConnection(status=500), "model_http_500"),
        (FakeConnection(error=OSError("private details")), "model_unreachable"),
        (FakeConnection(error=TimeoutError()), "model_timeout"),
    ],
)
def test_native_transport_failures_close_connection_and_report_sanitized_reason(
    monkeypatch, connection, reason
):
    install(monkeypatch, connection)
    with pytest.raises(LocalIntelligenceError, match=reason):
        request(LocalNativeModel(ConversationConfig()))
    assert connection.closed
