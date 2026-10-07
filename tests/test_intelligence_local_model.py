import asyncio
import json
import time

import pytest

from race_engineer.config import ConversationConfig
from race_engineer.intelligence.local_model import LocalIntelligenceError, LocalJsonModel


def payload(content: str = "{}", finish_reason: str = "stop") -> bytes:
    return json.dumps(
        {
            "choices": [
                {
                    "finish_reason": finish_reason,
                    "message": {"content": content},
                }
            ]
        }
    ).encode()


class FakeConnection:
    def __init__(self, *, raw: bytes | None = None, status: int = 200, error=None) -> None:
        self.raw = payload() if raw is None else raw
        self.status = status
        self.error = error
        self.closed = False
        self.requests: list[tuple[object, ...]] = []

    def request(self, method, path, body, headers):
        self.requests.append((method, path, json.loads(body), headers))
        if self.error is not None:
            raise self.error

    def getresponse(self):
        return self

    def read(self, limit):
        return self.raw[:limit]

    def close(self):
        self.closed = True


def install(monkeypatch, connection: FakeConnection) -> None:
    def factory(host, port, timeout):
        assert host == "127.0.0.1"
        assert port == 8087
        assert timeout == 30
        return connection

    monkeypatch.setattr(
        "race_engineer.intelligence.local_model.http.client.HTTPConnection",
        factory,
    )


def request(model: LocalJsonModel):
    return asyncio.run(
        model.request(
            system_prompt="system",
            content='{"turn":"test"}',
            schema={
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        )
    )


def test_local_json_model_is_loopback_only_and_schema_constrained(monkeypatch):
    connection = FakeConnection(raw=payload('{"result":"ok"}'))
    install(monkeypatch, connection)
    monkeypatch.setenv("HTTP_PROXY", "http://example.com:8888")

    result = request(LocalJsonModel(ConversationConfig()))

    assert result == {"result": "ok"}
    method, path, body, headers = connection.requests[0]
    assert method == "POST" and path == "/v1/chat/completions"
    assert body["response_format"]["schema"]["additionalProperties"] is False
    assert body["messages"][0]["content"] == "system"
    assert body["stream"] is False
    assert "Authorization" not in headers
    assert connection.closed


@pytest.mark.parametrize(
    ("connection", "reason"),
    [
        (FakeConnection(status=302), "model_http_302"),
        (FakeConnection(error=ConnectionRefusedError()), "model_unreachable"),
        (FakeConnection(error=TimeoutError()), "model_timeout"),
        (FakeConnection(raw=b"not-json"), "model_response_invalid"),
        (FakeConnection(raw=b"{}"), "model_response_invalid"),
        (FakeConnection(raw=b"x" * 70_000), "model_response_too_large"),
        (
            FakeConnection(raw=payload(finish_reason="length")),
            "model_response_incomplete",
        ),
        (
            FakeConnection(raw=payload('{"one":1,"one":2}')),
            "model_response_invalid",
        ),
    ],
)
def test_local_json_model_sanitizes_transport_and_response_failures(
    monkeypatch,
    connection,
    reason,
):
    install(monkeypatch, connection)
    with pytest.raises(LocalIntelligenceError, match=reason):
        request(LocalJsonModel(ConversationConfig()))
    assert connection.closed


def test_model_metrics_are_numeric_only_and_optional(monkeypatch):
    body = json.loads(payload('{"result":"private model content"}'))
    body["usage"] = {
        "prompt_tokens": 1400,
        "completion_tokens": 40,
        "prompt_tokens_details": {"cached_tokens": 1380},
    }
    body["timings"] = {"prompt_ms": 123.5, "predicted_ms": 2500, "private": "secret"}
    connection = FakeConnection(raw=json.dumps(body).encode())
    install(monkeypatch, connection)
    observed = []
    result = request(LocalJsonModel(ConversationConfig(), on_metrics=observed.append))
    assert result == {"result": "private model content"}
    assert observed == [
        {
            "prompt_ms": 123.5,
            "generation_ms": 2500,
            "prompt_tokens": 1400,
            "cached_prompt_tokens": 1380,
            "completion_tokens": 40,
        }
    ]
    assert "private" not in json.dumps(observed)


@pytest.mark.parametrize("invalid_timing", [float("nan"), float("inf"), 10**400])
def test_invalid_model_timing_fields_are_ignored(monkeypatch, invalid_timing):
    body = json.loads(payload())
    body["usage"] = {"prompt_tokens": True, "completion_tokens": "private"}
    body["timings"] = {"prompt_ms": -1, "predicted_ms": invalid_timing}
    install(monkeypatch, FakeConnection(raw=json.dumps(body).encode()))
    observed = []
    request(LocalJsonModel(ConversationConfig(), on_metrics=observed.append))
    assert all(value is None for value in observed[0].values())


def test_timed_out_worker_cannot_publish_late_metrics(monkeypatch):
    observed = []
    model = LocalJsonModel(
        ConversationConfig().model_copy(update={"timeout_s": 0.01}), on_metrics=observed.append
    )

    def slow_request(**kwargs):
        time.sleep(0.05)
        return {}, {"prompt_ms": 1}

    monkeypatch.setattr(model, "_request", slow_request)

    async def run():
        with pytest.raises(LocalIntelligenceError, match="model_timeout"):
            await model.request(system_prompt="", content="", schema={})
        await asyncio.sleep(0.1)
        assert observed == []

    asyncio.run(run())
