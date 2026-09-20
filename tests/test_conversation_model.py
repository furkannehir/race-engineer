import asyncio
import json

import pytest

from race_engineer.config import ConversationConfig
from race_engineer.conversation.local_model import ConversationModelError, LocalQwenPlanner
from race_engineer.core.conversation import ConversationRequest, RaceQuery


def payload(content=None, finish_reason="stop"):
    return json.dumps(
        {
            "choices": [
                {
                    "finish_reason": finish_reason,
                    "message": {
                        "content": content
                        or json.dumps(
                            {"language": "tr", "queries": ["position"], "clarification": "none"}
                        )
                    },
                }
            ]
        }
    ).encode()


class FakeConnection:
    def __init__(self, *, raw=None, status=200, error=None):
        self.raw = payload() if raw is None else raw
        self.status = status
        self.error = error
        self.closed = False
        self.requests = []

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


def install(monkeypatch, connection):
    def factory(host, port, timeout):
        assert host == "127.0.0.1"
        assert port == 8087
        assert timeout == 30
        return connection

    monkeypatch.setattr(
        "race_engineer.conversation.local_model.http.client.HTTPConnection", factory
    )


def test_local_adapter_sends_schema_and_does_not_use_proxies(monkeypatch):
    connection = FakeConnection()
    install(monkeypatch, connection)
    monkeypatch.setenv("HTTP_PROXY", "http://example.com:8888")
    result = asyncio.run(
        LocalQwenPlanner(ConversationConfig()).plan(ConversationRequest(question="Kaçıncıyız?"))
    )
    assert result.language == "tr"
    assert result.queries == (RaceQuery.POSITION,)
    method, path, body, headers = connection.requests[0]
    assert method == "POST"
    assert path == "/v1/chat/completions"
    assert body["model"] == "Qwen3-4B-Instruct-2507"
    branches = body["response_format"]["schema"]["oneOf"]
    assert all(branch["additionalProperties"] is False for branch in branches)
    assert branches[0]["properties"]["queries"]["minItems"] == 1
    assert branches[1]["properties"]["queries"]["maxItems"] == 0
    assert body["stream"] is False
    assert "Authorization" not in headers
    assert connection.closed


@pytest.mark.parametrize(
    ("connection", "reason"),
    [
        (FakeConnection(status=302), "model_http_error"),
        (FakeConnection(status=500), "model_http_error"),
        (FakeConnection(error=ConnectionRefusedError()), "model_unreachable"),
        (FakeConnection(error=TimeoutError()), "model_timeout"),
        (FakeConnection(raw=b"not json"), "model_response_invalid"),
        (FakeConnection(raw=b"{}"), "model_response_invalid"),
        (FakeConnection(raw=b"x" * 70_000), "model_response_too_large"),
        (FakeConnection(raw=payload(finish_reason="length")), "model_response_incomplete"),
        (FakeConnection(raw=payload('"not a plan"')), "model_response_invalid"),
        (
            FakeConnection(raw=payload('{"language":"en","language":"tr"}')),
            "model_response_invalid",
        ),
        (
            FakeConnection(
                raw=payload(
                    '{"language":"en","queries":["position"],"clarification":"none","text":"P1"}'
                )
            ),
            "model_response_invalid",
        ),
    ],
)
def test_local_adapter_rejects_failures_and_malformed_output(monkeypatch, connection, reason):
    install(monkeypatch, connection)
    with pytest.raises(ConversationModelError, match=reason):
        asyncio.run(
            LocalQwenPlanner(ConversationConfig()).plan(
                ConversationRequest(question="Where are we?")
            )
        )
    assert connection.closed
