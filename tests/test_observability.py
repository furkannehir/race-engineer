import json
import logging

from race_engineer.observability import JsonFormatter


def test_json_logs_include_trace_fields() -> None:
    record = logging.LogRecord(
        name="race_engineer.policy",
        level=logging.INFO,
        pathname=__file__,
        lineno=10,
        msg="candidate suppressed",
        args=(),
        exc_info=None,
    )
    record.__dict__["session_id"] = "session-1"
    record.__dict__["reason"] = "cooldown"
    payload = json.loads(JsonFormatter().format(record))
    assert payload["message"] == "candidate suppressed"
    assert payload["session_id"] == "session-1"
    assert payload["reason"] == "cooldown"
