import asyncio
import json
from pathlib import Path

import pytest
from test_lfm3_dialogue import Model, call, setup

from scripts.compare_dialogue_models import DATASET, RawConversation, configuration, summarize
from scripts.lfm3_dialogue import ChatReply, LocalDialogueModel


def test_application_rejection_does_not_erase_raw_reply_or_change_model_history(monkeypatch):
    executor, _, _ = setup()
    # Model evaluation must remain independent of diagnostic validator failures.
    def reject(*_args, **_kwargs):
        raise ValueError("diagnostic rejection")

    monkeypatch.setattr(executor, "_ground", reject)
    model = Model(
        ChatReply(None, (call(),)), ChatReply("You're P5, fifth place."),
        ChatReply("Yes, I'm here."),
    )
    chat = RawConversation(model, executor)

    async def run():
        first = await chat.respond("Position?", "en", "first")
        second = await chat.respond("Are you there?", "en", "second")
        return first, second

    first, second = asyncio.run(run())
    assert first["raw_reply"] == "You're P5, fifth place."
    assert first["application_validation"]["status"] == "rejected"
    assert len(first["requests"]) == 2 and "failure" not in first
    assert second["raw_reply"] == "Yes, I'm here."
    assert any(message.get("content") == first["raw_reply"] for message in model.requests[-1][0])


def test_unsupported_tool_cannot_conceal_bad_raw_recommendation():
    executor, _, _ = setup()
    model = Model(
        ChatReply(None, (call("report_unavailable_information", quantity="tyre choice"),)),
        ChatReply("Use the soft tyre."),
    )
    row = asyncio.run(RawConversation(model, executor).respond("Which tyre?", "en", "tyre"))
    assert row["raw_reply"] == "Use the soft tyre."
    assert "soft" not in row["application_validation"]["text"]


def test_turkish_is_not_sent_through_english_validation_or_greeting_rules():
    executor, _, _ = setup()
    model = Model(ChatReply("Buradayım."))
    row = asyncio.run(RawConversation(model, executor).respond("Orada mısın?", "tr", "tr"))
    assert row["application_validation"]["status"] == "not_applicable_turkish"
    assert "in Turkish." in model.requests[0][0][0]["content"]
    assert model.requests[0][0][-1]["content"] == "Orada mısın?"


def test_summary_does_not_call_grounded_or_completed_speech_accurate():
    rows = [{"language": "en", "duration_ms": 100, "raw_reply": "Wrong answer.",
             "tool_match": False, "requests": [{}, {}],
             "application_validation": {"status": "accepted"}}]
    result = summarize(rows)["en"]
    assert result["completed"] == 1 and result["tool_matches"] == 0
    assert result["automatic_speech_accuracy"] is None


def test_fixture_has_separate_bilingual_histories_and_identity_requires_missing_capability():
    dataset = json.loads(DATASET.read_text(encoding="utf-8"))
    sequences = dataset["sequences"]
    assert {item["language"] for item in sequences} == {"en", "tr"}
    identity = next(item for item in sequences if item["id"] == "reported-identity")
    assert identity["turns"][-1]["tool"] == "report_unavailable_information"
    assert sum(len(item["turns"]) for item in sequences) == 36


def test_comparison_profile_changes_no_default_file_and_never_falls_back():
    from scripts.compare_dialogue_models import CANDIDATES, ROOT

    default = ROOT / "config/default.toml"
    before = default.read_bytes()
    config, _ = configuration(CANDIDATES["qwen35"], "cpu", 18196)
    assert config.model == "Qwen/Qwen3.5-2B"
    assert config.runtime.mode == "cpu"
    assert not config.runtime.fallback_to_cpu and not config.runtime.reuse_external_server
    assert config.runtime.context_size == 8192 and config.runtime.enable_thinking is False
    assert default.read_bytes() == before
    assert isinstance(config.runtime.model_path, Path)


def test_generation_overrides_cannot_redirect_requests_or_force_a_tool():
    from race_engineer.config import ConversationConfig

    with pytest.raises(ValueError, match="generation parameter"):
        LocalDialogueModel(ConversationConfig(), generation={"tool_choice": "required"})


def test_timeout_stops_candidate_and_closes_owned_server(monkeypatch, tmp_path):
    from types import SimpleNamespace

    from scripts import compare_dialogue_models as comparison

    events = []

    class Server:
        def __init__(self, *_args, **_kwargs):
            self.process = SimpleNamespace(pid=123)
            self.effective = SimpleNamespace(backend="cpu")

        async def start(self):
            events.append("start")

        async def aclose(self):
            events.append("close")
            self.process = None

    class Dialogue:
        def __init__(self, *_args, **_kwargs):
            pass

        async def respond(self, *_args):
            events.append("request")
            return {
                "raw_reply": None, "language": "en", "selected_tools": [],
                "requests": [{}], "duration_ms": 100,
                "failure": "dialogue_timeout",
                "application_validation": {"status": "no_completed_reply"},
            }

    monkeypatch.setattr(comparison, "ConversationServer", Server)
    monkeypatch.setattr(comparison, "RawConversation", Dialogue)
    monkeypatch.setattr(comparison, "gpu_memory_snapshot", lambda: {})
    monkeypatch.setattr(comparison, "require_free_port", lambda _: None)
    monkeypatch.setattr(comparison, "file_sha256", lambda _: comparison.CANDIDATES["lfm"]["sha256"])
    dataset = json.loads(DATASET.read_text(encoding="utf-8"))
    args = SimpleNamespace(port=18196, repeats=1, cpu_short=True)
    report = asyncio.run(comparison.run_candidate("lfm", "cpu", args, dataset, tmp_path))
    assert events == ["start", "request", "close"]
    assert report["server_closed"] and len(report["rows"]) == 1
    assert report["failure"]
