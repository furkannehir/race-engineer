import asyncio
import json
import tomllib
from pathlib import Path
from types import SimpleNamespace

import pytest

from race_engineer.config import AppConfig, ConversationConfig
from race_engineer.conversation.runtime import RuntimeCandidate, server_command
from scripts import setup_lfm3_trial, try_lfm3
from scripts.lfm3_prompt import PROMPT, EnglishPlannerClient

ROOT = Path(__file__).parents[1]


@pytest.mark.parametrize("mode", ["cpu", "gpu"])
def test_lfm_profile_is_explicit_english_trial_and_preserves_speech_policy(mode):
    source = (ROOT / "config/default.toml").read_text(encoding="utf-8")
    original = AppConfig.model_validate(tomllib.loads(source))
    config = AppConfig.model_validate(tomllib.loads(setup_lfm3_trial.trial_profile(source, mode)))
    assert config.tts == original.tts and config.stt == original.stt
    assert config.policy == original.policy
    assert config.conversation.model == "LiquidAI/LFM3-1.7B"
    assert config.conversation.port == 8091
    assert config.conversation.default_language == "en"
    assert config.conversation.context_planner == "qwen-context-v8"
    runtime = config.conversation.runtime
    assert runtime.enable_thinking is False
    assert runtime.mode == mode and runtime.runtime_revision == "b12088"
    assert not runtime.reuse_external_server and not runtime.fallback_to_cpu
    assert runtime.context_size == 8192 and runtime.parallel == 1
    assert "lfm3-prototype" in str(runtime.model_path)
    assert "lfm3-prototype" in str(runtime.accelerated_executable_path)
    assert original.conversation.runtime.enable_thinking is False


def test_asset_only_setup_reuses_installers_without_writing_comparison_profiles(
    monkeypatch, tmp_path
):
    calls = []
    monkeypatch.setattr("sys.argv", ["setup_lfm3_trial.py", "--runtime-only", "--backend", "gpu"])
    monkeypatch.setattr(setup_lfm3_trial, "ASSETS", tmp_path)
    monkeypatch.setattr(setup_lfm3_trial, "install_runtime", lambda mode: calls.append(mode))
    monkeypatch.setattr(setup_lfm3_trial, "download", lambda *args: calls.append(args))
    monkeypatch.setattr(
        setup_lfm3_trial,
        "trial_profile",
        lambda *args, **kwargs: pytest.fail("Asset-only setup must not create profiles"),
    )
    before = (ROOT / "config/default.toml").read_bytes()
    setup_lfm3_trial.main()
    assert calls == ["gpu", (setup_lfm3_trial.MODEL, tmp_path)]
    assert not list(tmp_path.iterdir())
    assert (ROOT / "config/default.toml").read_bytes() == before


@pytest.mark.parametrize("thinking", [None, False, True])
def test_thinking_template_switch_is_only_forwarded_when_explicit(thinking):
    config = ConversationConfig()
    config = config.model_copy(
        update={"runtime": config.runtime.model_copy(update={"enable_thinking": thinking})}
    )
    candidate = RuntimeCandidate("gpu", Path("server.exe"), Path("model.gguf"), "cuda", "CUDA0", -1)
    command = server_command(candidate, config)
    if thinking is None:
        assert "--chat-template-kwargs" not in command
    else:
        value = json.loads(command[command.index("--chat-template-kwargs") + 1])
        assert value == {"enable_thinking": thinking}


def test_lfm_benchmark_uses_english_only_without_editing_dataset_expectations():
    config = AppConfig()
    short, inputs = try_lfm3.english_cases(config, False)
    full, _ = try_lfm3.english_cases(config, True)
    assert short and len(full) > len(short)
    assert all(case[2].language == "en" for case in full)
    assert len({case[2].id for case in short}) == len(short)
    assert all("fixtures" in path.parts for path in inputs)


def test_compact_prompt_retains_driver_content_schema_and_output_budget(monkeypatch):
    captured = []

    async def request(self, **kwargs):
        captured.append(kwargs)
        return {"social_kind": "radio_check"}

    monkeypatch.setattr("race_engineer.intelligence.local_model.LocalJsonModel.request", request)
    schema = {"type": "object"}
    content = '{"transcript":"Are you there?","recent_dialogue":[]}'
    result = asyncio.run(
        EnglishPlannerClient(ConversationConfig()).request(
            system_prompt="original v8", content=content, schema=schema, max_tokens=384
        )
    )
    assert result == {"social_kind": "radio_check"}
    assert captured == [
        {"system_prompt": PROMPT, "content": content, "schema": schema, "max_tokens": 384}
    ]


@pytest.mark.parametrize("single_failure", [False, True])
@pytest.mark.parametrize("native", [False, True])
def test_terminal_trial_reuses_server_and_retains_only_successful_history(
    monkeypatch, single_failure, native
):
    config = AppConfig()
    config = config.model_copy(
        update={
            "conversation": config.conversation.model_copy(
                update={
                    "model": "LiquidAI/LFM3-1.7B",
                    "context_planner": "lfm3-native-v1" if native else "qwen-context-v8",
                }
            )
        }
    )
    monkeypatch.setattr(try_lfm3, "model_config", lambda _: config)
    monkeypatch.setattr(try_lfm3, "require_free_port", lambda _: None)
    events, histories = [], []

    class Server:
        def __init__(self, *args):
            pass

        async def start(self):
            events.append("start")

        async def aclose(self):
            events.append("close")

    class Engineer:
        async def respond(self, turn):
            histories.append(turn.recent_dialogue)
            if single_failure or len(histories) == 2:
                raise RuntimeError("context_model_response_invalid")
            return SimpleNamespace(text="Copy.")

    monkeypatch.setattr(try_lfm3, "ConversationServer", Server)
    monkeypatch.setattr(try_lfm3, "live_intelligence", lambda *_: Engineer())
    questions = iter(("Are you there?", "A failed question", "Fuel?", "/quit"))
    monkeypatch.setattr("builtins.input", lambda _: next(questions))
    args = SimpleNamespace(
        config=Path("unused.toml"),
        compact_planner=False,
        fixture=ROOT / "fixtures/synthetic/intelligence-capabilities",
        question="Position?" if single_failure else None,
    )
    assert asyncio.run(try_lfm3.chat(args)) == (1 if single_failure else 0)
    assert events == ["start", "close"]
    if not single_failure:
        assert histories == [
            (),
            ("driver: Are you there?", "engineer: Copy."),
            ("driver: Are you there?", "engineer: Copy."),
        ]


@pytest.mark.parametrize("start_failure", [False, True])
def test_interactive_runtime_is_closed_even_after_startup_failure(monkeypatch, start_failure):
    events = []

    class Server:
        def __init__(self, *args):
            pass

        async def start(self):
            events.append("start")
            if start_failure:
                raise RuntimeError("model_startup_failed")

        async def aclose(self):
            events.append("close")

    monkeypatch.setattr(try_lfm3, "ConversationServer", Server)

    async def run():
        async with try_lfm3.owned_server(ConversationConfig()):
            events.append("chat")

    if start_failure:
        with pytest.raises(RuntimeError, match="model_startup_failed"):
            asyncio.run(run())
        assert events == ["start", "close"]
    else:
        asyncio.run(run())
        assert events == ["start", "chat", "close"]
