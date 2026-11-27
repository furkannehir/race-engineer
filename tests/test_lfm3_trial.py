import json
import tomllib
from pathlib import Path

import pytest

from race_engineer.config import AppConfig, ConversationConfig
from race_engineer.conversation.runtime import RuntimeCandidate, server_command
from scripts import setup_lfm3_trial

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
