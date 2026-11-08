import asyncio
import copy
import json
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import pytest

from race_engineer.config import ConversationConfig, load_config
from race_engineer.conversation.runtime import (
    ConversationTrialOptions,
    RuntimeCandidate,
    server_command,
)
from scripts import benchmark_qwen_cache as benchmark

ROOT = Path(__file__).resolve().parents[1]
LOG = """
load_tensors: offloaded 37/37 layers to GPU
load_tensors: CUDA0 model buffer size = 2375.91 MiB
llama_context: n_ctx = 8192
llama_context: flash_attn = auto
llama_kv_cache: CUDA0 KV buffer size = 1152.00 MiB
llama_kv_cache: size = 1152.00 MiB (8192 cells, 36 layers), K (f16): 576 MiB, V (f16): 576 MiB
resolve_fused_ops: Flash Attention enabled
sched_reserve: CUDA0 compute buffer size = 83.01 MiB
sched_reserve: CUDA_Host compute buffer size = 18.01 MiB
"""


def test_effective_runtime_validation_uses_backend_logs_not_auto_request():
    observed = benchmark.parse_runtime_log(LOG)
    benchmark.validate_runtime(observed, "f16", 8192)
    assert observed["flash_attention_enabled"] is True
    assert observed["cuda_kv_mib"] == 1152
    assert observed["cuda_compute_mib"] == 83.01
    with pytest.raises(ValueError, match="precision"):
        benchmark.validate_runtime(observed, "q8_0", 8192)
    with pytest.raises(ValueError, match="context"):
        benchmark.validate_runtime(observed, "f16", 4096)
    missing_flash = benchmark.parse_runtime_log(LOG.replace("Flash Attention enabled", ""))
    with pytest.raises(ValueError, match="Flash Attention"):
        benchmark.validate_runtime(missing_flash, "f16", 8192)
    offload = benchmark.parse_runtime_log(LOG.replace("37/37", "10/37"))
    with pytest.raises(ValueError, match="offload"):
        benchmark.validate_runtime(offload, "f16", 8192)


def test_q8_backend_proof_works_without_halving_the_entire_model():
    quantized = LOG.replace("1152.00", "612.00").replace("f16", "q8_0")
    observed = benchmark.parse_runtime_log(quantized)
    benchmark.validate_runtime(observed, "q8_0", 8192)
    assert observed["cuda_model_mib"] == 2375.91
    assert observed["cuda_kv_mib"] == 612


def test_trial_arguments_are_opt_in_and_preserve_live_command_and_capacity(tmp_path):
    candidate = RuntimeCandidate("gpu", Path("server.exe"), Path("model.gguf"), "cuda", "CUDA0", -1)
    config = ConversationConfig()
    baseline = server_command(candidate, config)
    assert "--flash-attn" not in baseline and "--cache-type-k" not in baseline
    trial = ConversationTrialOptions("on", "q8_0", tmp_path / "runtime.log")
    command = server_command(candidate, config, trial)
    assert command[: len(baseline)] == baseline
    for flag in ("--cache-type-k", "--cache-type-v"):
        assert command[command.index(flag) + 1] == "q8_0"
    assert command[command.index("--flash-attn") + 1] == "on"
    with pytest.raises(ValueError):
        ConversationTrialOptions("off", "f16", tmp_path / "log")
    with pytest.raises(ValueError):
        ConversationTrialOptions("on", "q4_0", tmp_path / "log")


def test_trial_profile_leaves_live_settings_stt_tts_model_and_prompts_untouched():
    config = load_config(ROOT / "config/qwen-legacy.toml")
    before = config.model_dump()
    trial = benchmark.trial_config(config, 8097)
    assert config.model_dump() == before
    original = config.conversation.model_dump()
    original.update(port=8097)
    assert trial.model_dump() == original
    assert trial.context_planner == "qwen-context-v7" and trial.runtime.context_size == 8192


def test_default_case_selection_is_balanced_with_seeded_history_and_is_not_locked_data():
    config = load_config(ROOT / "config/qwen-legacy.toml")
    cases, inputs = benchmark.load_cases(Namespace(case=None, full_suite=False), config)
    assert len(cases) == 20
    assert sum(case[2].language == "en" for case in cases) == 10
    assert any(case[2].recent_dialogue for case in cases)
    assert all("locked" not in str(path) and "calibration" not in str(path) for path in inputs)


def test_v8_is_explicit_trial_only_and_validation_adds_development_data():
    config = load_config(ROOT / "config/qwen-legacy.toml")
    profile = benchmark.trial_config(config, 8097, "qwen-context-v8")
    assert profile.context_planner == "qwen-context-v8"
    assert config.conversation.context_planner == "qwen-context-v7"
    cases, _ = benchmark.load_cases(Namespace(case=None, full_suite=True, validation=True), config)
    assert len(cases) == 100
    assert sum(case[2].language == "en" for case in cases) == 50


def test_output_is_new_and_confined_to_ignored_data(tmp_path, monkeypatch):
    monkeypatch.setattr(benchmark, "OUTPUT_ROOT", tmp_path / "data/benchmarks")
    target = benchmark.new_output(benchmark.OUTPUT_ROOT / "trial")
    with pytest.raises(FileExistsError):
        benchmark.new_output(target)
    with pytest.raises(ValueError):
        benchmark.new_output(tmp_path / "config")


def row(case_id="radio-en", passed=True):
    return {
        "case_id": case_id,
        "question_id": case_id,
        "pipeline_completed": passed,
        "planner_exact_match": passed,
        "evidence_expectation_match": passed,
        "core_retries": 0,
        "timing_ms": {"total": 100, "context": 40, "core": 60},
        "failure": None,
    }


def test_pairs_distinguish_regressions_improvements_and_preexisting_failures():
    left = {
        "round": 1,
        "cache_type": "f16",
        "status": "completed",
        "effective_runtime": {"cuda_kv_mib": 1152},
        "results": [row("regression"), row("improvement", False), row("shared", False)],
    }
    right = {
        "round": 1,
        "cache_type": "q8_0",
        "status": "completed",
        "effective_runtime": {"cuda_kv_mib": 612},
        "results": [row("regression", False), row("improvement"), row("shared", False)],
    }
    result = benchmark.paired_comparison([left, right])
    assert result["verified_kv_saved_mib"] == 540
    assert result["paired_turns"] == 3
    assert (
        len(result["regressions"])
        == len(result["improvements"])
        == len(result["shared_failures"])
        == 1
    )
    assert result["automatic_promotion"] is False and result["manual_reply_review_required"]
    broken = copy.deepcopy(right)
    broken["results"][0]["question_id"] = "different"
    with pytest.raises(ValueError, match="schedules differ"):
        benchmark.paired_comparison([left, broken])
    broken["status"] = "failed"
    assert benchmark.paired_comparison([left, broken])["paired_turns"] == 0


@pytest.mark.parametrize("outcome", ["success", "timeout", "cancel"])
def test_trial_closes_owned_server_and_keeps_partial_results(tmp_path, monkeypatch, outcome):
    instances = []
    entered = asyncio.Event()

    class Server:
        def __init__(self, root, config, trial):
            self.closed = False
            self.process = SimpleNamespace(pid=123)
            instances.append(self)

        async def aclose(self):
            self.closed = True

    async def verified(*args):
        entered.set()
        if outcome != "success":
            await asyncio.sleep(10)
        return {"cuda_kv_mib": 1152}, {}

    async def measure(config, contexts, group, question, **kwargs):
        return row(group.id), "Authored synthetic answer"

    monkeypatch.setattr(benchmark, "ConversationServer", Server)
    monkeypatch.setattr(benchmark, "verified_server", verified)
    monkeypatch.setattr(benchmark, "require_free_port", lambda _: None)
    monkeypatch.setattr(benchmark, "gpu_memory_snapshot", lambda: {"available": False})
    monkeypatch.setattr(benchmark, "measure_intelligence_turn", measure)
    cases = [
        (
            "test",
            (),
            SimpleNamespace(id=f"test-{lang}", language=lang, recent_dialogue=()),
            "Authored question",
        )
        for lang in ("en", "tr")
    ]
    config = load_config(ROOT / "config/qwen-legacy.toml").conversation
    args = Namespace(seed=7, timeout_s=0.05 if outcome == "timeout" else 10)

    async def run():
        task = asyncio.create_task(benchmark.run_trial(args, config, cases, tmp_path, "f16", 1))
        if outcome == "cancel":
            await entered.wait()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        else:
            result = await task
            assert result["status"] == ("completed" if outcome == "success" else "failed")

    asyncio.run(run())
    assert instances[0].closed
    result = json.loads((tmp_path / "round-01-f16/result.json").read_text(encoding="utf-8"))
    if outcome == "success":
        assert len(result["results"]) == 2
        assert result["automatic_reply_accuracy"] is None
    elif outcome == "cancel":
        assert result["status"] == "cancelled"


def test_occupied_port_is_refused_without_contacting_or_stopping_the_listener(monkeypatch):
    class Socket:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def setsockopt(self, *args):
            pass

        def bind(self, address):
            assert address[0] == "127.0.0.1"
            raise OSError("occupied")

    monkeypatch.setattr(benchmark.socket, "socket", Socket)
    with pytest.raises(ValueError, match="left untouched"):
        benchmark.require_free_port(8097)
