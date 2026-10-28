import asyncio
import copy
import json
import wave
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import benchmark_chatterbox_precision as benchmark


def test_schedule_is_paired_balanced_and_reproducible():
    jobs = benchmark.schedule(3, 7)
    assert jobs == benchmark.schedule(3, 7)
    assert len(jobs) == 30
    assert len({(repeat, index) for repeat, index, _ in jobs}) == 30
    assert len({seed for _, _, seed in jobs}) == 30
    assert {benchmark.CASES[index][1] for _, index, _ in jobs} == {"en", "tr"}
    assert [index for repeat, index, _ in jobs if repeat == 1] != [
        index for repeat, index, _ in jobs if repeat == 2
    ]


def test_summary_reports_language_counts_latency_and_duration_normalized_speed():
    samples = [
        {"language": "en", "synthesis_ms": 100, "real_time_factor": 0.5},
        {"language": "tr", "synthesis_ms": 300, "real_time_factor": 0.25},
    ]
    summary = benchmark.summarize(samples)
    assert summary["p50_synthesis_ms"] == 200
    assert summary["p95_synthesis_ms_nearest_rank"] == 300
    assert summary["p50_real_time_factor"] == 0.375
    assert summary["languages"]["tr"] == {"completed": 1, "p50_synthesis_ms": 300}
    assert benchmark.summarize([]) == {"completed": 0}


def test_output_is_new_and_confined_to_ignored_benchmark_tree(tmp_path, monkeypatch):
    root = tmp_path / "data/benchmarks"
    monkeypatch.setattr(benchmark, "OUTPUT_ROOT", root)
    output = benchmark.new_output(root / "trial")
    with pytest.raises(FileExistsError):
        benchmark.new_output(output)
    for invalid in (root, tmp_path / "config", root / "../outside"):
        with pytest.raises(ValueError, match="NEW subdirectory"):
            benchmark.new_output(invalid)


def test_peak_memory_includes_startup_even_when_per_request_counters_reset():
    report = {
        "memory_after_load": {
            "peak_allocated_mib": 3200,
            "peak_reserved_mib": 3300,
            "allocated_mib": 3100,
        },
        "memory_after_warmup": {
            "peak_allocated_mib": 3250,
            "peak_reserved_mib": 3350,
            "allocated_mib": 3120,
        },
        "memory_final": {
            "peak_allocated_mib": 3150,
            "peak_reserved_mib": 3400,
            "allocated_mib": 3120,
        },
        "samples": [],
    }
    assert benchmark.summarize_memory(report) == {
        "peak_allocated_mib_including_startup": 3250,
        "peak_reserved_mib_including_startup": 3400,
        "allocated_growth_after_warmup_mib": 0,
    }
    assert benchmark.summarize_memory({"samples": []}) == {}


def test_t3_cast_precedes_cuda_transfer_decoder_stays_fp32_and_ids_remain_integer():
    calls = []

    class Module:
        def __init__(self, name):
            self.name = name
            self.dtype = "fp32"

        def to(self, **kwargs):
            calls.append((self.name, kwargs))
            self.dtype = kwargs.get("dtype", self.dtype)

    class Conditioning(Module):
        # Mirrors the pinned T3Cond.to contract: do not cast integer token IDs.
        ids_dtype = "int64"

    t3_cond = Conditioning("t3_cond")
    conds = SimpleNamespace(t3=t3_cond, to=lambda device: calls.append(("conds", device)))
    model = SimpleNamespace(
        t3=Module("t3"), s3gen=Module("s3gen"), ve=Module("ve"), conds=conds, device="cpu"
    )
    torch = SimpleNamespace(
        float32="fp32",
        bfloat16="bf16",
        float16="fp16",
        cuda=SimpleNamespace(is_bf16_supported=lambda **_: True),
    )
    benchmark.place_model(model, torch, "bf16")
    assert calls[:3] == [
        ("t3", {"dtype": "bf16"}),
        ("t3_cond", {"dtype": "bf16"}),
        ("t3", {"device": "cuda"}),
    ]
    assert model.t3.dtype == t3_cond.dtype == "bf16"
    assert model.s3gen.dtype == model.ve.dtype == "fp32"
    assert t3_cond.ids_dtype == "int64"
    assert model.device == "cuda"
    torch.cuda.is_bf16_supported = lambda **_: False
    with pytest.raises(RuntimeError, match="no precision/CPU fallback"):
        benchmark.place_model(model, torch, "bf16")


def candidate():
    return {
        "status": "completed",
        "samples": [
            {
                "case_id": "flag-en",
                "repeat": 1,
                "seed": 7,
                "wav": "01-flag-en.wav",
                "audio_duration_s": 2,
            }
        ],
        "memory_after_warmup": {"allocated_mib": 3100, "reserved_mib": 3300},
        "summary": {"p50_synthesis_ms": 2000},
    }


def test_comparison_requires_successful_baseline_and_matched_requests():
    baseline, reduced = candidate(), candidate()
    reduced["memory_after_warmup"]["allocated_mib"] = 2078
    reduced["summary"]["p50_synthesis_ms"] = 1800
    reduced["samples"][0]["audio_duration_s"] = 2.5
    report = benchmark.comparison({"fp32": baseline, "bf16": reduced})
    assert report["candidates"]["bf16"]["warm_allocated_saved_mib"] == 1022
    assert report["candidates"]["bf16"]["p50_synthesis_ms_change"] == -200
    assert report["candidates"]["bf16"]["paired_audio"][0]["duration_ratio"] == 1.25
    assert report["candidates"]["bf16"]["listening_required"]
    reduced["samples"][0]["seed"] = 123
    with pytest.raises(ValueError, match="schedules differ"):
        benchmark.comparison({"fp32": baseline, "bf16": reduced})
    assert not benchmark.comparison({"bf16": reduced})["available"]
    reduced["status"] = "failed"
    assert not benchmark.comparison({"fp32": baseline, "bf16": reduced})["available"]


def test_wav_is_pcm_and_existing_listening_samples_are_never_overwritten(tmp_path):
    path = tmp_path / "sample.wav"
    pcm = b"\x01\x00" * 24_000
    benchmark.save_wav(path, pcm, 24_000)
    with wave.open(str(path), "rb") as audio:
        assert audio.getparams()[:4] == (1, 2, 24_000, 24_000)
        assert audio.readframes(24_000) == pcm
    with pytest.raises(FileExistsError):
        benchmark.save_wav(path, pcm, 24_000)


def test_model_checksums_are_verified_before_launch(tmp_path, monkeypatch):
    import race_engineer.components as components

    names = benchmark.MODEL_FILES
    monkeypatch.setattr(components, "digest", lambda path: f"sha-{path.name}")
    artifacts = [
        SimpleNamespace(artifact_id=f"chatterbox-{i}", destination=Path(name), sha256=f"sha-{name}")
        for i, name in enumerate(names)
    ]
    monkeypatch.setattr(components, "load_manifest", lambda: ("2", artifacts))
    assert len(benchmark.verified_weights(tmp_path)) == 5
    artifacts[0].sha256 = "wrong"
    with pytest.raises(ValueError, match="checksum mismatch"):
        benchmark.verified_weights(tmp_path)


@pytest.mark.parametrize("outcome", ["success", "timeout", "cancel"])
def test_controller_isolates_children_and_waits_for_cleanup(tmp_path, monkeypatch, outcome):
    launches = []
    process = SimpleNamespace(returncode=None, killed=False, waited=False)

    async def lines():
        if outcome != "success":
            if outcome == "cancel":
                raise asyncio.CancelledError
            await asyncio.sleep(10)
        yield b"progress\n"

    process.stdout = lines()

    async def wait():
        process.waited = True
        process.returncode = 0
        return 0

    def kill():
        process.killed = True
        process.returncode = -1

    process.wait, process.kill = wait, kill

    async def spawn(*args, **kwargs):
        launches.append((args, kwargs))
        report = copy.deepcopy(candidate())
        benchmark.write_json(tmp_path / "bf16/candidate.json", report)
        return process

    monkeypatch.setattr(benchmark, "start_owned_process", spawn)
    speech = SimpleNamespace(
        python_path=Path("python.exe"), model_path=tmp_path, threads=4, volume=0.8
    )
    args = Namespace(repeats=1, seed=7, timeout_s=0.01)
    if outcome == "cancel":
        with pytest.raises(asyncio.CancelledError):
            asyncio.run(benchmark.run_candidate(args, "bf16", tmp_path, speech))
    else:
        result = asyncio.run(benchmark.run_candidate(args, "bf16", tmp_path, speech))
        assert result["status"] == ("completed" if outcome == "success" else "failed")
    assert process.waited
    assert process.killed == (outcome != "success")
    command, options = launches[0]
    assert "--worker" in command and "--precision" in command and "bf16" in command
    assert options["env"]["HF_HUB_OFFLINE"] == options["env"]["TRANSFORMERS_OFFLINE"] == "1"
    assert options["stderr"] is not None and "shell" not in options
    assert json.loads((tmp_path / "bf16/candidate.json").read_text())["samples"]
