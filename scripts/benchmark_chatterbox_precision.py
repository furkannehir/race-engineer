"""Offline, opt-in T3 precision experiment. Never changes the live speech worker/config.

Run with the application Python; candidates run sequentially in the isolated TTS
Python. Outputs contain authored test speech only, not recorded driver audio.
"""

import argparse
import asyncio
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import random
import statistics
import subprocess
import sys
import threading
import time
import traceback
import wave
from datetime import UTC, datetime
from pathlib import Path

from race_engineer.config import load_config
from race_engineer.processes import start_owned_process
from race_engineer.tts_metadata import CHATTERBOX_VERSION, MODEL_FILES, MODEL_REVISION

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "data/benchmarks"
PRECISIONS = ("fp32", "bf16", "fp16")
CASES = (
    ("radio-en", "en", "Radio check. Your race engineer is ready."),
    ("radio-tr", "tr", "Telsiz kontrolü. Yarış mühendisin hazır."),
    ("flag-en", "en", "Yellow flag. Be careful."),
    ("flag-tr", "tr", "Sarı bayrak. Dikkatli ol."),
    (
        "position-en",
        "en",
        "You're sixth out of twenty-eight cars. The gap ahead is one point five seconds.",
    ),
    (
        "position-tr",
        "tr",
        "Yirmi sekiz araç arasında altıncı sıradasın. Öndeki araçla fark bir buçuk saniye.",
    ),
    (
        "fuel-en",
        "en",
        "You have twelve point five litres remaining. That's enough for five more laps.",
    ),
    ("fuel-tr", "tr", "On iki buçuk litre yakıtın var. Beş tur daha atabilirsin."),
    ("lap-en", "en", "Your last lap was one minute forty-two point three seconds."),
    ("lap-tr", "tr", "Son turun bir dakika kırk iki virgül üç saniyeydi."),
)


def schedule(repeats, seed):
    """Same per-case seeds/order for every candidate; rounding can still change speech."""
    result = []
    for repeat in range(repeats):
        indices = list(range(len(CASES)))
        random.Random(seed + repeat).shuffle(indices)
        result.extend((repeat + 1, index, seed + repeat * len(CASES) + index) for index in indices)
    return result


def percentile(values, fraction):
    ordered = sorted(values)
    return ordered[max(0, math.ceil(len(ordered) * fraction) - 1)]


def summarize(samples):
    if not samples:
        return {"completed": 0}
    times = [sample["synthesis_ms"] for sample in samples]
    return {
        "completed": len(samples),
        "p50_synthesis_ms": statistics.median(times),
        "p95_synthesis_ms_nearest_rank": percentile(times, 0.95),
        "p50_real_time_factor": statistics.median(s["real_time_factor"] for s in samples),
        "languages": {
            language: {
                "completed": len(selected := [s for s in samples if s["language"] == language]),
                "p50_synthesis_ms": statistics.median(s["synthesis_ms"] for s in selected),
            }
            for language in sorted({s["language"] for s in samples})
        },
    }


def summarize_memory(report):
    snapshots = [
        report[key]
        for key in ("memory_after_load", "memory_after_warmup", "memory_final")
        if key in report
    ] + [sample["memory"] for sample in report["samples"]]
    if not snapshots:
        return {}
    return {
        "peak_allocated_mib_including_startup": max(s["peak_allocated_mib"] for s in snapshots),
        "peak_reserved_mib_including_startup": max(s["peak_reserved_mib"] for s in snapshots),
        "allocated_growth_after_warmup_mib": (
            report["memory_final"]["allocated_mib"] - report["memory_after_warmup"]["allocated_mib"]
            if "memory_final" in report and "memory_after_warmup" in report
            else None
        ),
    }


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def new_output(path):
    resolved = path.resolve()
    if not resolved.is_relative_to(OUTPUT_ROOT.resolve()) or resolved == OUTPUT_ROOT.resolve():
        raise ValueError("output must be a NEW subdirectory of data/benchmarks")
    resolved.mkdir(parents=True, exist_ok=False)
    return resolved


def verified_weights(model_path):
    from race_engineer.components import digest, load_manifest

    _, artifacts = load_manifest()
    expected = {
        Path(a.destination).name: a.sha256
        for a in artifacts
        if a.artifact_id.startswith("chatterbox-")
    }
    hashes = {}
    for name in MODEL_FILES:
        hashes[name] = digest(model_path / name)
        if hashes[name] != expected.get(name):
            raise ValueError(f"model checksum mismatch: {name}")
    return hashes


def device_snapshot():
    """Whole-device sample, NOT per-process attribution or a measured device peak."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        return {"available": True, "csv": result.stdout.strip()}
    except (OSError, subprocess.SubprocessError):
        return {"available": False}


def memory_snapshot(torch):
    torch.cuda.synchronize()
    return {
        "allocated_mib": torch.cuda.memory_allocated() / 2**20,
        "reserved_mib": torch.cuda.memory_reserved() / 2**20,
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
        "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
    }


def place_model(model, torch, precision):
    """Cast on CPU first. Only floating T3 conditioning changes; token IDs stay integers."""
    if precision not in PRECISIONS:
        raise ValueError("unsupported precision")
    if precision == "bf16" and not torch.cuda.is_bf16_supported(including_emulation=False):
        raise RuntimeError("Native CUDA BF16 is unavailable; no precision/CPU fallback is allowed")
    dtype = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[precision]
    model.t3.to(dtype=dtype)
    model.conds.t3.to(dtype=dtype)
    model.t3.to(device="cuda")
    model.s3gen.to(device="cuda", dtype=torch.float32)
    model.ve.to(device="cuda", dtype=torch.float32)
    model.conds.to("cuda")
    model.device = "cuda"


def model_inventory(model):
    result = {}
    for name in ("t3", "s3gen", "ve"):
        sizes = {}
        for parameter in getattr(model, name).parameters():
            dtype = str(parameter.dtype)
            sizes[dtype] = sizes.get(dtype, 0) + parameter.numel() * parameter.element_size()
        result[name] = {"parameter_mib_by_dtype": {k: v / 2**20 for k, v in sizes.items()}}
    return result


def save_wav(path, pcm, rate):
    with path.open("xb") as handle, wave.open(handle, "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(rate)
        audio.writeframes(pcm)


def seed_inference(torch, seed):
    import numpy

    random.seed(seed)
    numpy.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_worker(args):
    """Only this subprocess imports GPU libraries. Exceptions stay visible in its log/report."""
    started = time.perf_counter()
    report = {
        "precision": args.precision,
        "status": "running",
        "pid": os.getpid(),
        "samples": [],
        "human_listening_passed": None,
        "automatic_promotion": False,
    }
    output = args.worker_output
    result_path = output / "candidate.json"
    write_json(result_path, report)
    phase = "initializing"
    try:
        import torch

        from race_engineer.tts.chatterbox_worker import (
            SpeakRequest,
            load_model,
            select_device,
            synthesize_pcm,
        )

        select_device(torch, "cuda", False)  # Real kernel check; never benchmark CPU accidentally.
        if args.precision == "bf16" and not torch.cuda.is_bf16_supported(including_emulation=False):
            raise RuntimeError("Native CUDA BF16 is unavailable")
        torch.cuda.reset_peak_memory_stats()
        report["runtime"] = {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "chatterbox": importlib.metadata.version("chatterbox-tts"),
            "transformers": importlib.metadata.version("transformers"),
            "gpu": torch.cuda.get_device_name(),
            "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        }
        report["device_before_load"] = device_snapshot()
        print(
            f"[{args.precision}] Loading cached model on CPU, then placing candidate on CUDA...",
            flush=True,
        )
        phase = "loading"
        seed_inference(torch, args.seed)
        load_started = time.perf_counter()
        model, _, _ = load_model(args.model, "cpu", False, args.threads)
        place_model(model, torch, args.precision)
        report["memory_after_load"] = memory_snapshot(torch)
        report["load_and_place_ms"] = (time.perf_counter() - load_started) * 1000
        report["inventory"] = model_inventory(model)
        report["conditioning_dtypes"] = {
            "t3": {
                key: str(value.dtype)
                for key, value in vars(model.conds.t3).items()
                if torch.is_tensor(value)
            },
            "decoder": {
                key: str(value.dtype)
                for key, value in model.conds.gen.items()
                if torch.is_tensor(value)
            },
        }

        phase = "warming"
        for index, language, text in ((0, "en", "Radio check."), (1, "tr", "Telsiz kontrolü.")):
            seed_inference(torch, args.seed + index)
            synthesize_pcm(
                model,
                SpeakRequest(id=index + 1, text=text, language=language, play_audio=False),
                threading.Event(),
                args.volume,
            )
        report["memory_after_warmup"] = memory_snapshot(torch)
        report["device_after_warmup"] = device_snapshot()
        report["ready_ms"] = (time.perf_counter() - started) * 1000
        write_json(result_path, report)

        phase = "generating"
        jobs = schedule(args.repeats, args.seed)
        for number, (repeat, case_index, seed) in enumerate(jobs, 1):
            case_id, language, text = CASES[case_index]
            print(f"[{args.precision}] {number}/{len(jobs)} repeat={repeat} {case_id}", flush=True)
            seed_inference(torch, seed)
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            generation_started = time.perf_counter()
            pcm, rate = synthesize_pcm(
                model,
                SpeakRequest(id=number + 2, text=text, language=language, play_audio=False),
                threading.Event(),
                args.volume,
            )
            torch.cuda.synchronize()
            elapsed_ms = (time.perf_counter() - generation_started) * 1000
            memory = memory_snapshot(torch)
            filename = f"{repeat:02d}-{case_id}.wav"
            save_wav(output / filename, pcm, rate)
            duration = len(pcm) / (2 * rate)
            report["samples"].append(
                {
                    "case_id": case_id,
                    "language": language,
                    "repeat": repeat,
                    "seed": seed,
                    "text": text,
                    "wav": filename,
                    "sample_rate": rate,
                    "audio_duration_s": duration,
                    "synthesis_ms": elapsed_ms,
                    "real_time_factor": elapsed_ms / (duration * 1000),
                    "memory": memory,
                    "pcm_sha256": hashlib.sha256(pcm).hexdigest(),
                    "all_zero_pcm": not any(pcm),
                }
            )
            write_json(result_path, report)
        report["status"] = "completed"
        report["device_after_requests"] = device_snapshot()
        report["inventory_after_requests"] = model_inventory(model)
        report["memory_final"] = memory_snapshot(torch)
    except Exception as error:
        report["status"] = "failed"
        report["failure"] = {
            "phase": phase,
            "type": type(error).__name__,
            "message": str(error)[:1500],
        }
        traceback.print_exc(file=sys.stderr)
        print(
            f"[{args.precision}] FAILED during {phase}; see candidate.json and worker.log",
            flush=True,
        )
    report["summary"] = summarize(report["samples"])
    report["summary"]["memory"] = summarize_memory(report)
    report["elapsed_ms"] = (time.perf_counter() - started) * 1000
    write_json(result_path, report)
    return 0 if report["status"] == "completed" else 1


async def run_candidate(args, precision, output, speech):
    directory = output / precision
    directory.mkdir()
    command = [
        str(speech.python_path.resolve()),
        "-u",
        str(Path(__file__).resolve()),
        "--worker",
        "--precision",
        precision,
        "--worker-output",
        str(directory),
        "--model",
        str(speech.model_path.resolve()),
        "--threads",
        str(speech.threads),
        "--volume",
        str(speech.volume),
        "--repeats",
        str(args.repeats),
        "--seed",
        str(args.seed),
    ]
    environment = dict(os.environ)
    environment.update(
        HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", DO_NOT_TRACK="1", PYTHONIOENCODING="utf-8"
    )
    with (directory / "worker.log").open("x", encoding="utf-8") as log:
        process = await start_owned_process(
            *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=log,
            env=environment,
            cwd=ROOT,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )

        async def consume():
            async for line in process.stdout:
                print(line.decode("utf-8", errors="replace").rstrip(), flush=True)
            return await process.wait()

        try:
            returncode = await asyncio.wait_for(consume(), args.timeout_s)
        except TimeoutError:
            returncode = "timeout"
        finally:
            if process.returncode is None:
                process.kill()
            await process.wait()  # No candidate overlap, including timeout/Ctrl+C.
    path = directory / "candidate.json"
    report = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {"samples": []}
    if returncode != 0:
        report["status"] = "failed"
        report.setdefault("failure", {"phase": "process", "message": str(returncode)})
    report["returncode"] = returncode
    write_json(path, report)
    return report


def comparison(candidates):
    baseline = candidates.get("fp32", {})
    if baseline.get("status") != "completed":
        return {"available": False, "reason": "FP32 baseline did not complete"}
    result = {}
    for precision, candidate in candidates.items():
        if precision == "fp32" or candidate.get("status") != "completed":
            continue
        pairs = list(zip(baseline["samples"], candidate["samples"], strict=True))
        if any(
            (a["case_id"], a["repeat"], a["seed"]) != (b["case_id"], b["repeat"], b["seed"])
            for a, b in pairs
        ):
            raise ValueError("candidate sample schedules differ")
        result[precision] = {
            "warm_allocated_saved_mib": baseline["memory_after_warmup"]["allocated_mib"]
            - candidate["memory_after_warmup"]["allocated_mib"],
            "warm_reserved_saved_mib": baseline["memory_after_warmup"]["reserved_mib"]
            - candidate["memory_after_warmup"]["reserved_mib"],
            "p50_synthesis_ms_change": candidate["summary"]["p50_synthesis_ms"]
            - baseline["summary"]["p50_synthesis_ms"],
            "listening_required": True,
            "paired_audio": [
                {
                    "case_id": a["case_id"],
                    "repeat": a["repeat"],
                    "baseline": f"fp32/{a['wav']}",
                    "candidate": f"{precision}/{b['wav']}",
                    "duration_ratio": b["audio_duration_s"] / a["audio_duration_s"],
                }
                for a, b in pairs
            ],
        }
    return {"available": bool(result), "candidates": result}


async def benchmark(args):
    config = load_config(args.config)
    if not config.tts.python_path.is_file():
        raise ValueError("TTS runtime missing; run scripts/setup_radio_tts.py --backend cuda")
    print("Verifying the pinned local model checksums...", flush=True)
    hashes = await asyncio.to_thread(verified_weights, config.tts.model_path)
    output = new_output(args.output)
    print(f"Results and listening samples: {output}", flush=True)
    report = {
        "schema_version": "chatterbox-precision-benchmark.v1",
        "created_at": datetime.now(UTC).isoformat(),
        "chatterbox_version": CHATTERBOX_VERSION,
        "model_revision": MODEL_REVISION,
        "model_sha256": hashes,
        "repeats": args.repeats,
        "seed": args.seed,
        "threads": config.tts.threads,
        "volume": config.tts.volume,
        "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "production_worker_sha256": hashlib.sha256(
            (ROOT / "src/race_engineer/tts/chatterbox_worker.py").read_bytes()
        ).hexdigest(),
        "limitations": [
            "Authored synthetic speech, not a live/game-load or end-to-end radio benchmark.",
            "CUDA allocator peaks exclude driver allocations; device observations are samples, "
            "not peaks or per-process attribution.",
            "Same seeds do not guarantee identical output across precisions "
            "or deterministic CUDA kernels.",
            "Successful synthesis is not human speech-quality acceptance; no automatic promotion.",
            "FP32 also loads on CPU first; compare with a live smoke test before integration.",
        ],
        "candidates": {},
    }
    for precision in args.candidate or ("fp32", "bf16"):
        report["candidates"][precision] = await run_candidate(args, precision, output, config.tts)
        report["comparison"] = comparison(report["candidates"])
        write_json(output / "report.json", report)
    for precision, candidate in report["candidates"].items():
        print(
            f"{precision}: {candidate['status']} {json.dumps(candidate.get('summary', {}))}",
            flush=True,
        )
    print(f"Report: {output / 'report.json'}\nLive configuration has NOT changed.", flush=True)
    return 1 if any(c["status"] != "completed" for c in report["candidates"].values()) else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config/default.toml")
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_ROOT / f"chatterbox-precision-{datetime.now(UTC):%Y%m%dT%H%M%SZ}",
    )
    parser.add_argument(
        "--candidate",
        choices=PRECISIONS,
        action="append",
        help="default: fp32 then bf16; repeat to select candidates",
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--timeout-s", type=float, default=900, help="maximum seconds per isolated candidate"
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--precision", choices=PRECISIONS, help=argparse.SUPPRESS)
    parser.add_argument("--worker-output", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--model", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--threads", type=int, default=4, help=argparse.SUPPRESS)
    parser.add_argument("--volume", type=float, default=0.8, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not 1 <= args.repeats <= 20 or not 0 <= args.seed < 2**31 or not 1 <= args.timeout_s <= 3600:
        parser.error("repeats must be 1..20, seed 0..2^31-1 and timeout-s 1..3600")
    if args.candidate and len(args.candidate) != len(set(args.candidate)):
        parser.error("candidate must not be repeated")
    if args.worker:
        if args.precision is None or args.worker_output is None or args.model is None:
            parser.error("internal worker arguments missing")
        os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", DO_NOT_TRACK="1")
        return run_worker(args)
    try:
        return asyncio.run(benchmark(args))
    except KeyboardInterrupt:
        print(
            "Benchmark interrupted; the owned worker was closed. Live configuration is unchanged.",
            flush=True,
        )
        return 130
    except (OSError, ValueError, RuntimeError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    raise SystemExit(main())
