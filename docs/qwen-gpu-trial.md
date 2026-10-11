# Qwen-only NVIDIA GPU setup and trial measurements

Status, 2026-10-11: the measured NVIDIA/CUDA path is now the standard source-mode
default on `main`. Interpreter experiments were separated onto `light-version`.
This is not packaged GPU distribution, AMD validation, an Automatic-mode release
gate, or proof that interpretation/reply quality is solved.

## What changes

Only Qwen3-4B moves to CUDA: it still performs factual context planning and Core/reply
generation with the same Q4_K_M model, v6 prompt, eight threads, 8192-token context,
one parallel slot and 30-second request timeout. Qwen3-ASR remains CPU and Piper/SAPI
remain unchanged. The standard edition does not load the trained MiniLM worker;
request-extraction training and the smaller CPU reply model belong to Pitward Light.
Missing iRacing gap inputs and incorrect/unsupported replies remain separate tasks.

`config/default.toml` now selects GPU mode; setup never rewrites it. Comparison
profiles copy its current settings and differ from each other only in Qwen runtime
mode. The normal configuration and both comparison profiles use loopback
port **8088**, require an owned server and disable compute fallback. That prevents
the old unmanaged CPU server on 8087, or a silent CPU fallback, from masquerading as
a GPU test. If the trial port is occupied, startup refuses without stopping or
borrowing the other server. CPU mode remains an explicit compatibility override,
not an automatic fallback. The generic runtime still supports opt-in external reuse
and compute fallback for deliberately configured integrations.

## Setup and race test

The local setup is already complete on the development machine. For another source
checkout, run the explicit online installer:

```powershell
.\.venv\Scripts\python.exe scripts\setup_qwen_gpu.py --runtime-only
```

It downloads only the official llama.cpp **b10964 CUDA 13.3 x64 executable archive**
and its matching CUDA DLL archive, verifies pinned SHA-256 checksums and sizes, and
extracts to `data/conversation-prototype/llama-b10964-cuda-13.3/`. Downloaded archives
and per-file provenance remain in ignored project data. No model is downloaded,
no Python environment is upgraded, and no system driver or CUDA toolkit is installed.
llama.cpp uses MIT licensing; NVIDIA's CUDA redistributable terms still apply. The
packaged component manifest remains CPU-only.

The setup verifies that the installed runtime can see `CUDA0`. `--runtime-only`
installs/verifies CUDA without touching any existing comparison profiles.
Existing altered archives, runtime files or trial profiles are never overwritten.
If the source configuration changes, preserve/update the existing profiles explicitly;
the setup deliberately refuses to replace a differing snapshot. Failed downloads
retain their `.partial` files for diagnosis, and failed extraction preserves its
fresh stage rather than deleting an existing directory.

Close the existing panel, then launch the standard GPU edition:

```powershell
.\start-panel.cmd
```

Click Start Engineer normally; **do not start a separate server on 8088 first**.
The panel owns and closes its CUDA server. The existing CPU server on 8087 is left
untouched. The effective-runtime event in `logs/control-panel.jsonl` should contain
`conversation_runtime_ready`, `mode: gpu`, `backend: cuda`. A missing CUDA runtime,
unsupported device, insufficient memory or occupied trial port produces a startup
error rather than a misleading CPU result.

Join a test race and compare several naturally worded English/Turkish questions.
Record button-release-to-first-audio delay, whether the whole request was answered,
and game frame-time/FPS behavior. Check VRAM while driving, particularly with VR:

```powershell
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv
```

Speech recognition and synthesis still add time. Missing gap data can still produce
an honest unavailable answer, and fast computation does not guarantee correct or
complete interpretation. No game-load performance claim has been established yet.

`start-panel-gpu.cmd` is retained as an alias of the regular launcher; it does not
load old ignored trial files, which may contain interpreter-only settings.
To compare on CPU, close the panel and use a newly generated CPU comparison profile
with `start-panel.cmd --config <path>`, or explicitly set runtime mode to `cpu` in a
compatible config. This remains Qwen on CPU, not the future Pitward Light pipeline.
Model files, profiles, audio preferences and PTT bindings are preserved. Do not
delete runtime/model files to switch modes.

## Reproduce the typed comparison

Generate profiles in a fresh directory so older interpreter-enabled profiles are
preserved rather than overwritten (choose another directory on later config changes):

```powershell
.\.venv\Scripts\python.exe scripts\setup_qwen_gpu.py `
    --profiles-dir data\conversation-prototype\main-comparison
```

Run the following sequentially, with the panel stopped. Choose new report filenames
on every repeat; the benchmark refuses to overwrite existing results:

```powershell
.\.venv\Scripts\python.exe scripts\benchmark_intelligence.py `
    --config data\conversation-prototype\main-comparison\cpu-trial.toml `
    --repeats 2 --require-owned-runtime --expect-backend cpu --sample-gpu-memory `
    --output data\evaluations\cpu-trial-new.json

.\.venv\Scripts\python.exe scripts\benchmark_intelligence.py `
    --config data\conversation-prototype\main-comparison\gpu-trial.toml `
    --repeats 2 --require-owned-runtime --expect-backend cuda --sample-gpu-memory `
    --output data\evaluations\cuda-trial-new.json
```

These use six already-observed bilingual development controls repeated twice, with
identical question order/seed and fresh owned runtimes. They measure the full typed
Qwen Context -> application retrieval -> Qwen Core/response -> refresh/grounding path,
not the trained social shortcut, microphone, STT, radio queue, TTS or audio playback.
The benchmark stops after a transport timeout rather than piling more requests onto
a possibly still-busy server, and closes only its own process.

## Initial measurements

These historical measurements predate the branch split and were taken through the
full typed Qwen path; the experimental action classifier was not used by this benchmark.

The content-free aggregate and exact fingerprints are in
[qwen-gpu-trial-2026-10-11.json](evaluations/qwen-gpu-trial-2026-10-11.json).
Detailed content-free reports remain under ignored `data/evaluations/`.

| Measurement | CPU | CUDA |
| --- | --- | --- |
| First identical question | Context timeout at 30.01 s | Full typed pipeline completed in 1.19 s |
| Completed turns | 0/1 attempted; stopped early | 12/12 |
| Completed-turn median / p95 | Not available | 0.772 / 1.215 s |
| Runtime startup/health check | 3.83 s | 3.17 s |
| Exact planner selections / evidence expectations | No completed plan | 12/12 each |

The CPU run was **censored by its first timeout**, so there is no completed CPU
median or general speedup factor to quote. GPU timing ranged 0.486–1.243 seconds,
including two Core repair requests across the run. Excluding the first turn, median
was 0.703 seconds and sample p95 1.111 seconds. Small synthetic development results
are not release acceptance or race-load latency guarantees.

Whole-device VRAM samples rose from **7,298 MiB** before CUDA startup to a highest
sample of **11,214 MiB**, then returned to **7,297 MiB** after owned shutdown. This
is approximately 3.8 GiB of additional observed usage, including other applications'
variation; it is not an isolated process-memory or true peak measurement. Roughly
5 GiB remained free in that sample, which must be rechecked under the actual racing
workload. The same host used driver 591.86 and an RTX 5080 with 16,303 MiB reported
capacity. No driver update was required for this successful run.

Correct planner/evidence results do **not** prove complete reply quality. Manual
inspection found one Turkish fuel-and-speed reply omitted the requested fuel amount
after a repair, and some Turkish unit wording remained awkward. Automatic reply
accuracy remains unknown. Keep these observations separate from the measured speed
improvement and the existing short-STT/Turkish-TTS issues.

Upstream references: [pinned b10964 assets](https://github.com/ggml-org/llama.cpp/releases/tag/b10964),
[GPU builds](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md), and
[CUDA minor-version compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html).
Minor-version compatibility has feature limitations; device discovery alone is not
a substitute for running the real model and checking the effective backend.
