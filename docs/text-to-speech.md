# Local speech: Chatterbox Multilingual

The default setup uses **Chatterbox TTS 0.1.7**, its **500M multilingual V2**
model and built-in default voice. An opt-in **Multilingual V3 race trial** is
available below. Both English (`en`) and Turkish (`tr`) use
the same model and conditioning. There is no reference clip, uploaded voice, hosted
TTS API, SAPI process or Piper fallback. PTT open/close WAV cues are unchanged.
The default precision policy uses BF16 for T3 on natively supported CUDA GPUs;
the audio decoder stays FP32. CPU speech stays FP32. Reference-only encoders stay
in system RAM when using the cached default voice, including on CUDA sessions.

The pinned package's `ChatterboxMultilingualTTS.from_local(path, device=...)`
loads `ve.pt`, `t3_mtl23ls_v2.safetensors`, `s3gen.pt` and the tokenizer.
`conds.pt` supplies built-in conditioning. Pitward requires it and verifies that
both languages are supported before accepting speech. Generation calls
`model.generate(text, language_id=language)` without `audio_prompt_path`.
The package's default style controls and watermarking are retained; no emotion or
custom-voice controls are exposed. This deliberately uses the V2 files expected by
0.1.7, not a newer upstream README's V3 recipe or English-only Turbo.

## V3 source race trial (2026-11-07)

The V3 trial uses the same shared backend for conversation and automatic calls,
with explicit `[tts] model_variant = "v3"`. The worker calls
`from_local(..., t3_model="v3")` and reports the loaded variant at readiness. A V3
request rejects a legacy V2 worker instead of accepting a different checkpoint.
The English/Turkish built-in voice requires no reference recording.

Setup uses upstream source commit `ab55923c27757c90cfcc407baa57dc25a7d25217`
and the checksum-verified `t3_mtl23ls_v3.safetensors` checkpoint from model revision
`5bb1f6ee58e50c3b8d408bc82a6d3740c2db6e18`. Upstream still labels its source
package 0.1.7, so that package version alone does not identify V3 support.
The source archive SHA256 observed during installation is
`19379659a7464e37237a63fbc788ad8c1d3c387f82b69765fe82099bb4e7f752`.

Provision the separate runtime and cached checkpoint:

```powershell
.\.venv\Scripts\python.exe scripts\setup_radio_tts.py --backend cuda --model-variant v3
```

This installs `data/tts-chatterbox-v3/runtime` and reuses verified common model
assets. The V2 runtime and default component manifest remain available. CUDA uses
the existing Windows RTX 50xx PyTorch 2.7.1/cu128 compatibility override and the
same explicitly constrained dependency file. V3 is currently a source-app trial;
the existing packaged release/component set still uses V2.

In this workspace, `data/trials/chatterbox-v3-live.toml` is prepared from the last
v8/FP16 live profile. It preserves Qwen, STT, the delivery budget, SQLite profile
and the panel's saved input/output/PTT settings, selecting these TTS settings:

```toml
[tts]
model_variant = "v3"
python_path = "data/tts-chatterbox-v3/runtime/Scripts/python.exe"
device = "cuda"
precision = "bf16"
fallback_to_cpu = false
```

The profile requires CUDA for an interpretable GPU race test. Normal CPU fallback
configuration remains available in the default setup. On another checkout, copy
the desired live TOML into `data/trials` and change only these TTS keys, retaining
its existing `model_path` and other settings.

Close any open panel completely before launching the new profile:

```powershell
.\start-panel.cmd --config data\trials\chatterbox-v3-live.toml
```

Rollback to the previous live test with
`.\start-panel.cmd --config data\trials\qwen-v8-live.toml`, or use the ordinary
launcher for the V7/V2 defaults. Both paths use the same saved panel settings.

Startup logs now record the selected variant, worker PID, initialization time,
CUDA tensor allocation and reservation after warm-up. These memory figures belong
to the speech worker's PyTorch allocator; they are not total VRAM or measured peaks.
Existing per-turn diagnostics retain synthesis, playback and delivery outcomes.

Actual smoke validation completed five conversational outputs (EN/TR radio checks,
the two English failure sentences and the Turkish historical-data limitation) and
two automatic calls with one CUDA/BF16 V3 worker. Preparation took 1.29–1.52 seconds
for those replies, after a 33.66-second initialization. Allocated tensors were
2100 MiB and reservation 2190 MiB after warm-up. These are isolated observations,
not a controlled V2/V3 comparison or race-load acceptance. No claim that unwanted
tails are eliminated is made from playback completion alone.

Readiness report: `data/diagnostics/v3-race-readiness-20261107.json`.
Setup log: `data/diagnostics/v3-setup-20261107.log`.
576 tests passed, with two NumPy-dependent skips in the main application environment;
Ruff lint and strict mypy passed. Focused coverage checks V3 asset selection, explicit
model loading, runtime incompatibility, reported variant mismatch and worker reuse.
An additional native V3 generation cancellation check also completed: cancellation
acknowledgment took 1.78 ms in that sample, the next request used the same worker,
and shutdown closed it. This is one sampled generation interruption, not a bound
for every playback/cancellation phase. Artifact:
`data/diagnostics/v3-cancellation-20261107.json`.

## Default-voice memory placement (2026-11-08)

The source worker now loads on CPU first for both BF16 and FP32 CUDA speech.
Only T3, the decoder's flow/vocoder, decoder buffers and cached conditioning move
to CUDA. The reference speech tokenizer, speaker encoder and voice encoder remain
in system RAM throughout the session. Active synthesis weights remain resident
on the GPU; there is no per-reply weight transfer or model reload.

The pinned V2/V3 decoder normally derives its device from the reference tokenizer.
A worker-local adapter instead follows active flow weights, checks the expected
module layout, and rejects reference-audio encoding. Unsupported layouts fail
visibly with `tts_default_voice_layout_unsupported`; they do not silently change
voice or computation policy. Existing OOM CPU fallback still reloads the original
FP32 model when allowed. CPU speech uses the original placement path.

This applies automatically to both replies and announcements through their shared
worker. Restart the engineer/panel completely to load the updated source worker;
no new model download or configuration key is needed. Packaged applications need
the usual rebuild to include the new adapter. Qwen, STT, playback priorities,
selected output device and PTT WAV cues retain their existing behavior.

The updated V3 CUDA/BF16 worker measured **1582 MiB allocated / 1654 MiB reserved**
after warm-up, compared with the previous **2100 / 2190 MiB**: roughly **0.5 GiB**
less VRAM for speech. Readiness logs include `reference_modules_device = "cpu"`.
Allocator figures exclude CUDA contexts, drivers, other processes and Qwen.
Speech still uses additional system RAM; this optimizes VRAM rather than total RAM.

Native validation reports are kept in
`data/diagnostics/default-voice-ram-20261108/verification.json`. Both pinned V2 and
V3 generated five EN/TR replies and two automatic calls through one unchanged
CUDA/BF16 worker per session, without playing audio. Native V3 cancellation,
Turkish follow-up reuse and shutdown passed. The full host suite passed 582 tests
with two NumPy-dependent skips; Ruff and strict mypy passed.
The isolated comparison that led to this change is in
`data/diagnostics/low-vram-investigation-20261108/`; its six paired
English/Turkish outputs were byte-identical with no observed latency penalty.
Combined iRacing/Qwen contention and listening quality remain live acceptance
checks; the placement change does not establish that unwanted tails are fixed.

## Windows source setup

Use Python 3.12 and the normal editable application installation:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev,desktop]"
.\.venv\Scripts\python.exe scripts\setup_radio_tts.py
```

Setup creates `data/tts-chatterbox/runtime`, independently of the application,
ASR and old Piper environments. Auto chooses the CUDA package when an NVIDIA GPU
is detected, otherwise CPU. It downloads about 3.2 GB of model assets into
`data/components/tts/chatterbox/5bb1f6ee58e50c3b8d408bc82a6d3740c2db6e18`.
Every asset has an immutable URL, size and SHA-256 in component set 2. Cached files
are verified/reused; downloads can resume through the existing component installer.
The ignored `data/` tree, model weights and generated audio must not be committed.

Explicit choices:

```powershell
.\.venv\Scripts\python.exe scripts\setup_radio_tts.py --backend cuda
# CPU-only machines:
.\.venv\Scripts\python.exe scripts\setup_radio_tts.py --backend cpu
```

Windows dependency constraints are in `requirements/chatterbox-runtime.txt`.
The CPU runtime uses upstream's torch/torchaudio 2.6.0. CUDA uses **2.7.1+cu128**
to support RTX 50xx/Blackwell. This is a deliberate compatibility override of
Chatterbox 0.1.7's torch/torchaudio 2.6.0 metadata pins, not an upstream-supported
dependency combination. Setup installs explicit dependencies first, then only
the Chatterbox wheel with `--no-deps` so pip cannot downgrade CUDA. Consequently
`pip check` in the CUDA worker reports those two known pin mismatches.
Do not install Chatterbox into the ASR or application environment. The setup
recipe, not a plain `pip install chatterbox-tts`, is the Windows supported path
for this integration. Revalidate after changing package/PyTorch versions.

Live workers run with Hugging Face/Transformers offline mode. Missing files cause
a visible setup error, never a network download during a race or another voice.

## One shared backend and playback floor

Each running engineer session creates one persistent worker/model, shared by
`ConversationSpeaker` and the automatic `TextToSpeechEngine` facade.
The live radio scheduler still orders/drops/interrupts jobs by priority, reserves
the microphone floor, and invalidates old work on disconnect/session reset/mute.
A fixed, silent English warm-up runs before worker readiness so lazy CUDA/vocoder
initialization does not consume the first race request's freshness window.
The backend serializes requests too; inference and PortAudio run only inside the
worker, never on UI, telemetry or microphone-handling threads.

Before playback, conversational audio requires a fresh live telemetry stream in the
same session generation. `[conversation].max_snapshot_age_s` remains three seconds
for telemetry; `[conversation].reply_delivery_timeout_s` separately allows up to
15 seconds for queueing and speech preparation after model processing. Preparation
timeout cancels the worker request; expired, muted and interrupted replies never
enter delivered dialogue history. This prevents ordinary 3–6 second synthesis from
being discarded solely because the originally retrieved frame aged while live
telemetry kept updating. Values still describe the snapshot retrieved before
synthesis, not a measurement taken at the instant of playback.

Automatic calls retain their original intent deadlines, checked again after
synthesis. The panel displays `Preparing speech…` during generation/output preparation
and `Speaking…` only when playback is dispatched. A discarded or failed answer gives
a visible notice instead of appearing to have been spoken. The backend logs prepared
audio timing and playback dispatch separately, without recording audio or prompts.

Cancellation reaches the protocol loop while inference runs on a separate thread.
A worker-local PyTorch forward pre-hook checks the cancellation event between
module executions; output is checked/aborted between 20 ms chunks. Ordinary
interruptions retain the model. The parent waits for cancellation acknowledgment
before releasing the radio floor. A stuck cancel exceeding the shutdown bound
kills/disables the worker until a session restart rather than playing overlapping
audio or repeatedly loading another model. Individual GPU kernels cannot be
preempted, so cancellation latency is hardware/workload dependent.

Initialization/generation/protocol/output failures are logged and surfaced to
the terminal/panel. Telemetry and text replies continue. No failure silently
reverts to Piper, SAPI or a different voice.

## Configuration and migration

There is only one speech section now:

```toml
[tts]
enabled = true
adapter = "chatterbox"
python_path = "data/tts-chatterbox/runtime/Scripts/python.exe"
model_path = "data/components/tts/chatterbox/5bb1f6ee58e50c3b8d408bc82a6d3740c2db6e18"
device = "auto"               # auto, cuda or cpu
precision = "auto"            # BF16 T3 on supported CUDA; otherwise FP32
fallback_to_cpu = true
volume = 0.8                  # 0..1, shared across both speech paths
threads = 4
startup_timeout_s = 180.0
playback_timeout_s = 120.0
shutdown_timeout_s = 5.0
queue_capacity = 8
# output_device = 4
```

Remove old `[radio_tts]`, `voice`, `rate`, `length_scale` and per-language
ONNX paths from custom configs; the validator rejects them explicitly. Output
selection and volume in the panel now apply to announcements, replies and PTT cues.
A device rejecting native 24 kHz speech is resampled to its reported preferred
sample rate, not silently routed to another device.

Auto prefers a usable CUDA device. If unavailable and fallback is allowed, CPU is
reported visibly with a latency warning. Set `device = "cpu"` deliberately on CPU
machines, or disable fallback to require GPU. AMD/ROCm/DirectML support is not
implemented; AMD machines can use CPU. No real-time CPU guarantee is made.
`precision = "auto"` selects native BF16 T3 on supported CUDA (including this
RTX 5080). An older CUDA GPU uses FP32 with a visible notice; CPU always uses
FP32. `precision = "fp32"` is the explicit rollback. `precision = "bf16"` requires
native BF16 when running on CUDA and fails visibly if unsupported; an allowed
CPU fallback still uses original FP32 weights and the existing latency warning.
This setting never changes Qwen's precision or the audio decoder. Effective
T3/device/decoder precision is logged in `tts_ready` and printed by the smoke tool.
For BF16 the loader stages/casts T3 and its floating conditioning on CPU before
CUDA placement, preserving integer token IDs and avoiding a full-FP32 CUDA
staging allocation. Generation/playback failures never silently change precision.

Automatic templates retain their existing English policy wording; this migration
does not add translation. The shared synthesizer accepts English or Turkish
language metadata; conversation language selection is unchanged.

## Verification without a race

Synthesis only:

```powershell
.\.venv\Scripts\python.exe -m race_engineer test-radio --config config\default.toml --language en --no-playback
.\.venv\Scripts\python.exe -m race_engineer test-radio --config config\default.toml --language tr --no-playback
```

Audible checks (selected/default output; each standalone command is its own session):

```powershell
.\.venv\Scripts\python.exe -m race_engineer test-radio --config config\default.toml --language en
.\.venv\Scripts\python.exe -m race_engineer test-radio --config config\default.toml --language tr
.\.venv\Scripts\python.exe -m race_engineer test-tts --config config\default.toml --text "Green flag."
.\.venv\Scripts\python.exe -m race_engineer test-tts --config config\default.toml --language tr --text "Sarı bayrak."
.\.venv\Scripts\python.exe scripts\smoke_chatterbox.py --no-playback
.\.venv\Scripts\python.exe scripts\smoke_chatterbox.py
```

The smoke script reuses one backend for EN/TR requests and automatic calls and
prints synthesis/playback timings. Then test the panel's **Test voice**, actual
PTT cues, mute and Stop. During a race verify critical calls interrupt answers,
output follows the selected headset, and stale/cancelled audio never plays later.
Assess naturalness with a human listener; successful synthesis is not quality proof.
GPU VRAM contention with Qwen/iRacing and CPU/game-load latency remain acceptance work.

For frozen-worker checks see [distribution](distribution.md). Existing executables
must be rebuilt; updating source does not replace an already built Piper worker.

## Isolated T3 precision comparison

The experimental benchmark leaves the production worker, `[tts]` configuration,
Qwen, model weights on disk and PTT cues untouched. It loads the same pinned
default-voice checkpoint on CPU first, casts **only T3 and its floating-point
conditioning** to BF16, then transfers the model to CUDA. The audio decoder and
its conditioning stay FP32; token IDs stay integers. CPU-first placement also
avoids initially putting the full FP32 T3 weights on the GPU. There is no
autocast-only claim of weight savings, package patch or automatic promotion.

Close the engineer and other GPU workloads for the isolated measurement. Run
from the repository root using the normal application Python; the controller
launches the existing isolated TTS Python for each candidate, **sequentially**:

```powershell
.\.venv\Scripts\python.exe scripts\benchmark_chatterbox_precision.py
# Short pilot: one pass over ten authored English/Turkish phrases per candidate.
.\.venv\Scripts\python.exe scripts\benchmark_chatterbox_precision.py --repeats 1
```

Default candidates are FP32 and T3-BF16, three repeats of ten phrases each.
An optional FP16 alternative can be selected explicitly:

```powershell
.\.venv\Scripts\python.exe scripts\benchmark_chatterbox_precision.py --candidate fp32 --candidate fp16
```

This requires a CUDA-capable TTS runtime. BF16 additionally requires native GPU
BF16 support. Unsupported hardware, missing/corrupt cached models, generation
errors and timeouts produce visible failures, not another precision or CPU
fallback. No models are downloaded and no audio is played automatically.

Each run creates a **new** ignored directory under
`data/benchmarks/chatterbox-precision-<UTC timestamp>`; `--output` can name a new
subdirectory of `data/benchmarks`. Existing directories/audio are never replaced.
It contains `report.json` plus one directory per precision with `candidate.json`,
`worker.log` and 24 kHz mono PCM WAVs. Samples contain only the authored phrases,
not microphone recordings. Partial measurements survive a worker failure;
timeouts and Ctrl+C close only the benchmark's owned child.

The report includes verified model hashes, source/config hashes, package and
GPU versions, precision inventories, matched request seeds/order, load/ready
times, warm memory, startup/request CUDA allocator peaks, retained allocation
growth, per-request synthesis time, p50/p95, audio duration and real-time factor
(generation seconds per audio second). Whole-device `nvidia-smi` observations
are samples, not peaks or reliable per-process attribution. Same seeds cannot
guarantee identical speech or deterministic kernels across precisions. Compare
duration and real-time factor as well as milliseconds: a shorter/truncated reply
must not be mistaken for a faster equivalent reply.

Listen to matching filenames in `fp32` and `bf16`, checking English/Turkish
pronunciation, numbers, missing/repeated words, silence, artifacts and consistent
voice identity. The report's paired-audio list makes those comparisons explicit.
Successful synthesis does **not** mean quality passed. Any candidate needs human
listening, production cancellation/playback tests and a combined Qwen/iRacing
trial for production acceptance. The comparison itself never changes live
settings. Following the driver's acceptance of the candidate, the October 28
integration uses BF16-capable `precision = "auto"` in the shared live backend;
FP32 rollback remains available.

### First precision comparison (2026-10-27)

The RTX 5080 / PyTorch 2.7.1+cu128 run completed all 30 measured requests per
candidate (15 EN, 15 TR), in separate processes, using Chatterbox 0.1.7 and the
same checksum-verified V2/default-voice files. Results are in the ignored
`data/benchmarks/chatterbox-precision-20261026T223525Z` directory (UTC identifier).

| Measurement | FP32 | T3-BF16 / FP32 decoder |
| --- | ---: | ---: |
| Warm allocated tensors | 3121.7 MiB | 2100.0 MiB |
| Warm allocator reservation | 3272 MiB | 2332 MiB |
| Peak allocated, including startup | 3254.2 MiB | 2225.9 MiB |
| Median synthesis | 2.193 s | 2.358 s |
| p95 synthesis (nearest rank, 30 samples) | 2.953 s | 3.100 s |
| Median real-time factor | 0.515 | 0.584 |
| Allocated growth after warm-up | 0 MiB | 0 MiB |

This supports about **1 GiB less tensor memory**, not a speed-up: median latency
increased about 165 ms (7.5%) in this small run. The BF16 conditioning inventory
retained integer token IDs and FP32 decoder conditioning. Every generated WAV was
nonzero, but pronunciation, completeness and naturalness have **not** been
accepted by a human listener. Zero allocation growth over this short run is not a
long-session leak guarantee. No gameplay, playback, interruption or production
integration acceptance is implied by that benchmark. BF16 was subsequently
enabled on supported CUDA GPUs after the driver's acceptance, not automatically
by the benchmark.

### Production BF16 integration (2026-10-28)

On this RTX 5080, the normal shared backend reported CUDA / T3-BF16 /
decoder-FP32. Real EN/TR conversational and automatic synthesis completed using
one unchanged worker. A separate real live-radio diagnostic interrupted both
generation and audible playback, completed subsequent critical announcements
without reloading, then stopped during another generation and closed the worker.
Observed cancellation cleanup was about 1 ms during generation and 23 ms during
playback; these are idle-host samples, not a latency guarantee. Existing synthetic
mute/reset/shutdown, output selection and PTT-cue tests also passed.

This enables the accepted memory-saving candidate; it does not claim a speed-up
or eliminate the existing freshness deadline. Game/Qwen contention, long races
and rebuilt frozen executables remain separate checks. Existing executables must
be rebuilt for the new precision argument/ready handshake.

## Migration verification (2026-10-25)

On this Windows/RTX 5080 host, the isolated 0.1.7 runtime with PyTorch/torchaudio
2.7.1+cu128 successfully loaded the pinned V2 cache and default conditioning.
All five model-file checksums passed. Actual EN/TR synthesis completed without
playback, then EN/TR replies and announcements completed with audio on both the
system-default headset and its explicitly selected WASAPI output, one unchanged
worker per session. No reference recording was supplied.

The cold first generation initially took 10.1 s, so the worker now performs its
silent startup warm-up before reporting ready. Final warmed WASAPI conversational
samples synthesized in 1.87 s (English) and 1.66 s (Turkish), before playback.
Actual priority interruption during generation and playback returned interrupted
in approximately 1.4 ms and 5.0 ms on an idle machine; the model was not reloaded,
subsequent critical announcements completed and both unchanged PTT WAV cues played.
These are individual synthetic checks, not racing-load latency percentiles.

Forced CPU inference in the same 2.7.1 CUDA-capable runtime completed EN/TR with
one worker, but took about 6.57 s and 16.09 s for short checks (startup/warm-up
19.99 s). This is not real-time CPU acceptance and does not validate the separate
2.6.0 CPU release build. CPU speech can exceed the existing freshness/deadline
bounds and be discarded. Use CUDA for the next live trial on this host.

The main application's lightweight test environment passed 488 tests, with two
NumPy-dependent worker tests intentionally skipped there. All 27 Chatterbox tests
and the related scheduler/PTT tests passed in the isolated TTS environment (42
tests, no skips). Ruff and mypy passed. Frozen executables were not rebuilt or
inference-tested in this migration; clean-machine packaging, combined Qwen/iRacing
GPU contention, human naturalness/number pronunciation and actual in-race acceptance
remain open. No benchmark here proves universal conversational correctness.

Upstream references: [pinned package/API examples](https://pypi.org/project/chatterbox-tts/0.1.7/),
[pinned model assets](https://huggingface.co/ResembleAI/chatterbox/tree/5bb1f6ee58e50c3b8d408bc82a6d3740c2db6e18),
[PyTorch Blackwell support](https://pytorch.org/blog/pytorch-2-7/).
