# Local speech: Chatterbox Multilingual

All engineer speech now uses **Chatterbox TTS 0.1.7**, its **500M multilingual V2**
model and built-in default voice. Both English (`en`) and Turkish (`tr`) use
the same model and conditioning. There is no reference clip, uploaded voice, hosted
TTS API, SAPI process or Piper fallback. PTT open/close WAV cues are unchanged.

The pinned package's `ChatterboxMultilingualTTS.from_local(path, device=...)`
loads `ve.pt`, `t3_mtl23ls_v2.safetensors`, `s3gen.pt` and the tokenizer.
`conds.pt` supplies built-in conditioning. Pitward requires it and verifies that
both languages are supported before accepting speech. Generation calls
`model.generate(text, language_id=language)` without `audio_prompt_path`.
The package's default style controls and watermarking are retained; no emotion or
custom-voice controls are exposed. This deliberately uses the V2 files expected by
0.1.7, not a newer upstream README's V3 recipe or English-only Turbo.

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

Before opening an output stream, prepared conversational audio passes the existing
freshness gate. Automatic calls also recheck their intent deadline after synthesis.
Slow synthesis is not permission to play stale information. The existing
three-second conversational freshness default is unchanged; CPU or long GPU
generation may miss it. Do not increase it merely to hide latency.

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
