# Development

How to set up, run and test Pitward from source on Windows. For the design, see
[architecture.md](architecture.md). For packaged builds and releases, see
[distribution.md](distribution.md).

Run every command from the repository root. Everything under `data/`, `logs/` and
`recordings/` is git-ignored. Model weights, runtimes and generated audio must never be
committed.

## Requirements

- Windows 10 or 11, x64
- Python 3.12
- An NVIDIA GPU for the source default. CPU works, but replies and speech can miss
  racing deadlines.
- About 10 GB free for models and runtimes

## Application environment and checks

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev,desktop]"
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m mypy
```

CI runs the same three checks. Two NumPy-dependent worker tests are skipped in the
application environment; they run in the speech runtimes.

## Local models and runtimes

Each setup script verifies cached files before downloading, pins every file by URL and
sha256, and can resume.

| Part | Command | Installs into |
|---|---|---|
| Dialogue (Ministral + llama.cpp) | `scripts\setup_ministral.py --backend gpu` | `data/lfm3-prototype/` (runtimes), `data/dialogue-comparison/` (model) |
| Speech recognition (Qwen3-ASR) | `scripts\setup_stt.py` | `data/stt-prototype/` (~1.6 GB) |
| Speech output (Chatterbox) | `scripts\setup_radio_tts.py` | `data/tts-chatterbox/runtime`, `data/components/tts/chatterbox/<revision>` (~3.2 GB) |

Run each with `.\.venv\Scripts\python.exe`. The `lfm3-prototype` and
`dialogue-comparison` folder names are historical; they hold the Ministral runtime and
model.

**Dialogue.** `--backend cpu` installs the CPU llama.cpp build for an explicit CPU test.
The source default (`config/default.toml`) is CUDA with no silent CPU fallback.

**Speech recognition.** Setup installs a separate CPU PyTorch runtime with Transformers
5.17.0 and the Qwen3-ASR-0.6B-hf model at revision
`7f1569a48a89f3e3f4dc3a5c9d28bddd903bc76c`. Recognition runs offline.

**Speech output.** `--backend auto` (default) picks CUDA when an NVIDIA GPU is present.
Use `--backend cuda` or `--backend cpu` to choose. CUDA uses torch 2.7.1+cu128 for RTX 50xx
support. This is a deliberate override of Chatterbox 0.1.7's 2.6.0 pins, so `pip check`
in that runtime reports two known mismatches. Never install Chatterbox into the
application or ASR environment. `--model-variant v3` installs the V3 checkpoint into
`data/tts-chatterbox-v3/runtime`.

## Run the panel

```powershell
.\start-panel.cmd
```

or `.\.venv\Scripts\python.exe -m race_engineer.ui`. The panel starts and owns the
llama.cpp server on `127.0.0.1:8092`, so don't start one yourself. Opening the window
doesn't start telemetry, inference, capture or playback.

First shakedown:

1. Pick the microphone and output. The output applies to all speech and the PTT cues.
2. **Test mic** reports a level without saving audio. **Test voice** plays a radio check.
3. **Change** beside PTT, then press the control to bind: a key, a mouse button, or a
   digital wheel or controller button or hat. Analog axes are ignored.
4. **Start engineer**, then wait for iRacing and speech readiness. Start stationary in a
   practice session.
5. Hold PTT, wait for the opening cue to finish, speak, release.

**Mute all audio includes critical calls.** Device, PTT, language and volume controls are
locked while running. Closing a running panel stops the session and the server it
started.

Layout preview with no hardware: `python -m race_engineer.ui --preview`.
`scripts\check_control_panel.py` renders the real widgets offscreen and compares them
with the reference mockup.

Don't run `voice-iracing`, `read-iracing` or `voice-replay` alongside the panel: separate
processes can't share the radio and produce overlapping calls.

## Terminal workflows

Dialogue without iRacing, microphone or speech (needs a running model server):

```powershell
.\.venv\Scripts\python.exe scripts\start_conversation_model.py
# second terminal:
.\.venv\Scripts\python.exe -m race_engineer intelligence-replay `
  --fixture fixtures\synthetic\ministral-lap-progress --question "What's the gap behind?" --json
```

Production-path smoke test, with its own server on port 18201 and no audio (add
`--synthesize` to run replies through Chatterbox without playback):

```powershell
.\.venv\Scripts\python.exe scripts\smoke_ministral_live.py --backend gpu
```

Dialogue eval (reports go under `data/diagnostics/`):

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_ministral_dialogue.py `
  --dataset fixtures\intelligence-evaluation\ministral-followups.json --repeats 2 `
  --output data\diagnostics\ministral-check.json
```

Live terminal session (instead of the panel), with the model server running:

```powershell
.\.venv\Scripts\python.exe -m race_engineer voice-iracing --config config\default.toml
```

Options: `--device N` and `--output-device N` (from `list-input-devices` and
`list-output-devices`), `--ptt-key RCTRL` (F1–F24, SPACE, RCTRL, RALT), `--language en|tr`,
`--text-only`, `--limit N` frames, and `--output recordings\<new-folder>` to record frames,
events, decisions, intents and automatic utterances. Questions, replies and audio are
never recorded.

Speech checks:

```powershell
.\.venv\Scripts\python.exe -m race_engineer test-radio --config config\default.toml --language en
.\.venv\Scripts\python.exe -m race_engineer test-radio --config config\default.toml --language tr --no-playback
.\.venv\Scripts\python.exe -m race_engineer test-tts --config config\default.toml --text "Green flag."
.\.venv\Scripts\python.exe scripts\smoke_chatterbox.py --no-playback
.\.venv\Scripts\python.exe -m race_engineer transcribe-wav path\to\question.wav --config config\default.toml
```

`transcribe-wav` takes PCM16 WAV at 16, 44.1 or 48 kHz and needs only the ASR worker.

Telemetry and policy:

```powershell
.\.venv\Scripts\python.exe -m race_engineer read-iracing --config config\default.toml
.\.venv\Scripts\python.exe -m race_engineer read-iracing --config config\default.toml --limit 600 --output recordings\my-session
.\.venv\Scripts\python.exe -m race_engineer replay-policy fixtures\synthetic\m2_green_flag --config config\default.toml
.\.venv\Scripts\python.exe -m race_engineer replay-language fixtures\synthetic\m3_green_flag --config config\default.toml
.\.venv\Scripts\python.exe -m race_engineer validate-config --config config\default.toml
```

Recordings never overwrite an existing folder. Fixture layout is in
[fixture-format.md](fixture-format.md).

Driver profile and history:

```powershell
.\.venv\Scripts\python.exe -m race_engineer profile --config config\default.toml show
.\.venv\Scripts\python.exe -m race_engineer profile --config config\default.toml set reply_language tr
.\.venv\Scripts\python.exe -m race_engineer profile --config config\default.toml reset
.\.venv\Scripts\python.exe -m race_engineer history --config config\default.toml recent --limit 5
.\.venv\Scripts\python.exe -m race_engineer history --config config\default.toml prune
```

## Configuration

`config/default.toml` is the source default. Paths are relative to the working directory.

- `[conversation]`: `timeout_s`, `history_turns` (0–12), `max_snapshot_age_s` (telemetry
  freshness, 3 s), `reply_delivery_timeout_s` (queue and synthesis budget, 15 s).
- `[conversation.runtime]`: `mode` (`gpu` or `cpu`), executable and model paths,
  `context_size`, `fallback_to_cpu`.
- `[stt]`: `language` (`auto`, `en`, `tr`), `ptt_key`, `max_capture_s`, `sample_rate_hz`.
  Some WASAPI devices need `48000`; the worker resamples to 16 kHz. If quiet speech is
  dropped, raise the input gain first, then lower `silence_threshold_dbfs`.
- `[tts]`: `device` (`auto`, `cuda`, `cpu`), `precision` (`auto` gives BF16 T3 on
  supported CUDA; `fp32` is the rollback), `fallback_to_cpu`, `volume`, `model_variant`
  (`v2` or `v3`), `output_device`. Old `[radio_tts]` sections are rejected.
- `[history]`: `enabled`, `retention_days` (30).
- `[privacy]`: `record_radio_text` saves transcripts and replies in the local logs;
  `record_raw_telemetry` adds evidence values. Microphone audio and full prompts are
  never logged.

Legacy dialogue rollbacks: `.\start-panel.cmd --config config\lfm-legacy.toml` or
`config\qwen-legacy.toml`. Their assets must already exist.

## Diagnostics

Panel logs are in `logs/control-panel.jsonl` (2 MB plus two rotated files). Each radio
attempt has a `run_id` and `trace_id` linking capture, transcription, tool selection,
model timing, refreshed facts, queue wait, synthesis and the final outcome. The TTS
worker logs its device, precision and allocator memory at readiness.

Common problems:

- **PTT returns to ready without an answer:** the notice says whether PTT was released
  before the opening cue ended, the stream delivered no audio, or capture failed. Check
  the device and Windows microphone permissions.
- **"Speech unavailable":** the TTS worker failed to start. The `tts_*` reason code in
  the log says why; telemetry and text replies keep running.
- **Replies are slow or expire:** check the `compute` mode in the log. CPU dialogue and CPU
  speech can both exceed the delivery budget.
