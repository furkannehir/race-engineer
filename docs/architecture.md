# Architecture

Pitward is a local race engineer for iRacing on Windows. It reads telemetry, makes
automatic radio calls, and answers push-to-talk questions in English and Turkish. Speech
recognition, the language model and speech synthesis all run on the driver's PC. The only
network use is the explicit component download and the optional update check.

## Processes

| Process | What it runs | Notes |
|---|---|---|
| Panel (`Pitward.exe` or `python -m race_engineer.ui`) | Qt control panel, telemetry reader, policy, radio scheduler, dialogue engine | One per machine; a lock prevents a second panel from the same checkout |
| `llama-server` (llama.cpp b12088) | Ministral 3 3B Instruct 2512, Q4_K_M | Loopback HTTP on `127.0.0.1:8092`, owned and stopped by the panel |
| STT worker | Qwen3-ASR 0.6B (Transformers) | Isolated Python runtime on CPU, private stdin/stdout pipe |
| TTS worker | Chatterbox Multilingual 0.1.7, built-in voice | Isolated PyTorch runtime, CUDA preferred, private pipe |

The two speech workers have separate runtimes because they pin different PyTorch and
Transformers versions. The panel never runs inference on the UI, telemetry or microphone
threads.

## Flow

Automatic calls:

```text
iRacing shared memory -> normalizer -> event deriver -> race context
  -> strict policy -> scheduler -> deterministic wording -> LiveRadio -> Chatterbox
```

Driver questions:

```text
PTT capture -> Qwen3-ASR -> Ministral picks one read-only tool
  -> Python computes the facts -> Ministral writes a short reply -> reply check
  -> LiveRadio slot -> tool re-run on fresh telemetry -> Chatterbox -> playback
```

## Package boundaries

- `core` owns contracts and interfaces and imports no adapter.
- `telemetry`, `policy`, `language`, `stt` and `tts` implement core interfaces.
- `intelligence` owns the dialogue engine, the race capabilities and reply validation.
- `conversation` owns the llama-server runtime manager (`runtime.py`).
- `memory` owns SQLite persistence and accepts only versioned core commands.
- `application` composes the pieces and does not import vendor SDKs.
- `ui` is the only package that imports Qt.
- Simulator SDK types are normalized before they reach `core` or `policy`.

## Data conventions

- Serialized contracts and fixtures carry a fixed `schema_version`.
- Internal measurements use SI units, with the unit in the field name.
- Wall-clock times are timezone-aware UTC. `session_time_s` is monotonic session time and
  drives deterministic replay.
- Missing telemetry is `None` plus an explicit capability list. Values are never invented.
- TOML is the configuration format. Unknown keys fail validation. `RACE_ENGINEER_*`
  environment variables override selected values.

## Telemetry

`PyIrSdkSource` (pyirsdk 1.3.6) is the only module that imports the SDK. It waits for
iRacing, reconnects after a disconnect, freezes the latest variable buffer for each read,
and drops incomplete, duplicate and out-of-order samples. A session change resets ordering.
Replay samples are dropped unless configured.

It samples at 10 Hz against monotonic deadlines and skips missed slots rather than
bursting stale reads. Slow reads (`slow_read_warning_s`, 50 ms) and frame gaps
(`sample_gap_warning_s`, 250 ms) are logged.

Selected values: session identity and phase, flags, player position, current and completed
laps, last and best lap times, lap limit and remaining laps, session clock, speed, fuel,
pit road, distance to the cars ahead and behind, opponent positions and completed laps, and
same-lap race gaps from `CarIdxF2Time`. Opponent `gap_to_player_s` is signed: negative is
ahead. It is set only when both cars are on the same lap in a race.

A blue flag in a race needs a valid player position and an active opponent with more
completed laps. Otherwise it fails closed. Outside races the raw blue bit is kept.

Only car indices, numeric user IDs and pace-car markers are kept from driver metadata.
Names and the raw session YAML never enter contracts, fixtures or recordings.

## Automatic calls

`DefaultRaceContextBuilder` keeps bounded session state: recent events, stint lap, rolling
fuel use per lap, nearest gaps, and battle state (`attacking`, `defending`, `sandwiched`,
`contested`, `clear`).

The strict policy approves session-phase and flag calls and, by default, position changes.
Pit transitions are silent by default. Caution and red, yellow and black flags are
critical: they use fixed templates and may interrupt. The scheduler applies priority,
expiry, de-duplication, cooldowns (15 s routine, 5 s important, none for critical) and one
call per frame. Every candidate produces a `PolicyDecision` with a reason: `approved`,
`disabled`, `expired`, `duplicate`, `cooldown` or `superseded`.

The `deterministic` language adapter turns approved intents into short English templates.
Automatic calls are English only; replies to questions follow the driver's language. The
adapter rejects unknown facts, unsupported languages and over-long text rather than
improvising.

## Dialogue

The dialogue engine (`intelligence/ministral_dialogue.py`) makes two model calls per
question:

1. **Tool selection.** Ministral picks one read-only tool, or `reply_to_driver` for a social
   reply. Tools cover position and classification, gaps ahead and behind, relative pace and
   catch time, last and best lap times, race progress, fuel range, fuel to finish, history
   queries and recall of earlier radio readings.
2. **Wording.** Python runs the tool against telemetry memory and returns facts. Ministral
   writes a short reply from them.

The reply check binds every number, unit and verdict in the draft to the tool's facts. One
repair is allowed. Python owns all race arithmetic; the model never calculates.

When `LiveRadio` grants the reply its slot, the tool runs again on current telemetry, with
no extra model call. A changed session, changed availability or a contradicted verdict
cancels the reply. Only replies that actually played enter dialogue history, which holds up
to `history_turns` exchanges and is cleared on reconnect or language change.

Race capabilities (`intelligence/capabilities.py`) are plain Python over normalized
telemetry: classification, fuel range (fuel divided by rolling burn), fuel to finish (with a
one-lap reserve), 30 s position change, gaps, 10 s relative-pace trend and catch time, lap
times and race progress. Pit-loss and rejoin projections are declared but always return an
explicit unavailable reason.

The older Qwen and LFM planners are still in the code as explicit rollback configs
(`config/qwen-legacy.toml`, `config/lfm-legacy.toml`). They are never selected
automatically.

## Radio scheduling and freshness

`LiveRadio` is the single queue for automatic calls and replies. It orders work by
priority, drops expired work, and grants the microphone a capture slot.

- Replies have priority 50: important and critical calls go first, routine calls after.
- A PTT press while the radio is busy is refused.
- Critical calls cancel capture or lower-priority playback, wait for cleanup, then play.
- Telemetry must be fresh (`max_snapshot_age_s`, 3 s) and in the same session generation at
  playback dispatch. Replies get a separate 15 s budget (`reply_delivery_timeout_s`) for
  queueing and synthesis.
- Disconnects, session changes and stale telemetry invalidate pending and playing speech. A
  watchdog checks every 100 ms.

## Speech

**Input.** PTT capture is memory-only, 0.2–15 s, with an energy gate against silence.
Bindings can be keyboard keys, mouse buttons, or digital wheel and controller buttons or
hats. Analog axes are never read. Opening and closing WAV cues play around capture and are
excluded from the clip.

**Output.** One persistent Chatterbox worker serves replies and automatic calls. It uses the
built-in voice for both languages, with no reference audio. On CUDA, T3 runs in BF16 and the
decoder in FP32. The cloning modules stay in system RAM. A silent warm-up runs before
readiness. Cancellation is checked between modules and between 20 ms output chunks, and the
model is kept across interruptions. A stuck cancel disables speech until restart.
Synthesis failures leave telemetry and text running and never switch to another voice.

## Local data

`data/race_engineer.sqlite3` (or `%LOCALAPPDATA%\Pitward` when installed) holds:

- **Driver profile (schema v1).** One default profile with three driver-scoped settings:
  `reply_language`, `announce_position_changes`, `announce_pit_transitions`. Each change is
  stored as an idempotent `PreferenceCommand` for provenance. Critical calls can't be
  disabled.
- **Decision history (schema v2).** Content-free policy decisions and automatic-radio
  outcomes, kept 30 days by default (`[history] retention_days`). No telemetry values,
  spoken text, audio or names. Writes are failure-isolated from the live loop.

Machine-local PTT, device and volume settings live in `data/control-panel.json`.

## Privacy and logs

Microphone audio is never written to disk. Diagnostic logs (`logs/control-panel.jsonl`, or
`%LOCALAPPDATA%\Pitward\logs`) record timing, routing and reason codes per radio attempt.
`[privacy] record_radio_text` adds transcripts and replies; `record_raw_telemetry` adds
evidence values. Full model prompts are never logged.

## Key decisions

**Local only.** A race engineer must work offline, with no account and no telemetry
leaving the PC. Every model runs locally, and the only network calls are the
user-started component download and the optional update check.

**Ministral 3 3B for dialogue.** It replaced LFM in 0.2.1. In the November 2026
comparison against LFM, Qwen3-4B and Qwen3.5-2B, it had the best Turkish tool selection
(18/20, English also 18/20) and the fastest typed replies (about 0.1 s median on CUDA). It
costs more GPU memory than LFM (about 2.9 GiB) and still makes wording mistakes. It is
Apache-2.0. The model chooses tools and words; Python owns the facts.

**Chatterbox Multilingual for speech.** It is the only permissively licensed option found
with acceptable English and Turkish in one voice. Its costs are GPU memory, a CPU path that
isn't real-time, and occasional tails after a sentence (TTS-002).

**Qwen3-ASR for recognition.** It handles English and Turkish in one 0.6B model on CPU.
faster-whisper and whisper.cpp remain possible alternatives if accuracy or resource use
requires a change.

**Deterministic critical calls.** Flags and safety calls use fixed templates and never wait
on a model.
