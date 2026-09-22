# Race Engineer

Race Engineer is a local-first, simulator-adaptable foundation for real-time racing
assistance. Its central policy engine decides what is worth communicating; telemetry,
language generation, and speech synthesis remain replaceable adapters.

This repository contains the M0 architecture baseline, the M1 iRacing telemetry reader,
the deterministic M2 policy slice, and the local M3 speech path:

- versioned domain contracts and adapter protocols;
- validated TOML configuration with environment overrides;
- structured console or JSON logging;
- deterministic, versioned replay fixtures;
- a headless async pipeline and fake implementations;
- automated contract, configuration, fixture, and pipeline tests;
- a reconnecting iRacing shared-memory reader with atomic snapshots, normalization,
  event derivation, and fixture-compatible recording;
- deterministic race-context features, strict rules, scheduling constraints, and
  traceable policy decisions;
- fact-bound English wording with deterministic replay and an explicit fallback boundary;
  and
- offline Windows speech with bounded priority queueing, expiry, interruption, and
  failure isolation; and
- local English/Turkish conversations over paused replay data, using Qwen to
  interpret questions and follow-ups with fact-bound bilingual replies; and
- local Qwen3-ASR push-to-talk and Piper spoken replies with text fallback; and
- a native Windows control panel for the live radio, audio checks, press-to-bind
  keyboard/mouse/wheel PTT, mute and SQLite-backed driver preferences; and
- versioned local SQLite driver profiles with validated, auditable explicit preferences;
  and
- privacy-safe durable policy-decision and automatic-radio outcome history.

Model-backed language generation, higher-quality speech adapters, adaptive policies, and
advanced desktop controls remain later work.

The conversational prototype supports text chat and local Qwen3-ASR push-to-talk, with
bilingual Piper spoken replies over replay data and live iRacing context. The combined
`voice-iracing` command coordinates automatic calls and questions. The driver accepted
the initial live prototype as sufficient to continue; quality upgrades and quantitative
validation remain open. See the [live run guide](docs/live-conversation.md) and
[current roadmap / next task](docs/roadmap.md).

Conversation model identity and llama.cpp execution are separately configurable. The panel
and standalone launcher share the same bounded runtime lifecycle and one CPU fallback.
CPU remains the installed default; CUDA/Vulkan packages and UI selection are later measured
work, not current support claims. See the [conversation setup](docs/conversation.md#local-model-setup).

faster-whisper is reserved as a future performance alternative. See the
[speech-input setup](docs/speech-to-text.md),
[spoken-reply setup and voice license notices](docs/conversational-speech.md),
[conversation setup and demo](docs/conversation.md) and
[prototype decisions](docs/architecture.md#conversational-prototype).

## Requirements

- Python 3.12 or newer
- Git
- Windows with an installed SAPI voice for live iRacing speech

## Development setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev,desktop]"
pytest
ruff check .
mypy
```

## Desktop control panel

With the existing local speech/model installations available, double-click
`start-panel.cmd` or run `python -m race_engineer.ui` from the repository root.
The panel starts/reuses the local conversation server when you click Start. It does not
start the microphone just by opening the window. See the [control panel guide](docs/control-panel.md)
for audio checks, steering-wheel/controller binding, preferences, tray controls and
current limitations.

Do not run a separate live engineer CLI alongside the panel.

Inspect or update the default local driver profile:

```powershell
race-engineer profile --config config/default.toml show
race-engineer profile --config config/default.toml rename "Furkan"
race-engineer profile --config config/default.toml set reply_language tr
```

The panel and these commands use the same default profile. See the
[driver-profile storage guide](docs/driver-profile.md) for persistence, migration and reset
behavior.

Inspect recent content-free decision and automatic-radio totals:

```powershell
.\.venv\Scripts\python.exe -m race_engineer history --config config\default.toml recent
```

See the [decision-history guide](docs/decision-history.md) for the exact data boundary and
30-day default retention.

## Command-line tools

Validate the default configuration or inspect a replay fixture:

```powershell
race-engineer validate-config --config config/default.toml
race-engineer inspect-fixture fixtures/synthetic/green_flag
race-engineer replay-policy fixtures/synthetic/m2_green_flag `
  --config config/default.toml
race-engineer replay-language fixtures/synthetic/m3_green_flag `
  --config config/default.toml
```

List the installed local voices, then run an audible radio check:

```powershell
race-engineer list-tts-voices --config config/default.toml
race-engineer test-tts --config config/default.toml
```

With iRacing running, print normalized frames until interrupted:

```powershell
race-engineer read-iracing --config config/default.toml
```

Record 600 normalized frames plus their events, policy decisions, speech intents, and
generated utterances into a new replayable session directory. Approved, non-expired
utterances are also spoken through the configured local voice:

```powershell
race-engineer read-iracing --config config/default.toml --limit 600 `
  --output recordings/my-session
```

The live reader intentionally does not retain driver names or arbitrary raw SDK data.

## Design constraints

- The domain never imports a simulator SDK, Qt, an LLM SDK, or a TTS SDK.
- Serialized contracts and fixtures carry explicit schema versions.
- Internal measurements use SI units.
- Wall-clock timestamps are timezone-aware UTC values; session time is monotonic seconds.
- Queues are bounded and expired work is designed to be discarded rather than accumulated.
- Driver data and logs remain local unless explicitly exported.

See [docs/architecture.md](docs/architecture.md) and
[docs/fixture-format.md](docs/fixture-format.md) for the M0 decisions, and
[docs/strict-policy.md](docs/strict-policy.md) for the M2 policy behavior. The M3 wording
boundary is described in [docs/language-generation.md](docs/language-generation.md), and
[docs/text-to-speech.md](docs/text-to-speech.md) covers local playback. The architecture
also records the historical Jev-assisted ranking proposal. The next increment is the
[local context engine and portable inference plan](docs/context-engine-implementation-plan.md),
covering smaller bilingual judges, optional CPU/GPU execution, and local shadow ranking.
It extends the [original implementation
plan](docs/personalized-ai-race-engineer-implementation-plan.docx); hosted Jev is deferred.
These next-increment capabilities are planned, not yet implemented. Open defects are tracked in
[docs/known-issues.md](docs/known-issues.md).
