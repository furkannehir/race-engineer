# M0 architecture baseline

## Decision status

This document records the decisions implemented by M0. Changes to a compatibility
boundary require a schema-version change and a test demonstrating migration or rejection.

## Runtime shape

The application is a headless Python 3.12+ modular monolith built around bounded asyncio
work. The logical flow is:

```text
TelemetryAdapter
  -> EventDeriver
  -> RaceContextBuilder
  -> PolicyStrategy
  -> LanguageGenerator
  -> TextToSpeechEngine
```

These are in-process interfaces, not network services. Model inference may later move to a
child process without changing the domain contracts.

## Dependency rules

- `core` owns contracts and interfaces and imports no adapter implementation.
- `telemetry`, `policy`, `language`, and `tts` implement core interfaces.
- `application` composes interfaces but does not depend on vendor SDKs.
- `observability` and `config` are cross-cutting infrastructure.
- Qt is permitted only under a future `ui` package.
- Game SDK types must be normalized before entering `core` or `policy`.

## Data conventions

- Serialized models use a fixed `schema_version` discriminator.
- Internal physical measurements use SI units and include the unit in field names.
- `observed_at`, deadlines, and expirations are timezone-aware and normalized to UTC.
- `session_time_s` is monotonic simulator/session time and drives deterministic replay.
- Source sequence numbers establish frame order. Adapters will define duplicate, stale,
  and session-transition behavior before M1 live telemetry is accepted.
- Missing simulator capabilities are represented explicitly with optional fields and a
  sorted capability list; fabricated values are forbidden.
- Facts and metadata contain JSON-compatible values only.

## Configuration and privacy

TOML is the human-editable configuration format. Selected machine-specific values can be
overridden with `RACE_ENGINEER_*` environment variables. Unknown configuration keys fail
validation so misspellings cannot silently change runtime behavior.

Raw telemetry, prompts, and model outputs are not recorded by default. Logs are local and
must use identifiers and reason codes rather than unnecessary raw telemetry.

## Deferred decisions

M1 selects `pyirsdk` as the first Windows shared-memory binding, isolated behind an
`IracingSource` protocol. The domain receives only normalized contracts and can replay
privacy-safe source samples without importing `irsdk`. Language model, inference runtime,
TTS engine, adaptive ranker, UI design, distribution format, and license remain deferred.
SQLite is the persistence baseline, but concrete domain tables wait until their
requirements are introduced.

## iRacing M1 boundary

The live source freezes the latest SDK variable buffer before reading the selected fields,
then unfreezes it immediately. Session metadata is cached by the SDK update counter. Only
car indices, numeric user IDs, and pace-car markers are retained from driver metadata;
names are discarded.

The adapter reconnects after simulator startup and disconnect events, rejects duplicate or
out-of-order `SessionTick` values within a session, resets ordering on session transition,
and suppresses replay frames unless explicitly configured. M1 samples lap, position,
speed, fuel, flags, pit state, traffic distance, and reliable same-lap race gaps where the
SDK exposes all required fields.

## Deterministic M2 boundary

M2 receives only normalized frames and race events. The context builder owns bounded
session state, fuel-trend calculations, nearby battle context, and event expiry. The strict
policy maps eligible current-frame events into candidates; the scheduler then applies
priority, expiry, semantic deduplication, cooldowns, and per-frame capacity before emitting
`SpeechIntent` values.

Each candidate produces a versioned `PolicyDecision` containing an approval or suppression
reason. Decision time comes from the telemetry frame rather than the wall clock, keeping
replay deterministic. Critical calls may carry fixed templates, but ordinary intent wording
remains the responsibility of a later language adapter.

The live iRacing command runs this context and policy path for every accepted frame.
Version 2 session recordings persist frames, events, intents, and decisions as separate
streams. A policy failure is isolated: telemetry recording continues and the policy state
is rebuilt for the next frame.
