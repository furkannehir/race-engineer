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
privacy-safe source samples without importing `irsdk`. Model-backed language generation,
inference runtime, speech engines beyond the initial Windows adapter, adaptive ranker, UI
design, distribution format, and license remain deferred. SQLite is the persistence
baseline, but concrete domain tables wait until their requirements are introduced.

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

The live iRacing command runs this context and policy path for every accepted frame. A
policy failure is isolated: telemetry recording continues and the policy state is rebuilt
for the next frame.

## Local M3 speech boundary

The first language adapter converts only policy-approved `SpeechIntent` facts into short
English `Utterance` values. Its fixed templates cover the initial phase, flag, position,
pit, and fuel facts. Critical fixed wording bypasses normal fact rendering. Unknown facts,
unsupported languages, mismatched intent IDs, and word-limit violations fail closed rather
than encouraging invented advice.

The replaceable language interface includes a timeout-and-fallback wrapper for future
model-backed generators. The initial live configuration uses the deterministic adapter
directly. Language failure is isolated from telemetry and policy processing, and logs
contain identifiers and template metadata rather than utterance text.

Version 3 session recordings persist frames, events, intents, decisions, and utterances as
separate streams. Fixture replay compares generated utterances byte-for-byte at the
contract level without depending on audio hardware.

The initial text-to-speech adapter uses the Windows SAPI voices already installed on the
machine. Each playback runs in an isolated hidden process, keeping COM and platform details
outside the domain. The async playback queue is bounded, drops expired speech, orders
pending calls by priority, and honors the intent's interruption policy. Telemetry capture
does not wait for audio to finish.

Playback failures and shutdown timeouts are logged with intent IDs and reason codes but not
utterance text. They do not stop telemetry, policy evaluation, language generation, or
recording. Output-device selection and higher-quality voice engines remain replaceable
adapter work.

## Conversational prototype

The following direction was agreed on 2026-09-19. Text-over-replay, local push-to-talk
speech input, bilingual Piper audio output, and live iRacing integration are implemented;
in-game shakedown validation remains pending:

- push-to-talk input, with English and Turkish supported from the first prototype;
- entirely local speech recognition, conversational inference, and speech synthesis;
- natural-language questions and contextual follow-ups, without a fixed spoken command
  vocabulary; and
- answers grounded in current race data, with explicit handling of missing or stale facts.

Qwen3-ASR is the selected first speech-recognition engine. Keep recognition behind a
replaceable adapter boundary so faster-whisper can be evaluated as an alternative if
latency or resource use is unacceptable. These engines are alternatives, not sequential
stages. faster-whisper is not part of the initial implementation and is not an automatic
runtime fallback.

The initial ASR adapter uses the native Transformers Qwen3-ASR-0.6B-hf checkpoint in an
isolated CPU worker, with local-only loading and bounded private-pipe requests. Microphone
capture is push-to-talk, memory-only, duration-limited, and guarded against silence and
overflow. The worker is killed on cancellation or timeout. See
[speech-to-text.md](speech-to-text.md) for setup and limitations.
Evaluation must still cover human English and Turkish recognition quality,
push-to-talk release-to-first-audio latency, memory use, and iRacing frame-time impact.
Qwen3-4B-Instruct-2507 is the selected conversational model. The first adapter uses a
loopback-only llama.cpp endpoint with schema-constrained query plans. Q4_K_M on CPU is the
initial integration baseline; in-race performance still needs validation. Piper 1.8.0 is the
prototype bilingual output adapter, behind `ConversationSpeaker`. A separate persistent CPU
worker loads pinned English/Turkish voices, generates bounded PCM in memory, and plays it
only while capture is stopped. Timeout or cancellation kills the worker; errors fall back
to the printed answer. This does not replace SAPI for automatic calls. Voice quality
and distribution licensing remain open; see
[conversational-speech.md](conversational-speech.md).

The implemented replay client uses bounded session memory, read-only queries, fresh fact
retrieval after inference, and deterministic English/Turkish reply rendering. It does not
accept model-generated numeric values or unrestricted reply text. The model interprets
free-form questions rather than matching a fixed command vocabulary. Replay clock semantics
are explicit, and this command is separate from the unchanged live policy/speech path.
See [conversation.md](conversation.md) for setup, limitations, and real-model evaluation.

The `voice-iracing` composition shares the live context/policy/recorder path with the
existing reader. A latest-only `LiveRaceState` uses current UTC and connection generations;
no replay clock or synthetic default frame can masquerade as live data. The `LiveRadio`
scheduler serializes SAPI automatic calls and Piper answers, bounds/expires pending work,
and grants microphone capture slots. Critical calls cancel capture or lower-priority
playback and wait for cleanup off the telemetry path. Answers refresh facts after
inference and again at their radio slot; Piper confirms freshness after synthesis before
playing. Disconnect, session change, and stale-frame detection invalidate old work.
See [live-conversation.md](live-conversation.md) for operation and validation limits.
Critical calls remain deterministic; model inference does not run in the telemetry task.
The remote ranking proposal below is not part of this local conversational prototype.

## Post-M3 improvement: Jev-assisted policy ranking

After the M3 speech path is complete, evaluate Jev as an optional ranking adapter for
ambiguous, noncritical communication decisions. Jev may score whether a candidate is worth
announcing, rank simultaneous candidates, or classify a bounded message type. It does not
create race facts, generate wording, or own the final scheduling decision.

The integration must preserve these constraints:

- critical safety and race-control calls remain deterministic and bypass learned ranking;
- calls are event-triggered rather than issued for every telemetry frame;
- the scheduler continues to enforce expiry, cooldown, deduplication, and capacity;
- timeout, service failure, or insufficient confidence falls back to the strict policy;
- only compact, privacy-safe derived context may leave the machine, with explicit opt-in;
- initial operation is shadow-only, recording probabilities and disagreements without
  changing what the driver hears; and
- promotion beyond shadow mode requires replay fixtures plus measured latency and
  decision-quality evidence.

This keeps Jev behind a replaceable policy-ranking interface. The local strict policy
remains sufficient for offline operation and is never dependent on the remote adapter.
