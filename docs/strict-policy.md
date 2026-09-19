# M2 deterministic policy

M2 decides what is worth saying before a language model or speech engine is introduced.
It consumes only normalized `RaceContext` and `RaceEvent` contracts and emits constrained
`SpeechIntent` values. Simulator SDK types do not cross this boundary.

## Context features

`DefaultRaceContextBuilder` maintains bounded, session-local state. It provides:

- non-expired recent events;
- one-based stint lap, reset after leaving the pit lane;
- rolling fuel consumption in liters per lap, reset after refueling;
- nearest ahead and behind gaps as positive magnitudes; and
- `attacking`, `defending`, `sandwiched`, `contested`, or `clear` battle state.

Missing telemetry remains `None`; the builder does not invent gaps, laps, or fuel trends.
All calculations use frame timestamps and telemetry values, so fixture replay is
deterministic.

## Strict rules

The initial rules approve session-phase and added-flag calls. Position changes are enabled
by default. Pit transitions are silent by default because entering or leaving the pit lane
does not, on its own, imply a useful instruction. Fuel-threshold events are supported once
the event derivation threshold is introduced.

Safety priorities are fixed. Caution and red, yellow, or black flags are critical and may
use fixed templates without language generation. Green, blue, checkered, and formation
calls are important or routine. Critical calls can interrupt; other calls cannot.

## Scheduler and decision trace

The scheduler ranks candidates, rejects expired work, removes equivalent calls from the
same frame, applies priority-specific cooldowns, and suppresses lower-ranked work when the
per-frame budget is exhausted. Every candidate produces a versioned `PolicyDecision` with
one of these reasons:

- `approved`
- `disabled`
- `expired`
- `duplicate`
- `cooldown`
- `superseded`

Decisions are emitted as structured logs and may also be sent to a persistence callback.
The default configuration permits one intent per frame, uses 15-second routine and
5-second important cooldowns, and lets critical calls bypass cooldowns.

The live iRacing reader executes this policy on every accepted frame. When `--output` is
used, its version 3 recording stores frames, events, intents, decisions, and generated
utterances independently. This preserves suppressed decisions for auditing and makes the
policy and language paths replayable without launching iRacing.

Replay the M2 acceptance fixture with:

```powershell
race-engineer replay-policy fixtures/synthetic/m2_green_flag `
  --config config/default.toml
```
