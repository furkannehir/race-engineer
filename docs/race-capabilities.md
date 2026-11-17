# Deterministic race capabilities

The [telemetry expansion inventory](telemetry-expansion.md) records proposed
features and priorities beyond the implemented calculation baseline below.
Its [implementation/access matrix](telemetry-expansion.md#implementation-and-ministral-access-snapshot)
also records which calculations are exposed to the checked Ministral prototype;
registration in this core catalog alone does not make a capability model-accessible.

INT-06 adds typed race calculations beside the generic Telemetry Memory queries. A capability
is selected by meaning, not by matching a driver phrase:

1. The Context Engineer receives a value-free catalog with each capability's description,
   temporal scope, required inputs, outputs, freshness, uncertainty and current availability.
2. Qwen may select a capability request but cannot provide or calculate its result.
3. Application code executes the request against normalized Telemetry Memory.
4. The result enters the `ContextPacket` as provenance-bearing evidence, or as explicit unknown
   evidence plus an unavailable reason.
5. Any capability evidence cited by the response is recalculated in the delivery slot before
   deterministic grounding and TTS.

## Implemented INT-06 baseline

`current_classification`
: Returns player position, last occupied position, number of cars reporting a position and an
  authoritative `is_last` relationship. Cars without a current classified position are excluded.
  The boolean relationship guides Core/Qwen but is not directly speakable; spoken numbers still
  use evidence placeholders.

`fuel_range`
: Returns current estimated laps remaining as fuel divided by rolling observed fuel burn per lap.
  It is unavailable until both inputs exist and burn is positive. Its first estimate has no
  caution, traffic, driving-style, reserve or variance adjustment.

`position_change`
: Returns `gained`, `lost`, or `unchanged` plus the absolute net position movement over a fixed
  rolling 30-second window. It does not pretend to distinguish passes from pit cycles,
  disconnects, or classification corrections.

`gap_ahead` / `gap_behind`
: Return the current timing gap to the nearest same-lap car in that direction. Missing or
  lap-offset opponents remain unavailable rather than being represented by an invented gap.

`relative_pace_ahead` / `relative_pace_behind`
: Fit a deterministic linear trend to 10 seconds of gap history and return an application-owned
  relative state plus the absolute gap-change rate. This is short-window relative movement, not a
  clean-air lap-time model.

`catch_time_ahead` / `catch_time_behind`
: Divide the current gap by the measured closing rate when the relevant car is genuinely closing.
  The constant-trend projection is unavailable when the gap is stable, opening, or lacks a full
  10-second history.

`pit_loss_projection` / `pit_stop_duration_projection` / `projected_rejoin_position`
: Advertised future-counterfactual capabilities with stable typed outputs and authoritative
  unavailable reasons. They deliberately do not calculate until track/car pit loss, requested
  service, simulator rules, and field-trajectory models are normalized and validated.

## Lap timing and race progress, 18 November 2026

The core now adds `last_lap_time`, `best_lap_time`, `lap_time_comparison`,
`race_progress` and `fuel_to_finish`. These are Python calculations over nullable
normalized fields, exposed to the checked Ministral prototype through dedicated
tools. Last/best durations are simulator reports, not promises of clean laps or
multi-lap pace. Their comparison returns an absolute difference and direction.
Repeated last/best snapshots cannot be averaged as if each sample were a lap.

Last/best durations in the checked Ministral prototype now use race-radio
pronunciation: `92.5` seconds becomes **one thirty two point five**, and `91.25`
becomes **one thirty one point two five**. The shared language formatter keeps
up to three decimal places, trims trailing zeros and rounds before splitting
minutes/seconds. It supplies English/Turkish words without changing stored values,
units, calculations, delivery refresh or historical recall. Gaps, lap-time
differences and session-clock values retain their ordinary seconds rendering.
The model may copy a verified token or a matching duration; the application owns
the final pronunciation. This adds no model call or speech dependency.

Race progress distinguishes current lap, completed laps, scheduled lap limit,
SDK remaining race laps and session clock. Focused tools return only the requested
count/time to avoid confusing current lap with remaining laps. Timed-only races
have a clock but no exact remaining lap count; clock zero is not a finish crossing.

Fuel range is always rendered as **laps of fuel** in the checked prototype.
Fuel-to-finish uses the larger of `total laps - player completed laps` and the
reported remaining count, counts the unfinished lap in full, and budgets **one
additional lap of reserve** at observed consumption. Required fuel includes this
reserve already; the core returns the verdict and positive shortage/surplus too.
This conservative full-distance budget can overestimate what a lapped driver or
a race ending on time actually needs. The reserve is currently an explicit
constant, not a driver preference or a voice-adjustable setting.

Fuel-to-finish is unavailable without confirmed active race context, a scheduled
lap limit, a valid completed count and positive observed consumption. Timed-only
fuel projection, fuel saving, pit strategy, opponent lap comparisons and clean-lap
history remain future work. There is no new model or model-authored arithmetic.
The existing live model selection, speech and control scope are unchanged.

SDK field names/units were checked against the binding's
[upstream variable catalog](https://github.com/kutu/pyirsdk/blob/master/vars.txt).
Source tests and normalized replay are not a substitute for a fresh live recording
from the new adapter; old recordings cannot supply fields they never captured.

## Safety boundary

- The catalog contains metadata and availability, never live values.
- Capability IDs and empty argument objects are constrained at the planner boundary.
- Every capability is restricted to its declared current, historical, or future-counterfactual
  scope.
- Missing inputs produce explicit unavailable results; nearby telemetry is not substituted.
- Simulator-specific SDK fields remain behind normalized contracts.
- The language model never performs authoritative race arithmetic.

## Exit state and follow-up

The INT-06 baseline is complete when these contracts, calculations, explicit unavailable paths,
refresh behavior, and replay/unit controls pass. Building real pit-loss, service-duration and
field-trajectory models is a separate strategy-model follow-up; their capability IDs and safety
boundary are already stable.

## Replay smoke test

With the configured local Qwen server running, the combined INT-06 fixture can be exercised
without joining iRacing:

```powershell
.\.venv\Scripts\python.exe -m race_engineer intelligence-replay `
  --fixture fixtures\synthetic\intelligence-capabilities `
  --question "How many positions have we gained?"

.\.venv\Scripts\python.exe -m race_engineer intelligence-replay `
  --fixture fixtures\synthetic\intelligence-capabilities `
  --question "When will we catch the car ahead?"

.\.venv\Scripts\python.exe -m race_engineer intelligence-replay `
  --fixture fixtures\synthetic\intelligence-capabilities `
  --question "Where would I rejoin if I pit now?" `
  --json
```

The fixture deterministically represents three positions gained, 12 estimated fuel laps, a
one-second gap ahead that is closing at 0.05 seconds per second, and a 20-second constant-trend
catch estimate. The pit question must remain unavailable with
`missing_rejoin_projection_model`.
