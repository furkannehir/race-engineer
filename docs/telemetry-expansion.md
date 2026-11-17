# Telemetry expansion inventory

Recorded 17 November 2026 from the driver's telemetry-expansion discussion.
This page keeps the feature inventory and proposed order in one place. Features
marked as expansion or future work are not implemented or verified merely by
being listed here. Availability must be checked for the actual car and session.

Actions the engineer could perform are recorded separately in the
[iRacing control inventory](engineer-controls.md), including pit requests,
integration routes and per-car availability. Cockpit adjustments remain
driver-only; Pitward may read their telemetry and advise, but does not change them.

## Current implementation

The adapter reads session identity, ordering/time, phase, flags, replay state,
player identity, lap number, position, speed, fuel, pit-lane state, nearby
distances, and selected opponent classification/lap/timing data. See
[iRacing integration](iracing-telemetry.md) for the selected SDK fields,
normalization, sampling and privacy boundary.

The existing deterministic calculation baseline includes:

- Current classification and rank relationships.
- Fuel range from current fuel and observed rolling consumption.
- Net position change over a rolling 30-second window.
- Current gaps to the nearest same-lap cars ahead and behind.
- Relative gap movement over a 10-second history window.
- Constant-trend catch estimates when sufficient closing history exists.

These calculations require valid inputs and sufficient history; missing data
remains unavailable. The pit-loss, stop-duration and rejoin-position capability
IDs already exist, but their calculation models are unimplemented. See
[race capabilities](race-capabilities.md) for exact behavior and limitations.

The checked Ministral terminal prototype now exposes 14 implemented core
calculations, 26 normalized current signal paths, bounded numeric-history queries
and verified recall of delivered readings. Additional adapter fields remain a
separate task. The live/installed model selection has not been changed by this slice.
The English live trial is planned for 18 November, with opt-in live integration
still required; see [trial notes](dialogue-model-comparison.md#planned-english-live-trial-18-november-2026).

### Driver-facing measurements, 17 November 2026

Instantaneous speed/velocity is excluded from conversational tools: the driver
already has a speedometer. The adapter/core may retain it for internal analysis,
but pace must never be answered with speed or a speed history statistic.

Numerical gap gains/losses must be compared **lap by lap**, not reported as
seconds per second. Existing ten-second gap calculations currently supply only
qualitative recent movement (closing, falling back or stable) to Ministral.
Their numerical rates remain internal, including when used by the existing
constant-trend catch estimate. Generic gap-window queries cannot bypass this
boundary. Current gaps remain seconds; nearby physical distances remain metres.

As of 18 November, reported last/best lap durations and their difference are
implemented. Multi-lap pace trends and opponent per-lap gap comparisons remain
unimplemented. Requests for those must report that limitation. Do not multiply a short-window
rate by an assumed lap duration and call it a lap comparison. Future expansion
needs verified lap-boundary measurements for the same opponent, with explicit
handling of missing history, opponent changes, pit laps and lap differences.

## Implementation and Ministral access snapshot

Source audit and tool expansion: **18 November 2026**. Keep these stages separate when updating
the backlog:

1. **Adapter:** iRacing data selected and normalized by Pitward, subject to live
   availability. A field existing in the SDK is not an implemented adapter field.
2. **Core:** application calculations, history or event behavior already written.
   Implemented calculations can still return unavailable for missing inputs.
3. **Ministral:** an operation actually exposed to the checked terminal prototype.
   This means tool access, not general model ability, answer-quality acceptance or
   access from the live/installed application.

The live factory still uses the configured LFM/Qwen composition. Ministral is
not wired into it. Interactive chat defaults to the expanded synthetic telemetry
fixture; `--fixture` can read another normalized fixture/recording. Its evaluation
command can change synthetic values between turns. Thus **no Ministral live-iRacing
connection is implemented yet**, including for the tools marked accessible below.

| Feature (inventory ID) | iRacing adapter | Core implementation | Checked Ministral access |
| --- | --- | --- | --- |
| Session briefing (TEL-01) | Partial: IDs, time, phase; session type used internally | Basic session lifecycle; full briefing absent | Partial: session elapsed time and phase; no full briefing |
| Starting grid (TEL-02) | No separate grid data | No retained grid/briefing | No |
| Race progress (TEL-03) | Current/completed lap counts, scheduled lap limit, SDK remaining laps and remaining/total clock time | Race-progress reading; no exact timed-race lap projection or finish-crossing model | Focused current/completed/total/remaining lap/time reads and a progress overview |
| Standings (TEL-04) | Partial: overall positions and sanitized IDs; no class standings | Classification, last occupied rank, classified count and rank checks | Yes for existing classification, focused first/last checks and selected-opponent positions; no class/grid data |
| Position changes (TEL-05) | Yes: position stream | Position-change events and 30-second net movement; no full race/pit-cycle attribution | Yes for the 30-second calculation and supported history queries; no event-cause attribution |
| Gaps (TEL-06) | Partial: same-lap F2 timing inputs and opponent laps | Current nearest same-lap gaps ahead/behind; broader strategy absent | Yes: nearest ahead/behind tools and normalized signed gaps for selected opponents when known |
| Traffic and spotting (TEL-07) | Partial: nearby metre distances, positions/laps and blue bit | Blue applicability filtering and simple gap-based battle state; full spotting absent | Partial: metre distances, battle state and normalized flags; no full spotter/rejoin model |
| Personal lap timing (TEL-08) | Reported last and best lap durations; no current-lap elapsed timer | Last/best timing and their difference; missing/nonpositive timing stays unavailable | Last/best tools and last-versus-best comparison |
| Pace and consistency (TEL-09) | Last/best duration snapshots; no distinct clean-lap history | Last-versus-best only; average/trend/consistency absent | Single-lap timing/comparison only; repeated SDK snapshots are not a lap average |
| Opponent pace (TEL-10) | Partial: same-lap gap history inputs; no opponent lap times | 10-second gap trends and constant-trend catch estimates; lap-pace model absent | Qualitative recent movement and catch estimates in both directions; no numerical rates, per-lap gap comparison or lap-pace model |
| Sectors and corners (TEL-11) | No | No split/reference-lap model | No |
| Fuel (TEL-12) | Current fuel and race progress inputs | Observed burn/range plus conservative fuel-to-finish for scheduled lap races with a one-lap reserve | Separate fuel-endurance and sufficiency tools; timed-only finish prediction remains unavailable |
| Pit-lane assistance (TEL-13) | Partial: `OnPitRoad` only | Entry/exit events; no limiter/speed-limit/box guidance | Partial: current pit-lane Boolean only, when its source capability is present |
| Pit services (TEL-14) | No pending service selections or service-status fields | No service confirmation/duration tracking | No reads or actions |
| Pit-cycle tracking (TEL-15) | Partial: own pit-lane state; no opponent pit state | Basic own stint-lap counter and transitions; no field pit-cycle tracking | Partial: current own pit state/stint lap; no field pit-cycle history |
| Pit strategy (TEL-16) | Missing required strategy inputs/models | Three registered projection stubs deliberately return unavailable | No calculation tool; can report missing support |
| Tyres (TEL-17) | No compound/set/wear/temperature fields | Not implemented | No reads or actions |
| Car health (TEL-18) | No warning/repair/tow/temperature fields | Not implemented | No |
| Car systems (TEL-19) | No brake-bias/ABS/TC/map/energy fields | No read-only system integration; actuation is driver-only | No |
| Flags and race control (TEL-20) | Partial: seven normalized flags and session phase | Flag/phase events and existing automatic-call policy; broader penalties/rules absent | Yes for existing flags/phase when known; no broader penalty model or race-control actions |
| Incidents (TEL-21) | No counts/limits | Not implemented | No |
| Weather and track (TEL-22) | No weather/track-condition fields | Not implemented | No |
| Driving analysis (TEL-23) | Partial: speed only; no pedal/steering/track-position capture | Generic numeric history operations exist; detailed driving analysis absent | Speed excluded from conversation; no detailed driving analysis |
| Handling and setup (TEL-24) | Missing diagnostic inputs | No validated handling/setup analysis | No; reports unsupported diagnosis |
| Endurance/team racing (TEL-25) | No team requirements/driver-change data | No team-rules or handover model | No |
| Race outlook (TEL-26) | Partial: position/gap/fuel/phase building blocks | No combined outlook or expected-finish model | No; reports unsupported overview |

### Exact implemented reads

The selected SDK surface contains 25 variable names, plus filtered session YAML
metadata. The adapter normalizes these into the existing contracts:

| Data | SDK input | Available beyond the adapter | Ministral |
| --- | --- | --- | --- |
| Player position | `PlayerCarPosition` | `player.position`; classification calculation | Yes, numerical position only |
| Opponent positions | `CarIdxPosition` | Normalized opponents; classification and gap selection | Yes: select ahead/behind/leader or a current classified position; no driver names |
| Player lap number | `Lap` | `player.lap_number`; fuel/stint tracking | `read_current_telemetry` |
| Completed player laps | `LapCompleted` | Separate `player.laps_completed`; never inferred by subtracting one from the current lap | Focused count read and race progress |
| Last/best lap durations | `LapLastLapTime`, `LapBestLapTime` | Positive duration snapshots and last-versus-best calculation | Dedicated timing/comparison tools; no weighted sample average |
| Remaining race laps | `SessionLapsTotal`, `SessionLapsRemainEx` | Nullable scheduled limit and SDK-reported remaining count; unlimited sentinels removed | Focused remaining/total reads; never fuel endurance |
| Session clock | `SessionTimeRemain`, `SessionTimeTotal` | Nullable remaining/total seconds; unlimited sentinel removed | Current clock reads; no conversion to an exact timed-race lap count |
| Opponent completed laps | `CarIdxLapCompleted` | Opponent lap values and same-lap/blue checks | `read_current_telemetry` with an opponent selector |
| Timing gaps | `CarIdxF2Time` plus lap/position inputs | Signed opponent gaps and nearest-ahead/behind calculations | Yes, two gap tools |
| Player speed | `Speed` | `player.speed_mps`; generic numeric memory queries | Excluded from conversation, including generic queries and recall; retained internally |
| Current fuel | `FuelLevel` | `player.fuel_l`; observed burn and range calculation | Current read, history and `get_fuel_range` |
| Own pit-lane state | `OnPitRoad` | `player.in_pit_lane`; transition events and stint tracking | Current read; missing capability is unknown, not false |
| Nearby physical distances | `CarDistAhead`, `CarDistBehind` | Player metre-distance fields | Current read/history |
| Flags | `SessionFlags` | Green, yellow, red, blue, white, checkered and black; applicable filtering/events | Current read; a missing flags capability is unknown, not no active flags |
| Session state/phase | `SessionState`, `SessionFlags` | Formation, green, caution, checkered or unknown | Current read; unknown stays unavailable |
| IDs, ordering and time | `SessionUniqueID`, `SessionNum`, `SessionTick`, `SessionTime` | Session scope, sequence and elapsed time; freshness/lifecycle | Elapsed time readable; IDs/order remain internal scope/provenance |
| Player/driver identity | `PlayerCarIdx`, filtered `DriverInfo.Drivers` | Numeric user/car identifiers and pace-car markers; names discarded | No driver-name/identity lookup |
| Session type | Filtered `SessionInfo.Sessions.SessionType` | Internal normalizer input; not a field on `TelemetryFrame` | No |
| Replay marker | `IsReplayPlaying` | Replay filtering and normalized marker when enabled | Current read when its source capability is present |

### Calculation registry versus model tools

| Core capability ID | Implementation | Ministral exposure |
| --- | --- | --- |
| `current_classification` | Implemented, with classified-count/rank limitations | `get_current_race_position`, full `get_race_classification` and focused first/last tools |
| `fuel_range` | Implemented; needs valid fuel and observed burn | `get_fuel_range`; returns observed input values too, including partial known fuel when range is unavailable |
| `position_change` | Implemented; needs a 30-second window | `get_recent_position_change` |
| `gap_ahead` | Implemented for nearest same-lap car | `get_gap_ahead` |
| `gap_behind` | Implemented for nearest same-lap car | `get_gap_behind` |
| `relative_pace_ahead` | Implemented from 10 seconds of gap history | `get_relative_pace_ahead`, qualitative state/verdict only; numerical rate hidden |
| `relative_pace_behind` | Implemented from 10 seconds of gap history | `get_relative_pace_behind`, qualitative state/verdict only; numerical rate hidden |
| `catch_time_ahead` | Implemented for supported closing conditions | `get_catch_time_ahead`; keeps estimate scope |
| `catch_time_behind` | Implemented for supported closing conditions | `get_catch_time_behind`; unavailable when not closing |
| `last_lap_time` / `best_lap_time` | Implemented from positive SDK timing snapshots | `get_last_lap_time`, `get_best_lap_time` |
| `lap_time_comparison` | Last minus best, with direction and absolute difference computed in Python | `compare_last_lap_to_best`; no multi-lap/opponent trend claim |
| `race_progress` | Known current/completed/total/remaining race laps and clock | Overview plus focused remaining/completed/total/time tools |
| `fuel_to_finish` | Remaining full scheduled player laps plus one reserve lap at observed burn | `get_fuel_to_finish`; no timed-only or guaranteed finish prediction |
| `pit_loss_projection` | Unimplemented calculation; explicit unavailable stub | Not exposed |
| `pit_stop_duration_projection` | Unimplemented calculation; explicit unavailable stub | Not exposed |
| `projected_rejoin_position` | Unimplemented calculation; explicit unavailable stub | Not exposed |

The checked Ministral menu now has **26 read/conversation operations** in total:

| Operation | What is accessible |
| --- | --- |
| `get_current_race_position` | Fresh numerical player position from the supplied memory |
| `get_gap_ahead` | Fresh nearest same-lap gap ahead |
| `get_gap_behind` | Fresh nearest same-lap gap behind |
| `get_race_classification` | Existing classification outputs; reported-car count is not grid size |
| `check_if_driver_is_first` | Existing authoritative first-place check and supporting ranks |
| `check_if_driver_is_last` | Existing authoritative last-place check and supporting ranks |
| `get_fuel_range` | Existing fuel range, current fuel and observed burn where known |
| `get_recent_position_change` | Existing 30-second net movement and its window |
| `get_relative_pace_ahead` | Existing ahead-gap qualitative state and a Yes/No closing verdict; no numerical rate or per-lap claim |
| `get_relative_pace_behind` | Existing behind-gap qualitative state and a Yes/No closing verdict; no numerical rate or per-lap claim |
| `get_catch_time_ahead` | Existing constant-trend catch estimate |
| `get_catch_time_behind` | Existing constant-trend catch estimate |
| `get_last_lap_time` / `get_best_lap_time` | Simulator-reported last completed/session best durations |
| `compare_last_lap_to_best` | Application-calculated difference and direction |
| `get_remaining_race_laps` | Only the SDK-reported remaining race count, or unavailable for timed races |
| `get_completed_lap_count` | Only the player's completed lap count |
| `get_total_race_laps` | Only the scheduled race lap limit |
| `get_remaining_race_time` | Only the session clock remaining; not a finish-crossing prediction |
| `get_race_progress` | General race-progress overview; focused tools are preferred for specific counts |
| `get_fuel_to_finish` | Verdict, full scheduled distance, one-lap reserve and required fuel/positive shortfall or surplus |
| `read_current_telemetry` | Up to eight selected paths from the 26 exposed normalized current signals; no speed or arbitrary SDK attributes |
| `query_telemetry_history` | Existing numeric queries up to 120 seconds; no speed/derivatives; gap and duration snapshots support latest only |
| `recall_radio_reading` | Previously delivered supported readings/calculations; historical, not fresh telemetry or arbitrary past-point lookup |
| `reply_to_driver` | Social speech with no numerical race readings |
| `report_unavailable_information` | Explain missing support; does not retrieve the requested information |

Creating a core capability does not automatically publish it to this menu.
The generic reads are restricted to normalized contracts and the existing numeric
catalog/query rules. Opaque IDs stay internal and no arbitrary past-point or
simulator-action tool is exposed. Automatic announcements run through
the existing event/policy/speech path separately; their existence does not give
Ministral access to their facts.

### Source audit references

- [Selected SDK reads and filtered metadata](../src/race_engineer/telemetry/iracing/source.py).
- [Normalized player/opponent/frame/context contracts](../src/race_engineer/core/contracts.py).
- [Normalization and flag applicability](../src/race_engineer/telemetry/iracing/normalizer.py).
- [Derived events](../src/race_engineer/telemetry/iracing/events.py).
- [Fuel, stint and battle context](../src/race_engineer/policy/context.py).
- [Implemented calculations and projection stubs](../src/race_engineer/intelligence/capabilities.py).
- [Lap timing, race progress and fuel-to-finish math](../src/race_engineer/intelligence/race_progress.py).
- [Generic numeric history](../src/race_engineer/intelligence/telemetry_memory.py).
- [Base reading-tool execution](../scripts/lfm3_dialogue.py).
- [Checked Ministral tools and recall](../scripts/ministral_dialogue.py).
- [Read-only telemetry/core bridge and unit-aware binding](../scripts/ministral_telemetry.py).
- [Fixed-fixture chat command](../scripts/try_ministral.py).
- [Changing-fixture evaluation](../scripts/evaluate_ministral_dialogue.py).
- [Live application factory](../src/race_engineer/intelligence/factory.py)
  and [current default](../config/default.toml).

## Feature inventory

Work descriptions use the following terms:

- **Read:** obtain and normalize SDK or session metadata.
- **Calculate:** combine readings and retained history in application code.
- **Estimate:** use a validated model with explicit assumptions and uncertainty.
- **Conditional:** available only for some cars, sessions or SDK channels.

An existing foundation is not a claim that every listed question is exposed to
Ministral or accepted in live racing. Every row includes potential expansion.

| ID | Area | Uses in Pitward | Work and current boundary |
| --- | --- | --- | --- |
| TEL-01 | Session briefing | Track/layout, car/class, session type, race distance and relevant rules; opening briefing. | Expand normalized session metadata. Session identity and phase already exist. |
| TEL-02 | Starting grid | Actual starting position, class position, grid size and nearby starters. | Read and retain starting information separately from current standings. Deferred race-start briefing: [RACE-001](known-issues.md#race-001-announce-starting-grid-and-an-expected-finishing-position). |
| TEL-03 | Race progress | Current/completed laps, remaining laps/time, halfway, final-lap and finish reminders. | Basic counts/clock now implemented. Timed-race lap projections, final-crossing rules and proactive reminders remain future work. |
| TEL-04 | Standings | Overall/class position, leader/class leader, first/last checks and another competitor's position. | Overall classification foundation exists. Expand class and target data; distinguish entrants from currently classified cars. |
| TEL-05 | Position changes | Gains/losses since the start, since an update or during a stint; explain pit-cycle movement when supported. | Rolling 30-second net change exists. Add start/stint history and event attribution; a position change alone does not prove an overtake. |
| TEL-06 | Gaps | Gaps ahead/behind, to a leader/class leader or another target; lap-offset relationships. | Nearest same-lap gap foundation exists. Expand targets and lap handling; physical proximity and classified order are different quantities. |
| TEL-07 | Traffic and spotting | Cars alongside, lapping traffic, relevant blue flags, slower classes and traffic after a stop. | Conditional reads, class/lap context and calculations. Distances and blue-flag filtering exist; rejoin traffic is an estimate. |
| TEL-08 | Personal lap timing | Current lap time, last completed lap, session best, previous best and difference from best. | Last/best and their difference implemented. Current-lap elapsed timer, prior-best history and distinct lap records remain future work. |
| TEL-09 | Pace and consistency | Recent averages, best consecutive laps, improving/worsening trend, consistency and unusually slow laps. | Lap history and calculations; identify invalid, pit, caution and incomplete laps before comparison. |
| TEL-10 | Opponent pace | Recent valid-lap comparisons, faster/slower rivals, closing/opening gaps and catch estimates. | Short-window gap trends/catch foundation exists, with qualitative-only gap movement exposed. Add verified lap-to-lap comparisons; numerical gains/losses must be lap based, never seconds per second. Gap movement is not automatically clean-air pace. |
| TEL-11 | Sectors and corners | Where time was gained/lost, sector comparisons, theoretical best lap and comparison with a saved reference. | Timing boundaries, track mapping and detailed history; use published boundaries where available or explicitly derived splits. |
| TEL-12 | Fuel | Remaining fuel, consumption, estimated range, fuel needed to finish, reserve, refuelling amount and saving targets. | Range and conservative scheduled-lap finish budget with one-lap reserve implemented. Timed-race finish modelling, saving targets, service requests and configurable reserves remain future work. |
| TEL-13 | Pit-lane assistance | Entry/exit, speed-limit and limiter reminders, pit-box guidance where supported, service progress. | Pit-lane transitions exist. Expand conditional channels and track metadata. |
| TEL-14 | Pit services | Confirm requested fuel, tyres, compound, pressures, repairs and tear-off; report completion and actual stop duration. | Conditional reads and event history. Reading service requests does not add automatic car-control commands. |
| TEL-15 | Pit-cycle tracking | Own/opponent visits, stop counts, stint lengths, observed pit-lane time and competitors currently pitting. | Field history and calculations. A pit visit alone does not establish a tyre change or fuel quantity. |
| TEL-16 | Pit strategy | Pit window, stop loss, rejoin position, traffic risk, undercut/overcut scenarios and required stops. | Future calibrated estimates. Existing projection capabilities deliberately return unavailable until their inputs/models are implemented. |
| TEL-17 | Tyres | Compound, available sets, observed changes, last measured wear/temperatures and comparisons between stints. | Conditional reads and retained measurements. Label pit measurements with their observation time; do not present them as live on-track values. |
| TEL-18 | Car health | Engine warnings, oil/water readings, repair requirements, active repair timers, towing and fast-repair availability. | Conditional reads and alerts. Exact damage location/severity needs a verified source, not an inferred percentage. |
| TEL-19 | Car systems | Brake bias, ABS/TC settings, engine maps, boost, push-to-pass, DRS and hybrid/energy information where exposed. | Read-only integration and channel discovery. These controls remain driver-operated; Pitward may report or advise but must not change them. |
| TEL-20 | Flags and race control | Applicable flags, caution/restart, pit opening, repair flags, disqualification and available penalty information. | Basic flags/phase exist. Expand relevant channels and rules; do not invent a penalty reason or deadline. |
| TEL-21 | Incidents | Count, recent changes, margin to configured limits and off-track events. | Read, history and session rules. Incident counters do not establish fault or explain the whole incident. |
| TEL-22 | Weather and track | Air/track temperature, wind, precipitation, wetness, declared-wet status and observed changes; support tyre discussions. | Conditional reads and trends. Current observations are not a weather forecast or a map of every corner. |
| TEL-23 | Driving analysis | Braking points, throttle use, minimum corner speed, gears, steering corrections, ABS use and reference-lap comparisons. | Higher-rate capture and analysis. Define needed sample rates separately from the current 10 Hz engineer feed. |
| TEL-24 | Handling and setup | Investigate reported understeer, oversteer, instability, bottoming or traction issues; correlate feedback with measurements. | Conditional measurements and validated analysis. A single symptom or slower lap is not a definitive diagnosis. |
| TEL-25 | Endurance/team racing | Stint duration, driver changes, driving requirements, fuel-stop planning and handover summaries. | Session rules, conditional team data, history and estimates. |
| TEL-26 | Race outlook | Answer "How's the race going?" using position movement, pace, gaps, fuel, incidents and remaining distance; estimate expected finish when supported. | Combine available capabilities. Expected finish remains an estimate with uncertainty, separate from the actual starting grid. |

## Uses across the application

| Use | Application behavior |
| --- | --- |
| Automatic announcements | Relevant calls about flags, positions, fuel, pace, traffic and pit events, using existing priorities and coordinated playback. |
| Choosing when to speak | Delay routine messages during braking, cornering or close battles; retain urgent-message priority. |
| Understanding conversation | Combine recent dialogue with race context for references such as "him", "that lap" or "are we catching?"; clarify unresolved ambiguity. |
| Post-race summaries | Starting/finishing position, pace, consistency, fuel, pit performance, incidents and major turning points. |
| Driver memory | Comparable car/track baselines, observed consumption and recurring patterns alongside explicit preferences; observed data must not silently become an asserted preference. |
| Control-panel displays | Useful live facts, trends, freshness and unavailable measurements. |
| Replay and testing | Reproduce failures, test changing values and compare models without joining a race. |
| Diagnostics | Detect stale/disconnected telemetry, missing channels and capture gaps. Track application/GPU performance separately from race data. |

## Data and calculation boundaries

- Discover and validate channels for the active car/session. Preserve units,
  validity, observation time, source and current/historical/estimated scope.
- Preserve freshness checks and refresh current evidence before delivery.
  A remembered radio call is historical even when it contains a plausible value.
- Tyre wear and core temperatures can be pit measurements rather than live
  on-track readings. Retain when and how they were measured.
- Mandatory/optional repair-needed flags can support a repair warning. They do
  not by themselves identify the damaged component, percentage or time to repair;
  active repair timers have their own availability boundary.
- Precipitation at start/finish and overall wetness estimates do not describe
  every corner or guarantee future weather.
- Opponent fuel, private strategy and driver intent remain unknown without an
  explicit source. A pit visit is not proof of changed tyres; a slower lap is not
  proof of tyre degradation; a counter increase is not proof of blame.
- Keep race arithmetic and measured-history calculations in application code.
  Let the conversational model select information and explain results. Additional
  telemetry features do not require another language model per feature.
- Keep reads/calculations off latency-sensitive UI and audio paths, bound retained
  history, and retrieve compact relevant facts rather than sending every sample
  and every channel to the model. Reuse the existing session resets, replay and
  sanitized-recording boundaries.

## Proposed implementation order after the trial

This order is recorded as a recommendation, not as completed work or a newly
started implementation milestone.

1. **Lap timing:** last completed lap, session best, difference from best, recent
   average and trend (TEL-08/09); filter unsuitable lap records.
2. **Implemented in the terminal prototype, live connection pending:** existing
   calculations, normalized reads and bounded numeric history are exposed to
   Ministral (existing parts of TEL-05/10/12 and the matrix above).
3. **Race progress and fuel to finish:** remaining distance/time, required fuel
   and reserve (TEL-03/12).
4. **Grounded race overview:** combine these facts into situational answers
   (TEL-26), with explicit limits when useful inputs are missing.
5. **Pit services and strategy:** establish observed stop/service measurements
   before predicting pit loss or rejoin position (TEL-13 through TEL-16).

Before accepting a feature, verify meaning as well as successful tool execution:
changing/current values, historical follow-ups, units and direction, relevant
answers, unavailable inputs, session resets and bounded runtime/memory cost.
Replay tests should precede live validation of simulator-dependent behavior.

## References

- [Current adapter behavior](iracing-telemetry.md).
- [Implemented race calculations and unavailable projections](race-capabilities.md).
- [Working roadmap](roadmap.md).
- [Engineer control inventory](engineer-controls.md).
- [Ministral comparison, prototype results and planned trial](dialogue-model-comparison.md).
- [Upstream pyirsdk variable catalog](https://github.com/kutu/pyirsdk/blob/master/vars.txt).
- [iRacing tyre-data update behavior](https://support.iracing.com/support/solutions/articles/31000161130-2021-season-3-release-notes-2021-06-08-02-).
- [iRacing repair-needed telemetry additions](https://support.iracing.com/support/solutions/articles/31000177148-2025-season-4-release-notes-2025-09-08-02-).
- [iRacing precipitation telemetry](https://support.iracing.com/support/solutions/articles/31000172630-2024-season-2-release-notes-2024-03-05-01-).
- [iRacing track wetness and declared-wet telemetry](https://support.iracing.com/support/solutions/articles/31000172730-2024-season-2-patch-1-release-notes-2024-03-13-01-).
