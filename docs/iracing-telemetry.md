# iRacing telemetry integration

M1 reads iRacing's local Windows shared-memory SDK through `pyirsdk` 1.3.6. The binding is
isolated in `PyIrSdkSource`; normalizers, event derivation, fixtures, and policy-facing code
do not import it. The upstream binding supports live telemetry, session YAML data, and
binary memory-map replay: <https://github.com/kutu/pyirsdk>.

## Initial telemetry surface

| Domain value | iRacing SDK source |
| --- | --- |
| Session identity | `SessionUniqueID`, `SessionNum` |
| Ordering and time | `SessionTick`, `SessionTime` |
| Session phase | `SessionState`, `SessionFlags` |
| Player identity | `PlayerCarIdx`, sanitized `DriverInfo.Drivers` metadata |
| Lap and position | `Lap`, `PlayerCarPosition` |
| Vehicle state | `Speed`, `FuelLevel`, `OnPitRoad` |
| Nearby traffic | `CarDistAhead`, `CarDistBehind` |
| Opponents | `CarIdxPosition`, `CarIdxLapCompleted`, sanitized driver metadata |
| Same-lap race gaps | Difference between `CarIdxF2Time` values when both cars are on the same lap |
| Replay detection | `IsReplayPlaying` |

The upstream variable catalog describes `Speed` in m/s, `FuelLevel` in liters,
`SessionTime` in seconds, the car-index arrays, and flag/state fields:
<https://github.com/kutu/pyirsdk/blob/master/vars.txt>.

Opponent `gap_to_player_s` is signed: negative means the opponent is ahead; positive means
the opponent is behind. It is populated only in race sessions when both cars have the same
completed-lap count and valid `CarIdxF2Time` values. Other contexts remain `None` rather
than presenting an unreliable estimate.

### Player-applicable blue flags

The raw blue bit is not exposed blindly in a race session. A normalized blue flag requires
a valid player classification plus at least one active, non-pace-car opponent whose
completed-lap count is greater than the player's. This rejects impossible startup
transitions while retaining iRacing's raw signal as the indication that the lapping car is
close enough to matter. If the required race-order data is incomplete, blue fails closed.

In non-race sessions, the completed-lap relationship is not a reliable applicability test,
so the raw blue signal is preserved after the player has a valid position. See IR-001 in
[known-issues.md](known-issues.md) and its source-level regression fixture.

## Lifecycle behavior

The adapter waits for iRacing, reconnects after disconnection, freezes the latest SDK
variable buffer for each atomic read, and always releases the buffer. It drops incomplete,
duplicate, and out-of-order samples. A session-ID change resets ordering. Replay samples
are dropped by default and may be enabled in configuration.

At the default 10 Hz sample rate, the adapter schedules reads against monotonic deadlines.
Time spent reading and normalizing a sample is deducted from the wait before the next
deadline, so normal SDK work does not steadily reduce the effective sample rate. If work
overruns one or more deadlines, the adapter skips those expired slots rather than producing
a burst of stale reads.

Each read records buffer-wait, variable-read, metadata-refresh, and total-read timing.
Structured warnings identify missed deadlines, reads above `slow_read_warning_s`, and
accepted-frame gaps above `sample_gap_warning_s`. The defaults are 50 ms and 250 ms,
respectively. These signals distinguish a slow shared-memory read from an expensive session
metadata refresh when diagnosing capture gaps.

Selected telemetry is normalized into `TelemetryFrame`. Changes in phase, flags, position,
and pit state produce deterministic `RaceEvent` values. The `read-iracing` CLI can print
frames or create a new fixture-compatible recording. Every accepted frame also passes
through the deterministic context builder, strict policy, and fact-bound language adapter.
Version 3 recordings include frames, events, policy decisions, speech intents, and
generated utterances. Non-expired utterances enter the bounded local speech queue without
blocking telemetry reads. A language or playback failure does not interrupt telemetry
capture or policy processing. The command never overwrites an existing recording directory.

## Privacy boundary

The session YAML contains driver names and other metadata. M1 retains only numeric iRacing
user IDs, car indices, and pace-car markers. Names and the unfiltered YAML structure do not
enter source fixtures, normalized contracts, or recordings.
