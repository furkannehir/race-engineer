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

## Lifecycle behavior

The adapter waits for iRacing, reconnects after disconnection, freezes the latest SDK
variable buffer for each atomic read, and always releases the buffer. It drops incomplete,
duplicate, and out-of-order samples. A session-ID change resets ordering. Replay samples
are dropped by default and may be enabled in configuration.

At the default 10 Hz sample rate, selected telemetry is normalized into `TelemetryFrame`.
Changes in phase, flags, position, and pit state produce deterministic `RaceEvent` values.
The `read-iracing` CLI can print frames or create a new fixture-compatible recording. It
never overwrites an existing recording directory.

## Privacy boundary

The session YAML contains driver names and other metadata. M1 retains only numeric iRacing
user IDs, car indices, and pace-car markers. Names and the unfiltered YAML structure do not
enter source fixtures, normalized contracts, or recordings.
