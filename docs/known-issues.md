# Known issues

## IR-001: False blue-flag call immediately after race start

**Status:** Open - investigation deferred  
**Area:** iRacing telemetry normalization and flag-event derivation  
**Observed:** 2026-09-19, session `iracing:1:0`, source sequence `3748`

During the M3 live speech validation, the system announced "Blue flag" roughly 2.7 seconds
after the green-flag call. The driver confirmed that no blue-flag condition existed. The
policy, language, and TTS stages behaved correctly for the event they received; the defect
is upstream of policy, in the interpretation or transition handling of iRacing flag data.

Investigation tasks:

- inspect normalized frames and the derived flag event around source sequence `3748`;
- verify which iRacing flag source represents a player-specific blue-flag condition;
- distinguish session-wide flag state from flags applicable to the player's car;
- preserve the captured transition as a regression fixture; and
- prove that the fix suppresses the false start-of-race call without suppressing a genuine
  blue flag for lapping traffic.

Do not add a policy cooldown or timing workaround until the telemetry meaning is verified;
that could hide legitimate blue flags rather than correct the source event.
