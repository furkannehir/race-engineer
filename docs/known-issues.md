# Known issues

## CONV-001: Vague fuel follow-up switches to position

**Status:** Open - first conversational prototype

**Observed:** 2026-09-20, local Qwen3-4B-Instruct-2507 Q4_K_M, CPU

After "How is fuel looking?", "Where are we now?" returns current position rather than
staying with fuel or clarifying the topic. The position value is grounded, but the intended
topic is lost. Reproduced by `fuel-context` in `fixtures/conversation/cases.json`.

## CONV-002: Ambiguous opponent reference can be guessed

**Status:** Open - first conversational prototype

**Observed:** 2026-09-20, same local-model setup

With no preceding opponent context, "Is he pulling away?" can select the car ahead instead
of asking which car. Gap trends are unsupported, so it does not invent a trend, but the
reference resolution is wrong. Reproduced by `ambiguous-reference` in the model evaluation.

## CONV-003: Unsupported part of a compound question can be omitted

**Status:** Open - first conversational prototype

**Observed:** 2026-09-20, additional paraphrase evaluation

"Tell me the overall position and recommended tyre pressures" returns position but omits
the explicit acknowledgement that tire advice is unavailable. It does not invent tire
pressures. Reproduced by `partially-supported-compound` in
`fixtures/conversation/holdout.json`.

These are model interpretation issues, not transcription issues; this prototype takes typed
input. Keep the failing evaluation cases rather than weakening their expected behaviour.

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
