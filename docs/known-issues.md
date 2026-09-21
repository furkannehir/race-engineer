# Known issues and deferred improvements

## STT-001: Short and single-word questions are sometimes misrecognized

**Status:** Deferred improvement - accepted prototype limitation

**Reported:** 2026-09-21, driver feedback after the combined live test

Recognition sometimes fails, especially for one-word input. The driver accepts the current
prototype for now. This report does not yet identify whether capture boundaries, the energy
gate, language detection, or the recognizer is responsible; do not assume an engine defect.

Future work:

- distinguish clipped/discarded capture from incorrect transcription using explicitly
  supplied or consented English/Turkish test samples; no background microphone recording;
- evaluate one-word questions such as "Position", "Fuel", "Sıra", and "Yakıt", along with
  longer paraphrases and realistic racing noise;
- assess capture timing, silence thresholds, language selection, accuracy, and latency
  before choosing tuning or an alternative local recognizer; and
- request a repeat when input cannot be reliably understood, rather than inventing a query.

Keep Qwen3-ASR as the current adapter. faster-whisper remains an evaluation alternative,
not a newly selected engine or automatic fallback. Natural phrasing must remain supported;
the test examples are not a restricted command vocabulary.

## TTS-001: Spoken replies need a more natural voice

**Status:** Deferred improvement - accepted prototype limitation

**Reported:** 2026-09-21; follows earlier feedback about the Turkish voice

The voice feels unnatural/off to the driver, particularly in Turkish. Current Piper voices
are sufficient for the prototype; no voice or engine change is requested in this increment.

Future work: compare local English/Turkish voices or engines using racing vocabulary,
numbers, gaps, pronunciation, prosody, and short conversational acknowledgments. Include
driver listening feedback, synthesis latency, resource impact during iRacing, and engine/
voice distribution terms. Preserve replaceable adapters, language routing, cancellation,
and the live radio scheduler. Do not select on naturalness alone.

## CONV-004: Add contextual race-engineer acknowledgments and reassurance

**Status:** Deferred feature request - not implemented

**Requested:** 2026-09-21

Conversation should respond naturally to remarks as well as factual questions. For
example, "That's dirty" could receive a concise acknowledgment or "Focus on your race
now", rather than being forced into a position/fuel query or a generic unsupported reply.
The requested feel is a supportive race engineer, not simply a spoken telemetry lookup.

Future work:

- add bounded conversational acts for acknowledgments, reassurance, and refocusing,
  alongside the existing read-only factual-query route;
- support English/Turkish paraphrases and conversational context without requiring fixed
  phrases or generating a reply to every remark;
- keep responses short and subject to radio priority, expiry, and interruption; critical
  calls still take precedence and busy racing situations should not gain extra chatter;
- distinguish the driver's report from verified race evidence: "Copy. Focus on your race"
  is a possible neutral reply, while "Yes, we saw it" requires evidence the system can
  actually observe. Do not invent witnessed contact, assign blame, or imply steward review;
- keep factual race answers grounded and critical calls deterministic; personality must
  not become an unrestricted source of race facts or strategy; and
- add evaluation cases separating factual questions, emotional remarks, mixed inputs,
  and ambiguous comments, including when clarification or silence is preferable.

These three items are follow-up quality/features, not changes to current runtime behavior.
See [roadmap.md](roadmap.md) for milestone status and the proposed next implementation slice.

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

CONV-001 through CONV-003 were reproduced with typed input, independently of transcription.
Keep the failing evaluation cases rather than weakening their expected behaviour.

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
