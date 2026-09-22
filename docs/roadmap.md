# Working roadmap

Updated 2026-09-22. This is the current progress/backlog companion to the original
[implementation plan](personalized-ai-race-engineer-implementation-plan.docx), not a claim
that every original milestone exit criterion has been formally validated.

## Current checkpoint

The driver has tried the combined live iRacing prototype and considers it sufficient to
continue development. Short-input recognition, voice quality, and conversational reactions
are deferred under [STT-001, TTS-001, and CONV-004](known-issues.md).

This is qualitative functional acceptance. There are no new measured accuracy, latency,
frame-time, or endurance results accompanying the feedback. Existing defects remain open.

The next implementation increment is the
[local context engine and portable inference plan](context-engine-implementation-plan.md):
smaller bilingual judges, optional CPU/GPU execution including AMD/Vulkan, measured resource
budgets, and independent promotion gates for conversation and proactive ranking. All CE
CE-01 configuration/runtime foundations are implemented; CE-02 evaluation is next. This
updates the next-task order without closing pending live validation.

| Milestone | Current position | Remaining/deferred work |
| --- | --- | --- |
| M0: Foundation | Contracts, configuration, logging, fixtures, and tests implemented | Maintain compatibility as new domains are added |
| M1: iRacing telemetry | Live reading, normalization, recording, reconnect handling, and the IR-001 source-level fix implemented | Genuine-blue live validation; broader scenario validation |
| M2: Strict policy | Core policy, SQLite-backed preferences, and durable content-free decision/outcome history implemented | Broader scenario and real-race validation remain |
| M3: Speech / conversational increment | Automatic calls and local English/Turkish live voice conversation plus CE-01 runtime foundation implemented; driver accepts the current prototype | CE-02/04/05 evaluation and planner/context work; speech quality and extended live validation |
| M4: Adaptive personalization | SQLite profile/preferences plus observational policy-decision and radio-outcome history implemented; no inferred learning yet | CE-06/07 feedback definitions, local shadow ranker, small-model evaluation; no hosted Jev in current scope |
| M5: Driver controls / UI | Basic PySide6 radio desk implemented early: lifecycle, audio checks, press-to-bind keyboard/mouse/wheel PTT, mute, tray and driver preferences | CE-03/08 CPU/GPU controls and hardware validation; real-race shakedown, later voice preference commands/session review |
| M6: Second simulator | Not started | Another adapter and cross-simulator contract validation |

## Pending live validation: control-panel shakedown

The driver selected the second radio-desk design and asked to prioritize testing the
existing engineer in a real race before learning or voice-based settings. The basic
[control panel](control-panel.md) is implemented and tested with fake runtime/hardware.
The driver reports that the Thrustmaster TX binding and combined radio work well enough to
continue, with noticeable but currently acceptable layered speech latency. Extended-race
responsiveness, tray operation, timing breakdowns, and endurance remain validation tasks.
Existing defects are not resolved by the UI.

## Driver-memory foundation and scheduled shadow evaluation

The versioned SQLite profile repository, validated preference vocabulary, audit commands,
safe reset, headless controls, panel integration, runtime handoff, and content-free durable
decision/outcome history are now implemented. Panel JSON version 3 retains only machine
controls; legacy driver preferences are safely imported once. History has a 30-day default
retention boundary and does not influence live behavior.

Recommended implementation order:

1. **Done:** add versioned SQLite migrations and repositories for a local driver profile
   and explicit communication preferences, separate from inferred preferences.
2. **Done:** define the initial setting vocabulary, safe defaults, driver-only scope, reset
   behavior, headless controls, provenance and restart persistence.
3. **Done:** route the panel preference dialog through the repository and apply the
   selected default profile predictably to communication behavior while preserving
   critical-call rules. Do not grant the conversational model unrestricted database
   writes.
4. **Done:** persist privacy-conscious policy decisions and radio outcomes with identifiers and
   reasons, so later feedback and ranker evaluations can be traced to actual behavior.
   Do not start storing microphone audio, prompts, or transcripts by default.
5. Use that foundation to define feedback signals and evaluate a local ranker in shadow
   mode before allowing it to influence audible output.

Step 5 is now scheduled as CE-06 after the runtime/evaluation foundations and shared context
in the [merged implementation plan](context-engine-implementation-plan.md). Playback
outcomes are not usefulness labels; feedback semantics and separate shadow observations
must be defined before training. Natural-language preference commands and the existing UI
should share the validated service later. CONV-004 is scheduled in CE-05, not yet implemented.

## Next implementation order

1. **Done — CE-01:** configurable planners, shared runtime launch, and bounded backend
   discovery; CPU defaults preserved.
2. **Next — CE-02:** bilingual quality suite and CPU/stage-latency baseline, with promotion
   targets.
3. CE-03 and CE-04: optional CUDA/Vulkan controls and a small local NLI judge, each measured
   against the baseline. AMD support requires an actual AMD test.
4. CE-05: grounded hybrid conversation and bounded bilingual acknowledgments.
5. CE-06: labeled proactive usefulness evaluation in shadow mode.
6. CE-07: train/evaluate smaller classifiers or alternatives if data justifies them.
7. CE-08: representative hardware and race validation, conservative Automatic mode,
   independent promotion decisions, and open-source distribution readiness.

## Constraints carried forward

- Keep speech recognition, conversation, synthesis, and personalization local by default.
- The original plan's Jev evaluation remains optional/deferred. The later entirely-local
  decision does not authorize hosted ranking, data uploads, or remote shadow requests.
  Any such evaluation needs a separate explicit opt-in.
- Preserve critical deterministic calls and factual grounding; quality/personality upgrades
  must not weaken them.
- Keep STT-001 and TTS-001 as separate measured improvements. CONV-004 has a planned CE-05
  slice; none is resolved merely by adopting the new plan.
- CPU remains a required execution path; acceleration is optional and includes an AMD
  validation target. Automatic selection is not a current capability or a proven default.
- Do not treat the driver's prototype acceptance as resolution of the blue-flag defect or
  the existing conversation interpretation failures. IR-001 remains pending live
  validation even though its normalization fix and regression coverage are implemented.
