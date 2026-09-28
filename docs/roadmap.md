# Working roadmap

Updated 2026-09-28. This is the current progress/backlog companion to the original
[implementation plan](personalized-ai-race-engineer-implementation-plan.docx), not a claim
that every original milestone exit criterion has been formally validated.

## Current checkpoint

The driver has tried the combined live iRacing prototype and considers it sufficient to
continue development. Short-input recognition, voice quality, and conversational reactions
are deferred under [STT-001, TTS-001, and CONV-004](known-issues.md).

This is qualitative functional acceptance. There are no new measured accuracy, latency,
frame-time, or endurance results accompanying the feedback. Existing defects remain open.

The next implementation increment is the
[Context/Core Engineer architecture v3](context-core-architecture-v3.md). It replaces the
planned bounded semantic-query path with continuous telemetry memory, evidence-producing
context intelligence, a Core Engineer decision boundary, Qwen-generated conversation and a
deterministic grounding gate. CE-01 runtime foundations and the CE-02 evaluation baseline
remain reusable; the old CE-04/05 design is historical rather than the next task.

| Milestone | Current position | Remaining/deferred work |
| --- | --- | --- |
| M0: Foundation | Contracts, configuration, logging, fixtures, and tests implemented | Maintain compatibility as new domains are added |
| M1: iRacing telemetry | Live reading, normalization, recording, reconnect handling, and the IR-001 source-level fix implemented | Genuine-blue live validation; broader scenario validation |
| M2: Strict policy | Core policy, SQLite-backed preferences, and durable content-free decision/outcome history implemented | Broader scenario and real-race validation remain |
| M3: Speech / conversational increment | Automatic calls and local English/Turkish live voice conversation plus CE-01 runtime and CE-02 evaluation foundation implemented; driver accepts the current prototype | CE-04/05 planner/context work; speech quality and extended live validation |
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

1. **Done — INT-01:** versioned evidence/brief/generation contracts, Context/Core/Qwen
   interfaces, one-turn orchestration and deterministic grounding. Scripted tests cover the
   three architecture proofs: direct fact, telemetry analysis and natural social response.
2. **Done — INT-02 foundation:** bounded Telemetry Memory now exposes generic normalized
   signal selectors and replay-deterministic current/window operations. Missing signals or
   incomplete history produce explicit unknown evidence instead of fabricated values.
3. **Next — INT-03:** implement a Context Engineer baseline that selects measurements,
   deterministic derivations, learned inferences and explicit unknowns without a spoken
   command grammar.
4. **INT-04:** implement Core Engineer and Qwen conversational generation adapters plus
   evidence-aware English/Turkish evaluation.
5. **INT-05:** integrate the new path with live radio, delivery-time evidence refresh,
   cancellation and deterministic critical-call coexistence.
6. **INT-06:** evaluate temporal/context model candidates and portable/enhanced execution
   profiles.
7. Optional CUDA/Vulkan controls and stable combined-workload benchmarks remain after a
   stable intelligence workload; AMD support still requires an actual AMD test.
8. Proactive usefulness learning remains separate and shadow-only until independently
   labeled evidence justifies promotion.

## Constraints carried forward

- Keep speech recognition, conversation, synthesis, and personalization local by default.
- The original plan's Jev evaluation remains optional/deferred. The later entirely-local
  decision does not authorize hosted ranking, data uploads, or remote shadow requests.
  Any such evaluation needs a separate explicit opt-in. Laya's local Jev-compatible API
  does not revive or authorize the hosted Jev path.
- Preserve critical deterministic calls and factual grounding; quality/personality upgrades
  must not weaken them.
- Keep STT-001 and TTS-001 as separate measured improvements. CONV-004 has a planned CE-05
  slice; none is resolved merely by adopting the new plan. FreyaTTS-small is one local
  Turkish TTS-001 evaluation option among alternatives; Piper remains selected until a
  candidate earns replacement through offline Windows/racing-language quality and resource
  comparisons.
- CPU remains a required execution path; acceleration is optional and includes an AMD
  validation target. Automatic selection is not a current capability or a proven default.
- Do not treat the driver's prototype acceptance as resolution of the blue-flag defect or
  the existing conversation interpretation failures. IR-001 remains pending live
  validation even though its normalization fix and regression coverage are implemented.
