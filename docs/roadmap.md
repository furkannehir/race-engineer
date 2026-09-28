# Working roadmap

Updated 2026-09-27. This is the current progress/backlog companion to the original
[implementation plan](personalized-ai-race-engineer-implementation-plan.docx), not a claim
that every original milestone exit criterion has been formally validated.

## Current checkpoint

The driver has tried the combined live iRacing prototype and considers it sufficient to
continue development. Short-input recognition, voice quality, and conversational reactions
are deferred under [STT-001, TTS-001, and CONV-004](known-issues.md).

This is qualitative functional acceptance. There are no new measured accuracy, latency,
frame-time, or endurance results accompanying the feedback. Existing defects remain open.

The active implementation increment is the
[local context engine and portable inference plan](context-engine-implementation-plan.md):
smaller bilingual judges, optional CPU/GPU execution including AMD/Vulkan, measured resource
budgets, and independent promotion gates for conversation and proactive ranking. CE-01
through CE-05's opt-in dialogue preview are implemented. Driver feedback inserted the
CE-05 recovery stream before CE-06. CE-05R.1 live running-position derivation is implemented
with deterministic regression coverage and awaits real-race validation. This updates the
next-task order without closing pending live validation.

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
should share the validated service later. CONV-004 now has an opt-in CE-05 implementation;
semantic/listening/live validation remains before default activation.

## Next implementation order

1. **Done — CE-01:** configurable planners, shared runtime launch, and bounded backend
   discovery; CPU defaults preserved.
2. **Done — CE-02 typed baseline:** versioned bilingual suites, frozen promotion targets,
   content-free reports, and the owned-runtime Qwen CPU baseline are recorded. Audio-stage
   timing remains pending deliberately supplied test samples.
3. **Done — CE-04b:** versioned SemanticJudge/dialogue contracts, session-owned bounded
   state, stateless context assembly, deterministic control, and delivery tracking are
   implemented behind the current live route with scripted/fake-clock coverage.
4. **CE-04c–e — implemented, initial screening concluded:** versioned bilingual dialogue
   fixtures/split audit, MiniLM/Laya/Qwen-v2 adapters, offline CPU comparison, and bounded
   optional router. See [evaluation](dialogue-evaluation.md) and the
   [routing decision](ce04-routing-decision.md). Retain the current Qwen-v1 live route;
   no new judge or fallback is promoted. Larger independent qualification remains pending.
5. **CE-05 — implemented as an opt-in preview:** retained-Qwen compatibility judge,
   grounded mixed replies, bounded bilingual calm-teammate acts, fresh per-part/opponent
   refresh and delivery-aware radio integration. CE-05.1 adds grounded overall field-status
   comparisons (last/place out of field/cars behind). The development config enables the
   preview deliberately; fresh semantic, listening/radio and real-race evidence remain
   before promotion as the default route.
6. **CE-05R.1 — implemented; live validation pending:** use complete per-car live lap
   progress for current running position, with official-position fallback on incomplete
   data. Validate starts, overtakes, pits, lapped traffic and retirements in iRacing.
7. **CE-05R.2 — implemented; targeted model check passed, live validation pending:**
   preserve first/last, ahead/behind-count and place-out-of-field meaning through grounded
   comparison and fresh bilingual composition; contradictory yes/no plus P1 output now has
   regression coverage.
8. **CE-05R.3 — implementation complete; sample validation pending:** retain a bounded
   post-release capture tail, emit content-free gate/boundary measurements, and inspect WAVs
   without loading ASR. Use consented English/Turkish one-word samples to decide whether the
   remaining failure is capture, gating, language detection, or Qwen3-ASR recognition.
9. **CE-05R.4 — implemented:** use one deterministic fact-provider catalog for capability
   projection, semantic-to-fact binding, initial resolution and pre-playback refresh. Current
   answers are preserved; arbitrary SDK access and model-supplied values remain forbidden.
10. **CE-05R.5.1–R.5.4 — implemented; R.5.5 validation pending:** compile refreshed
   decisions into typed fact/social/clarification clauses, render deterministic natural
   English/Turkish radio variants, validate exact fact provenance and a 48-word bound, and
   fall back to the previous bounded wording on any failure. Composition remains after
   pre-playback telemetry refresh and adds no second model call. Complete the independent
   replay/listening/live acceptance pass in R.5.5.
11. CE-06: labeled proactive usefulness evaluation in shadow mode.
12. CE-07: optionally train/evaluate smaller classifiers or alternatives if data justifies it.
13. CE-03: optional CUDA/Vulkan controls and stable combined-workload benchmarks. AMD
   support still requires an actual AMD test.
14. CE-08: representative hardware and race validation, conservative Automatic mode,
   independent promotion decisions, and open-source distribution readiness.

## Constraints carried forward

- Keep speech recognition, conversation, synthesis, and personalization local by default.
- The original plan's Jev evaluation remains optional/deferred. The later entirely-local
  decision does not authorize hosted ranking, data uploads, or remote shadow requests.
  Any such evaluation needs a separate explicit opt-in. Laya's local Jev-compatible API
  does not revive or authorize the hosted Jev path.
- Preserve critical deterministic calls and factual grounding; quality/personality upgrades
  must not weaken them.
- Keep STT-001 and TTS-001 as separate measured improvements. CONV-004 has an opt-in CE-05
  implementation but remains open until validation and promotion. FreyaTTS-small is one local
  Turkish TTS-001 evaluation option among alternatives; Piper remains selected until a
  candidate earns replacement through offline Windows/racing-language quality and resource
  comparisons.
- CPU remains a required execution path; acceleration is optional and includes an AMD
  validation target. Automatic selection is not a current capability or a proven default.
- Do not treat the driver's prototype acceptance as resolution of the blue-flag defect or
  the existing conversation interpretation failures. IR-001 remains pending live
  validation even though its normalization fix and regression coverage are implemented.
