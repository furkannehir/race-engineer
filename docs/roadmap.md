# Working roadmap

Updated 2026-10-11. This is the current progress/backlog companion to the original
[implementation plan](personalized-ai-race-engineer-implementation-plan.docx), not a claim
that every original milestone exit criterion has been formally validated.

## Current checkpoint

Branch decision (2026-10-11): `light-version` owns the interpreter experiments,
draft datasets, training tools and bounded live pilot. The planned Pitward Light
pipeline is CPU STT -> trained request interpreter -> shared telemetry/calculation
engine -> smaller CPU response model -> CPU TTS. Full request extraction and the
smaller response model remain to be implemented; this branch still uses Qwen-4B.
The standard GPU/Qwen edition is maintained separately on `main`. No checkpoint,
draft annotation or existing experiment was discarded in this split.

October 11 checkpoint: the trained MiniLM v2 action pilot is wired into a bounded
source-mode social shortcut; full telemetry-request extraction is not trained yet.
The latest live trial submitted six questions: three information actions were detected,
three had no above-threshold action, all six retained Qwen planning, and five timed out
at the Context stage. The one spoken gap answer lacked a valid current ahead-gap input.
Richer request-extraction training, a measured small CPU reply model and missing-gap
diagnosis remain open. GPU implementation and measurements belong to `main`.

The earlier October 9 checkpoint follows for historical context.

October 9 checkpoint: v6 remains the live development baseline. Correlated radio diagnostics
now distinguish capture, ASR, interpretation, Core, evidence refresh and voice delivery.
The latest CPU-only live session transcribed 12 questions, but 10 timed out in the Context
planner and only two produced speech. One of those misinterpreted a presence check as a
position request. These results do not establish MVP accuracy or predictable latency.
The next experiment is an isolated local multilingual Laya interpreter evaluation, not a
live replacement. Preserve the earlier rejected Laya adapter results described below.
The [first isolated screening](laya-interpreter-experiment.md) reused the pinned Laya 0.3.20
checkpoint without downloads or training. Plain-text CPU inference had a 537 ms median and
636 ms p95, but passed only 4/56 component expectations (28 cases repeated twice), with
0/28 Turkish attempts. It is not promoted; the question interface/checkpoint needs further
investigation. Qwen v6 remains live, and INT-07 accuracy/endurance gates remain open.

The earlier prototype checkpoint and milestone accounting follow; functional progress is
not equivalent to passing the latest live shakedown.

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
| M3: Speech / conversational increment | Automatic calls, local English/Turkish voice, and the new evidence-driven Context/Core/Qwen path are connected to `voice-iracing` | Real-model bilingual evaluation, real-race shakedown, latency measurement, and speech quality |
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
3. **Done — INT-03 baseline:** a local model selects generic telemetry operations from the
   driver turn and a value-free normalized signal catalog. Bounded Telemetry Memory executes
   the plan, permits bounded re-execution of approved evidence queries, allows evidence-free
   social turns and represents unsupported or missing information explicitly. No spoken
   keyword grammar or answer template is part of this path. Current-field aggregates support
   ranking questions without a dedicated phrase intent. Learned inference and richer
   comparison/event tools remain follow-up schema work.
4. **Hardening — INT-04 portable baseline:** one local Qwen call combines the Core communication
   decision and conversational English/Turkish response. Evidence IDs and numeric placeholders
   remain mechanically validated. Unsupported projections now produce an explicit limitation,
   driver telemetry claims are checked against evidence instead of repeated, and deterministic
   classification relationships prevent the model from reversing ranking math. Canonical,
   metric-aware rendering examples cover generic scalar evidence such as fuel, speed and laps
   without spoken-question routing. A bounded repair inference runs only after a structurally
   invalid draft. Broad real-model bilingual evaluation and broader claim/correction coverage
   are still pending.
5. **Hardening — INT-05 integration baseline:** `voice-iracing` now feeds continuous telemetry to
   the new path and grounds refreshed evidence in the existing radio delivery slot. Critical
   deterministic calls, cancellation, expiry and TTS freshness behavior remain in place.
   The typed replay path now distinguishes an unreachable model, timeout, and a reachable
   server-side structured-response failure. Typed temporal scope prevents a future hypothetical
   from being answered with current measurements, and Windows replay output is UTF-8 safe.
   Real-race validation is pending.
6. **Done — INT-06 deterministic capability baseline:** the typed registry, value-free planner
   catalog, deterministic execution, delivery-time refresh and explicit unavailable results are
   implemented. Tested calculations cover classification, fuel range, 30-second net position
   movement, current same-lap gaps, 10-second relative gap trends, and constant-trend catch time.
   Pit loss, service duration, and rejoin position have stable future-capability contracts but
   remain authoritatively unavailable until real strategy-model inputs exist. Qwen selects
   semantically; tested application code owns arithmetic, provenance, freshness and uncertainty.
   See [race capabilities](race-capabilities.md).
7. **In progress — INT-07 evaluation foundation:** versioned bilingual development,
   calibration and locked seed workloads now exercise every INT-06 capability, generic raw
   queries, social abstention, temporal scope and authoritative unavailable results. A
   content-free runner reports exact plan/evidence scores, first/subsequent-call latency,
   optional server prompt/generation timings, model errors, candidate revisions, input and
   implementation fingerprints, and machine metadata. `qwen-context-v6` is the
   runnable portable baseline; the enhanced profile fails closed until a learned temporal
   adapter exists. The first CPU Qwen report is recorded: the original contract timed out on
   17/24 turns, while a compact development-only iteration cut mean planning from 29.5 to
   11.8 seconds but still passed 0/12 exact turns. Qwen is therefore an unpromoted baseline.
   The pinned dynamic-catalog MiniLM and Laya portable candidates scored 1/12 and 3/12 exact
   development turns respectively, with both at 0/6 Turkish; neither is promoted or connected
   live. Promotion-grade dataset expansion, any domain-trained/calibrated retry, robust compound/
   multi-turn coverage and a provenance-bearing enhanced temporal candidate remain. See
   [INT-07 evaluation](intelligence-evaluation.md).
   The October 6 first repair slice binds raw queries to catalog IDs (including operation
   constraints) and adds report-v2 partial-selection diagnostics, extra-request counts and
   separate failure stages. Rejected plans never execute or earn accepted-plan credit;
   exact scoring is unchanged. The v4 repair also separates social/factual/mixed purpose from
   factual time scope, replaces arbitrary unknown text with application-owned reasons and
   places stable catalogs before dialogue for prompt reuse. Eighteen bilingual development
   controls exercise mixed/compound requests, authored follow-ups and changing catalog availability.
   The v4 real-Qwen development run passes 24/30 (12/15 per language), including all original
   12 cases, with no transport/validation/execution errors. Six failures remain in mixed-purpose,
   compound and historical raw-speed interpretation; these are not marked fixed. Planner median
   is 2.21 s in sequential synthetic replay, not end-to-end radio latency. Locked evaluation,
   actual sequential dialogue, multi-scope requests and real-race acceptance remain pending.
   Integration checks also hardened silent Core output and added bilingual social output-shape
   examples. Four final synthetic text-pipeline checks completed without errors (roughly 5–14 s);
   this does not validate microphone/TTS behavior or broad conversation quality.
   The v5 continuation adds a bounded request inventory and independent social-content flag
   within the same planner call, plus 20 bilingual composition/statistic controls. The latest
   saved run passes 41/50: original 12/12, controls 15/18, composition 14/20. Historical
   statistic controls now pass and compound retrieval improves, but extra evidence, mixed-purpose
   mistakes and an English social regression remain. Planner median is 4.06 s on this broader
   workload; no full-radio latency claim is made. V5 was wired into this development branch's
   live/replay path before the October 7 v6 switch. INT-07 remains open.
   October 7 integration checks exposed and repaired an over-broad Core dependency check:
   unrelated retrieved comparisons no longer force their inclusion in an otherwise grounded
   answer. Used comparisons still require all dependency references. Automated verification:
   395 tests, Ruff lint and mypy pass. Whole-pipeline naturalness and live acceptance remain open.
   A standalone full text latency benchmark now runs the same production adapters and
   orchestration, with per-call provider timings, cache/token metrics, retries and source/input
   fingerprints. It manages its own local server and preserves existing external servers.
   The October 7 12-turn CPU baseline completed 11 turns: completed median 12.97 s, max 32.32 s,
   with one Turkish Core validation failure (INT-007) and 8/12 exact plans. Cache reuse helps,
   but a nearly fully cached compound reply still spent 13.75 s generating structured output.
   No prompt, model, runtime acceleration or live behavior was changed for this measurement.
   The first performance slice replaces Core's repeated metadata/reference objects with compact
   `goal` plus `speech` output and application-owned reference reconstruction. The matched CPU
   repeat completed 12/12 with no repair call: median fell to 8.85 s, sample p95 to 20.93 s and
   median Core time to 2.97 s. INT-007 was reproduced as alternate placeholder delimiters and
   fixed at the trusted binding boundary; live validation remains. Planner quality is unchanged:
   exact plans remain 8/12 and the repeated English vent still retrieves and speaks fuel.
   Automated verification now passes 410 tests, Ruff and mypy. Voice and game-load timing remain.
   A separately selectable v6 prompt candidate improved exact plans from 45/58 to 53/58 on a
   matched set and passed eight fresh bilingual contrasts. In the matching 30-turn reply
   workload it reached 30/30 exact plans (v5: 18/30) but raised median reply latency from
   8.01 s to 10.26 s. A user-run planner repeat reproduced 53/58 passes (median 4.05 s,
   p95 10.45 s). V6 is now the live/replay development default by user decision, prioritizing
   accuracy while speed work continues; v5 remains explicitly selectable for comparisons.
   Five older follow-up/composition failures, the latency tradeoff and release evaluation
   remain for INT-07.
   Default-v6 and explicit-v5 regression checks pass; current verification is 414 tests,
   Ruff lint and mypy.
8. **Follow-up — strategy models:** normalize track/car pit loss, requested service and field
   trajectory inputs before enabling the existing pit/rejoin capability contracts.
9. GPU runtime work and measured compute controls belong to the standard edition on
   `main`. This branch targets CPU-only interpretation and response generation;
   removing Qwen-4B requires full request-extraction training and a measured smaller
   response model, not merely disabling the current Qwen fallback.
10. Proactive usefulness learning remains separate and shadow-only until independently
   labeled evidence justifies promotion.

## Research and domain adaptation

Dataset preparation for RES-01 was activated on 2026-10-10 by user decision. An
exploratory draft-label training pilot was authorized and completed on 2026-10-11.
Production model selection remains pending, not a prerequisite silently added to the
MVP or a claim that specialization guarantees speed or accuracy. A separately requested
source-mode v2 social live trial is now wired and enabled in the repository config;
real-race timing/quality validation is pending. RES-02 remains deferred; Qwen still
plans factual telemetry requests and generates replies.

- **RES-01: purpose-trained radio interpreter — first action-training pilot done;
  dataset review and reliable interpretation remain in progress.**
  The review-only v2 corpus has 1,400 assistant-authored drafts (700 English,
  700 Turkish), preserving the original 800-row first-pilot corpus and 42-example
  seed unchanged. It has 157 scenarios, 100 provisional grouping IDs,
  open-vocabulary meaning/request annotations, provenance,
  contextual feedback, compounds, mixed scopes, and marked synthetic ASR examples.
  An authored catalog, deterministic materialization check, duplicate validator, coverage
  inventory and paginated terminal review filters are implemented. All full-row reviews
  remain draft and splits are unassigned; no model is promoted as a full interpreter. See
  [driver-meaning dataset and review guide](driver-meaning-dataset.md).
  Initial owner feedback accepts broad fuel/vent meanings and adds contextual back-off
  feedback after repeated reminders; full-row reviews remain draft.
  The [local MiniLM action pilot](driver-interpreter-training-pilot.md) compares
  unadapted similarities, a trained head and last-two-layer adaptation using a separate
  group-disjoint 552/126/122 manifest. Both trained candidates reach 50% exact match;
  the majority sanity check is 42.62%. Warm selected-model CPU inference is 9.225 ms
  median on this machine. It trains actions/clarification only, using plain PyTorch,
  not SetFit; full telemetry-request extraction is not implemented. Weak-action
  generalization and four missing positive-test categories block promotion.
  A targeted 600-row expansion adds 60 scenarios emphasizing pure/mixed vents,
  real car symptoms, corrections, preferences, current-reply feedback, ambiguity,
  weather and social/radio contrasts. Its read-only seed-17 coverage check finds
  positives for every action plus clarification in 948/216/236 candidate partitions;
  all reviews remain drafts. A separately authorized second CPU training run is
  completed with the same settings from the pinned base. On v2, the frozen control
  has 40.68% test exact match, versus 37.71% for the adapted validation-selected
  checkpoint (54.63% validation exact, threshold 0.7). The latter has 5.882 ms warm
  median classification on this machine and matches the two supplied dirty-move
  and weather development cases; acknowledgments, out-of-scope and other social
  intents remain weak. These percentages are not directly comparable with v1's
  different test set. Both checkpoints are preserved and neither is promoted.
  See [v2 pilot results](driver-interpreter-training-pilot.md#second-pilot-v2-2026-10-11).
  The inspector
  defaults to v2; training still defaults to v1 unless a corpus is explicitly selected.
  Human annotation review, error-driven sampling/training experiments and fresh
  acceptance data remain. Further methods such as SetFit need a new explicit experiment protocol,
  covering short inputs, dialogue, mixed purposes, uncertainty and open wording,
  rather than a prescribed spoken command set. The authorized bounded live trial uses
  a persistent offline CPU worker for singleton radio-check/greeting/vent shortcuts;
  uncertainty, mixed and factual turns retain Qwen planning. Qwen still generates every
  reply and evidence refresh/grounding remain enforced. Warm worker round trips were
  about 9–10 ms on five development probes, not total spoken-answer latency; common
  English/Turkish presence checks still fell back. Startup/route/stage logs, failure
  fallback, process cleanup and config rollback are implemented. Packaged builds retain
  the existing path. See [race-trial procedure](driver-interpreter-training-pilot.md#experimental-live-race-trial-2026-10-11).
  Actual race A/B validation and full telemetry-request extraction remain pending.
- **RES-02: specialize the Core/response model.** Investigate adapting the local conversational
  model to a calm race-engineer teammate style and supported evidence-based reasoning.
  Application code still owns telemetry retrieval, arithmetic, provenance, freshness and final
  grounding. Do not train memorized race values into answers, weaken safety boundaries, or assume
  that fine-tuning the same model makes inference cheaper. Compare quality and latency against
  the unadapted baseline before adoption.

The isolated Laya and mDeBERTa experiments screen existing checkpoints without training;
neither earned promotion. Dataset preparation builds on those failures without treating
already-probed examples as a fresh holdout. The MiniLM pilot is a separate, limited
adaptation experiment; its synthetic test is not a locked release test. Generic
"copy/checking" acknowledgments remain a later UX discussion.

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
