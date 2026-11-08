# Working roadmap

Updated 2026-11-08. This is the current progress/backlog companion to the original
[implementation plan](personalized-ai-race-engineer-implementation-plan.docx), not a claim
that every original milestone exit criterion has been formally validated.

## Current checkpoint

November 8 model decision after the VR shakedown: **LFM3-1.7B Q4_K_M is the English
MVP model**, using native tool selection and grounded JSON replies through one
CUDA b12088 server. The source default now selects LFM; Qwen remains an explicit
rollback in `config/qwen-legacy.toml`. Qwen3-ASR and the default Chatterbox settings
are unchanged. Installer component set 3 selects the matching LFM model/CPU
runtime; frozen inference and CPU race latency still require validation. Existing
executables must be rebuilt, and release-clearance gates remain closed.

The real VR session recorded 0.38 s mean LFM request time per spoken reply and
2.21 s speech synthesis, with roughly 14.6 GiB combined VRAM while driving.
Six of nine transcribed questions produced spoken replies; three best-lap
requests failed argument validation, and a last-lap request selected a count.
This model decision prioritizes the current resource/latency budget while keeping
those correctness failures open. It is not a claim that LFM passed semantic acceptance.

Next agreed sequence:

1. Tighten supported/unavailable selection, bounded handling of invalid calls,
   and prevention of measurement substitutions such as counts for lap duration.
2. Add last completed lap, session best, difference from best and recent trend
   from valid completed laps, accounting for invalid and pit laps.
3. Validate the driver's real questions, paraphrases, follow-ups and missing data
   for meaning as well as latency, then run another live shakedown.
4. Deepen pit strategy with separately defined stop duration, pit window and
   rejoin estimates, using verified inputs and explicit assumptions.

November 8 English-first LFM3 trial: verified Q4 weights and separate CPU/CUDA
llama.cpp b12088 runtimes are installed without replacing Qwen. An opt-in native
tool adapter now selects existing capabilities/raw telemetry and submits grounded
reply drafts through the same resident LFM server. It retains temporal validation,
evidence refresh, unavailable-data handling and existing grounding. Terminal
feedback exposed numeric replies bypassing the native function argument schema.
Native tool selection now uses a constrained JSON reply on the same resident
model, without changing Qwen. The conversational wording slice now gives LFM a
compact human-readable fact view, with original references retained for grounding
and live refresh. Unsupported-data follow-ups choose a bounded explanation style
instead of repeating the same generic sentence. Follow-up framing still varies
with longer dialogue. Current first/last checks now select distinct native tools,
use Python-calculated yes/no claims and refresh both the claim and supporting
position before speech. History is quoted context for selection; old spoken
measurements are excluded from factual reply generation. The reported repeated
ranking sequence and eight first/last classification states passed locally.
The latest 50-turn development run completed all 50 turns, with 29 exact plans and 35
evidence expectation matches, versus Qwen's 49 completed/45 exact/44 evidence
matches. Native turns had 322 ms median and 473 ms p95 typed latency, with zero
Core repairs; this excludes
STT, speech generation, playback and VR load. Native selection alone is not an
acceptance gate: social checks, broad overviews, compound questions and temporal
scope still need better selection. A prompt-cache opt-out probe did not resolve
the earlier history errors; cache settings are unchanged. These English-only
measurements preceded the model decision above; no training was performed. Its model/runtime buffers save roughly
2.34 GiB versus the tested Qwen baseline. See [the trial and follow-ups](lfm3-trial.md).

November 8 speech memory placement: CUDA default-voice synthesis keeps unused
reference encoders in system RAM, with active generation weights resident on the
GPU. V3 warm-up allocation dropped from about 2100 to 1582 MiB without a per-reply
weight transfer. The same placement adapter supports the pinned V2 runtime. Restart
the source panel/engineer to apply it; model assets and live profile keys are unchanged.
See [default-voice memory placement](text-to-speech.md#default-voice-memory-placement-2026-11-08).

November 7 V3 speech trial: Chatterbox Multilingual V3 is installed in a separate
source runtime with a verified cached checkpoint and explicit model selection.
`data/trials/chatterbox-v3-live.toml` retains the previous Qwen v8/FP16 setup and
selects CUDA/BF16 V3 for both conversational replies and automatic announcements.
Seven real playback checks completed using one model/worker. Logs now identify the
speech variant and allocator memory, alongside existing per-turn timings. Listening,
unwanted-tail reliability and performance in a race remain to be accepted; TTS-002
is open. Normal V2 setup and rollback remain available. See
[the V3 race trial](text-to-speech.md#v3-source-race-trial-2026-11-07).

October 29 live-delivery fixes: replies have a separate 15-second queue/preparation
budget while latest-frame freshness stays at three seconds. The panel now separates
Preparing speech from playback dispatch, and reports failed/unplayed answers.
Only completed spoken replies enter intelligence history; internal redacted numbers
cannot be spoken. The experimental v8 Core isolates radio checks/greetings from old
factual replies, and its planner distinguishes lap duration from race rank. Real
last-lap timing retrieval remains absent. See INT-012–014 for the live reproduction,
tests and remaining Turkish/planner limitations. V7/FP16 remains the normal default;
the separate `data/trials/qwen-v8-live.toml` profile remains the live shakedown candidate.

October 25 speech migration: replies and automatic calls now share one persistent
local Chatterbox Multilingual 0.1.7/V2 default-voice backend, selected through `[tts]`.
Piper/SAPI code and split settings are removed; PTT WAV cues are unchanged. CUDA is
preferred with visible CPU fallback. The existing radio priorities, cleanup and
freshness boundaries remain. Windows setup/worker packaging and pinned assets were
updated; listening, combined race-load performance and rebuilt executable acceptance
remain separate checks. See [the speech guide](text-to-speech.md).

October 28 memory tuning: the driver accepted the BF16 speech candidate after the
isolated EN/TR comparison. Shared Chatterbox speech now defaults to native BF16
T3 on supported CUDA, with the decoder and CPU path remaining FP32. The benchmark
saved about 1 GiB of tensor memory but did not improve synthesis latency; explicit
FP32 rollback remains available. Real bilingual routing, interruption and shutdown
checks passed. Combined iRacing/Qwen contention and longer race acceptance remain
open; no other model or radio-cue changes are included.

October 28 Qwen cache experiment: an offline, owned-server FP16/Q8 comparison
keeps v7, the Q4 model and 8192-token capacity unchanged. Q8 saved 540 MiB of KV
allocation and reduced isolated typed pipeline latency, but the full 62-question
development comparison caught a Q8-only Turkish starting-grid/current-position
confusion. Live Qwen remains FP16; reproduce that temporal regression and review
the common baseline failures before any cache promotion. See
[the cache trial and reports](qwen-cache-trial.md).

October 28 v8 follow-up: the experimental planner now interprets time before tool
selection, with bilingual contrasts and the unchanged v7 wire schema. Opt-in
temporal evidence metadata reaches Core and final grounding; v7's authored prompt
and live FP16 default stay unchanged. V8 passed the targeted starting-grid cases
and completed all 84 questions per cache in the wider matrix, but exact-plan
regressions and an unsupported negative damage claim remain. Review INT-010 and
the [four-way v8 comparison](qwen-cache-trial.md#v8-results-2026-10-28-not-promoted)
before promoting v8/Q8. No new model, extra LLM call, real starting-grid retrieval
or live configuration change is included.

October 28 unavailable-data follow-up: experimental v8 now separates an
application-owned EN/TR limitation from supported, model-generated speech.
Unavailable inputs are refreshed before final grounding; fully unavailable
answers skip Core inference. The original unsupported negative damage claim is
mitigated under FP16 and Q8 in the 92-question replay. FP16 completed 92/92 turns,
Q8 91/92; planner/compound regressions, ambiguous references, Turkish units and
unproven free-form semantics remain. Keep v7/FP16 live. See INT-010, INT-011 and
[the guarded comparison](qwen-cache-trial.md#unavailable-data-safeguard-results-2026-10-28-not-promoted).
This run had iRacing active alongside synthetic replay; its timings are not an
isolated benchmark or live voice acceptance.

Priority decision, October 12: stabilize Qwen request understanding before adding
race-start features or doing further speed tuning. Reproduce the observed failures
with typed questions (without STT), expand bilingual short/social/unsupported and
dialogue controls, then compare a separately evaluated candidate against v6 before
changing the live path. Do not replace this work with a fixed spoken-command table.
The race-start briefing (actual grid position plus an evidence-backed expected finish)
is deferred under [RACE-001](known-issues.md#race-001-announce-starting-grid-and-an-expected-finishing-position).
See [the interpretation follow-up](known-issues.md#int-004-context-planner-completeness-relevance-and-social-purpose-errors).

The first understanding slice is implemented as `qwen-context-v7`, with
44 bilingual development controls and 12 disjoint validation paraphrases. Common
exact-plan/evidence credit improved from v6's 67/102 to 95/102 on the matched CUDA
workload. On October 25, the user selected v7 as the development/live-test default;
v6 remains an explicit rollback, and all seven known failures remain open. Twelve fresh
full text-pipeline checks passed with a 0.609 s median, excluding STT/TTS and game
load. Next: incomplete compounds, unresolved references, the remaining Turkish
grid/relief mistakes, bounded planner recovery, then live-catalog/race acceptance.
See [the comparison and testing instructions](intelligence-evaluation.md#october-12-understanding-candidate-v7).

October 11 branch split: the standard edition on `main` now uses Qwen3-4B for
Context planning and Core/reply generation with NVIDIA/CUDA selected in the normal
source configuration. The trained interpreter is not part of this branch's live
pipeline. CPU Qwen3-ASR, Piper/SAPI, telemetry retrieval/calculations, profiles and
grounding are unchanged. The normal launcher owns its server on port 8088 and does
not silently reuse an external CPU server or fall back. Existing packaged components
remain CPU-only pending explicit GPU distribution work.

The Qwen-only development comparison completed 12/12 typed GPU turns with a 0.772 s
median and 1.215 s sample p95; the CPU run stopped after its first 30 s timeout.
The subsequent seven-question live run had no model timeouts but four empty factual
plans failed silently, two turns were mapped to unrelated speed evidence, and
one reply made an unsupported damage claim. These correctness and error-feedback
issues remain open; acceleration does not fix them. See [GPU guide](qwen-gpu-trial.md).

Interpreter experiments, authored draft datasets, training and the bounded social
pilot are preserved on [light-version](https://github.com/furkannehir/race-engineer/tree/light-version).
Pitward Light still needs full request extraction and a measured small CPU reply
model before Qwen-4B can be removed. No trained checkpoint is promoted by this split.

The earlier October 9 checkpoint follows for historical context.

October 9 checkpoint: v6 remains the live development baseline. Correlated radio diagnostics
now distinguish capture, ASR, interpretation, Core, evidence refresh and voice delivery.
The latest CPU-only live session transcribed 12 questions, but 10 timed out in the Context
planner and only two produced speech. One of those misinterpreted a presence check as a
position request. These results do not establish MVP accuracy or predictable latency.
The next experiment is an isolated local multilingual Laya interpreter evaluation, not a
live replacement. Preserve the earlier rejected Laya adapter results described below.

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
   implementation fingerprints, and machine metadata. `qwen-context-v7` is the
   current development/live-test default, with v6 available as an explicit rollback;
   the enhanced profile fails closed until a learned temporal
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
9. The measured source-mode NVIDIA/CUDA path is now the standard edition's default.
   Packaged GPU components, compute UI controls and stable combined game-load
   benchmarks remain. AMD/Vulkan support still requires actual hardware validation.
10. Proactive usefulness learning remains separate and shadow-only until independently
   labeled evidence justifies promotion.

## Future research: after the MVP

The standard edition uses existing local models. Interpreter/domain-adaptation work
is developed separately on `light-version`; no training result automatically
establishes production accuracy or makes the standard edition depend on that branch.

- **RES-01: purpose-trained radio interpreter.** Investigate a multilingual sentence encoder
  such as MiniLM with SetFit or another lightweight learning method for English/Turkish radio
  interpretation. Cover natural paraphrases, short inputs, dialogue references, mixed purposes,
  out-of-scope requests and uncertainty rather than a prescribed spoken command set. This
  needs suitable domain examples, independent evaluation and CPU resource measurements;
  The draft datasets, first training experiments and bounded live action pilot are
  preserved on `light-version`. Full request extraction and the small reply model
  remain pending; the standard edition retains Qwen context planning.
- **RES-02: specialize the Core/response model.** Investigate adapting the local conversational
  model to a calm race-engineer teammate style and supported evidence-based reasoning.
  Application code still owns telemetry retrieval, arithmetic, provenance, freshness and final
  grounding. Do not train memorized race values into answers, weaken safety boundaries, or assume
  that fine-tuning the same model makes inference cheaper. Compare quality and latency against
  the unadapted baseline before adoption.

Laya/mDeBERTa screening and MiniLM training artifacts belong to `light-version`.
Generic "copy/checking" acknowledgments remain a later UX discussion.

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
- CPU compatibility remains available explicitly, and Pitward Light targets CPU-only
  operation. The standard source edition now defaults to NVIDIA/CUDA; this does not
  establish automatic selection, packaged GPU support or AMD/Vulkan compatibility.
- Do not treat the driver's prototype acceptance as resolution of the blue-flag defect or
  the existing conversation interpretation failures. IR-001 remains pending live
  validation even though its normalization fix and regression coverage are implemented.
