# Local context engine and portable inference implementation plan

Updated 2026-09-24. Status: CE-01 and the CE-02 typed baseline are implemented; CE-04 is next.
CE-04 architecture revision 2 is documented; its first implementation slice is CE-04b.

This is the next implementation increment after the accepted live prototype. It combines
the smaller local context-judge evaluation with optional CPU/GPU execution and resource
controls. It extends M3 conversation, M4 ranking, and M5 controls in the
[working roadmap](roadmap.md). The [original implementation plan](personalized-ai-race-engineer-implementation-plan.docx)
remains the historical milestone baseline; this plan supersedes its recommendation to
implement a hosted Jev adapter next. The current scope is entirely local.

## Outcome and existing foundation

Deliver an English/Turkish engineer that understands natural questions, follow-ups, and
bounded conversational remarks, and assesses noncritical calls using current race context.
CPU execution must remain a supported complete path. Optional acceleration must include an
AMD route, with performance judged alongside the running simulator.

| Foundation | Current state | Work carried into this plan |
| --- | --- | --- |
| iRacing telemetry and strict policy | Implemented; IR-001 normalization fix has regression coverage | Genuine-blue live validation remains open |
| Context and grounded answers | Bounded events, fuel consumption, gaps, battle state, fresh fact retrieval, deterministic replies | Shared compact context, explicit uncertainty, query coverage |
| Local voice conversation | Qwen3-ASR, Qwen3-4B planning, Piper replies, shared live radio | Stage timings, smaller planners, natural acknowledgments |
| Hardware execution | Conversation launcher uses the CPU binary, eight threads, zero GPU layers; ASR supports CPU/CUDA settings | Configurable runtimes, CPU limits, CUDA/Vulkan validation |
| Driver controls and memory | PySide6 panel, wheel PTT, SQLite preferences and content-free automatic-call history | Compute controls and separate shadow observations |
| Adaptive judgment | Not implemented | Evaluation, labels, calibration, shadow mode, gated promotion |

The documented Qwen smoke result is 18/20 development turns and 10/11 holdout turns, with
roughly 1-3 seconds for warm text turns. It is neither an in-race benchmark nor an
end-to-end speech result. Preserve the failing CONV-001 through CONV-003 cases.

## Product and architecture decisions

- English and Turkish, free phrasing, and local inference remain requirements. Internal
  intent labels describe supported meanings; they are not a spoken command vocabulary.
- The first conversational tone is the driver-selected calm teammate: acknowledge briefly,
  then help. Define continuity, correction, clarification, and delivery-aware behavior in
  the [conversational-core design](conversational-core-design.md) before choosing a judge.
- Retain Qwen as the baseline and optional interpretation fallback while alternatives are
  evaluated. Selecting MiniLM or Laya for an experiment does not select either as the
  production model.
- Follow [architecture revision 2](ce04-architecture.md): `DialogueSession` owns transient
  `DialogueState`; `ConversationContextAssembler` builds immutable per-turn context;
  `SemanticJudge` proposes meaning; `DialogueController` validates proposals and returns
  state transitions for the session to commit. The controller is ordinary application code.
- Retain `ConversationPlanner` for the current v1 path and introduce the versioned
  `SemanticJudge` interface for CE-04. Reuse `DefaultRaceContextBuilder` calculations and
  existing fact/radio boundaries; adapters remain independent of the domain contracts.
- Select one primary semantic judge after evaluation. Laya and MiniLM are alternatives,
  with Qwen as the comparison baseline and possible single fallback. Retaining Qwen as
  primary is a valid outcome. CE-06 usefulness ranking is a separately evaluated role.
- Facts and derived quantities come from deterministic, capability-aware code. A judge
  selects a query, response act, or candidate; it cannot supply race numbers or invent
  telemetry. New information still needs a validated fact provider.
- Critical calls bypass learned judgment. Explicit preferences, expiry, cooldowns,
  deduplication, and radio priority remain authoritative for all noncritical suggestions.
- Inference runs outside the telemetry task, with bounded queues, deadlines, cancellation,
  and rejection of results from an old session or connection generation.
- CPU is the initial default and a required release path. GPU mode is optional. Automatic
  mode becomes recommended only after measured configurations pass the release gates.
- Ship the first support matrix for Windows/iRacing. Preserve adapter boundaries for Linux,
  macOS, and other simulators without claiming those complete applications are supported.

| Route | Intended responsibility | First implementation |
| --- | --- | --- |
| Driver question or remark | SemanticJudge resolves topics, per-part references, supported queries, and response acts | Laya and MiniLM as alternative candidates evaluated against v2 Qwen |
| Ambiguous or complex request | Recover a valid query plan within the remaining deadline | Optional Qwen call, then clarification/unavailable if unresolved |
| Proactive noncritical candidate | Estimate usefulness from derived numeric/categorical features | Deterministic baseline, then calibrated logistic regression in shadow mode |
| Facts and speech | Retrieve fresh facts, render short bilingual text, schedule playback | Existing deterministic renderer and radio, extended with bounded acknowledgments |

With a promoted small primary, common turns should complete with one bounded judge request
and deterministic composition. Account for every internal hypothesis/batch/pass. Optional
Qwen fallback runs at most once for unresolved interpretation within the turn deadline;
missing telemetry and genuine reference ambiguity go directly to an appropriate limitation
or clarification. Every model returns a validated proposal; unrestricted generated wording
is outside this increment.

## Hardware policy

Model choice, inference runtime, and hardware backend are separate configuration choices.
Settings are machine-local; they do not belong in a portable driver profile.

| Execution target | Conversation backend plan | Delivery boundary |
| --- | --- | --- |
| Windows CPU | Existing llama.cpp CPU path | Mandatory baseline, including machines without usable GPU inference |
| Windows NVIDIA | CUDA; Vulkan as an alternative | Optional initial acceleration target |
| Windows AMD | Vulkan | Optional initial acceleration target; actual AMD testing required |
| Linux AMD | HIP/ROCm or Vulkan | Later runtime validation, not a promise of Linux live iRacing support |
| Apple Silicon / Intel GPU | Metal / Vulkan or SYCL respectively | Later adapter and hardware validation |

These are backend candidates supported upstream, not a claim that every GPU, driver,
model, or quantization works. The pinned runtime must be verified against its own options.
llama.cpp documents backend/device selection, including `--list-devices`. A GPU-enabled
build can still use the GPU with zero offloaded layers; CPU mode must use a CPU-only build
or a verified full-disable option such as `--device none`.
[Upstream runtime documentation](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md)

The panel exposes CPU, GPU, and Automatic choices plus the effective mode and a short
fallback reason. GPU means an optional acceleration request; if unavailable, a visible
CPU fallback is allowed. Advanced settings select the device, threads, context limit,
and offload amount. A memory budget is an admission target, not a guaranteed hard VRAM
cap; measure actual allocations, including caches and other model workers.

Automatic mode considers installed compatible runtimes and a validated local profile.
An unfamiliar configuration defaults to CPU and can run an explicit pre-session test.
GPU detection or free VRAM alone is insufficient evidence to choose full offload. CPU
inference also competes with the simulator, so compare both routes. Partial offload is a
benchmark candidate rather than an assumed improvement.

Runtime failure triggers bounded cleanup and at most one CPU restart for that failure;
discard the interrupted or expired reply. If CPU startup also fails, show conversation
unavailable while telemetry and deterministic automatic calls continue. Avoid switching
backends repeatedly during a race. Never terminate a server the application does not own;
an external server's unverified device configuration must be shown as unmanaged.

ASR, the judge, conversation, and TTS report their effective execution separately. Vulkan
support for the conversation server does not accelerate the Transformers ASR worker.
Keep CPU ASR/TTS supported; evaluate AMD ASR separately only if measurements justify it.
ONNX, DirectML, or ROCm are candidates, not established drop-in Qwen3-ASR backends.

## Delivery sequence

Deliver each slice as a reviewable change with its own evidence. No model downloads, race
recording, or engine switches occur merely by accepting this plan. Slice identifiers stay
stable for history; the rows below are in the current execution order.

| Slice | Status | Deliverable | Dependency / completion gate |
| --- | --- | --- | --- |
| CE-01 | Implemented | Model-neutral planner configuration and shared runtime launcher | Existing CPU behavior preserved; backend discovery and fallback tested |
| CE-02 | Implemented (typed baseline) | Bilingual evaluation and performance baseline | Reproducible Qwen CPU report and frozen promotion criteria recorded; speech stages await supplied audio |
| CE-04 | Next; architecture revision 2 documented | Session/state, context assembler, SemanticJudge adapters and dialogue evaluation | CE-02; CE-04b fake-driven foundation, reviewed scenarios, v2 bilingual quality/resource report |
| CE-05 | Planned | Hybrid conversational routing and bounded social replies | CE-04; grounded outputs, bounded fallback, radio regression gates |
| CE-06 | Planned | Proactive usefulness baseline and shadow evaluation | CE-02 and CE-04 context; labels defined, audible behavior unchanged |
| CE-07 | Planned | Trained small judges and targeted alternatives | CE-04/06 labeled datasets; held-out improvement demonstrated |
| CE-03 | Deferred before CE-08 | Optional CUDA/Vulkan execution and compute controls | Stable CE-05/06 workload; real NVIDIA and AMD reports before claiming support |
| CE-08 | Planned | Promotion, conservative Automatic mode, and release evidence | CE-03/05/06; CE-07 only for models being promoted |

CE-04 proceeds next. CE-03 remains technically independent but is deliberately scheduled
after CE-06 and any selected CE-07 work, immediately before CE-08, so acceleration is
measured against the stable combined workload intended for release. Pull it forward only
if CPU performance blocks CE-04/05/06 development. Missing AMD access blocks an AMD support
claim, not CPU work. CE-07 is optional: it must demonstrate value before becoming a release
dependency.

### CE-01: Configurable planners and runtime foundation

- Move model identity, local model/server paths, context size, threads, device/backend,
  offload, and startup limits out of the UI launcher and standalone script into a shared
  configuration and launch service. Preserve the existing config or provide explicit
  versioning/migration when compatibility changes.
- Generalize `LocalQwenPlanner` and its hardcoded model config while preserving the
  loopback-only, schema-constrained request behavior. Application composition chooses an
  adapter through a factory; domain contracts stay independent of model libraries.
- Define backend capability records, bounded device probes, effective configuration, and
  failure reason codes. Detect from the installed runtime, not only the GPU vendor name.
- Keep a conservative configurable CPU thread budget across workers; do not assign every
  worker all available cores. Defaults must account for possible worker overlap.
- Preserve process ownership, hidden Windows workers, cancellation, local-only loading,
  and no runtime downloads. Reusing an external server must not imply verified CPU mode.

Exit: existing CPU replay/live composition remains usable, fake-runtime tests cover
unsupported devices, probe timeout, launch failure, fallback, and owned-process cleanup,
and CLI/panel use the same launch policy. Real GPU performance is not an exit claim here.

### CE-02: Evaluation harness and baseline

- Extend `scripts/evaluate_conversation.py` to select planners and emit versioned results
  with model revision, runtime/build, quantization, backend, threads, context limit,
  hardware/driver, dataset revision, and warm/cold state.
- Expand typed English/Turkish cases: paraphrases, one-word requests, negation, mixed
  languages, contextual follow-ups, ambiguous opponents, unsupported compound requests,
  emotional remarks, stale/disconnected data, and session changes. Keep existing failures.
- Separate training/development, calibration, and locked test sets by scenario and
  paraphrase family, including translated counterparts. Start with at least 100 held-out
  turns per language; enlarge the set when a category has too few examples to assess.
- Measure planning separately from capture/ASR, fresh retrieval/rendering, TTS synthesis,
  radio waiting, and first playback. Report PTT release-to-first-audio as the end-to-end
  metric, including fallback, failures, and timeouts rather than successful turns alone.
- Record p50/p95 latency, model startup, peak RAM/VRAM, CPU/GPU use where available,
  fallback frequency, accepted-answer accuracy, coverage, and clarification accuracy.
  Content-free timing records may be enabled; conversational content stays unrecorded.
- Run the existing Qwen CPU configuration as the baseline before tuning. Measure typed
  replay now; real audio uses deliberately supplied test samples. No new race is needed
  for the first report.

Exit: reproducible baseline plus frozen datasets, workload definitions, and thresholds.
Use small deterministic fakes in normal CI; large model/hardware evaluations are explicit
local suites, with no model download or microphone activation in ordinary tests.

### CE-04: Dialogue foundation, shared context and semantic judge evaluation

- Follow the [behavior design](conversational-core-design.md) and
  [architecture revision 2](ce04-architecture.md). CE-04a records behavior/ownership;
  CE-04b builds versioned contracts, session/state, context assembly, and the controller
  using a scripted SemanticJudge. CE-04c freezes the reviewed dialogue evaluation and
  split audit; CE-04d compares adapters; CE-04e selects routing from measured evidence.
- Make `DialogueSession` the only writer of conversational state. Validate turn/session/
  generation and relevant state revision on every proposed transition and delivery event.
  Keep `ConversationContextAssembler` stateless and independent of model dependencies.
- Keep received driver requests, proposed replies, and delivery outcomes distinct. Define
  pending clarification, topic/reference expiry, correction, opponent identity, and session
  reset behavior. A planned reply is not evidence that the driver heard its contents.
- Define a compact versioned context view: current capabilities, relevant recent events,
  validated derived features, freshness/session identifiers, bounded dialogue topic and
  opponent reference, explicit language preference, and recent radio categories.
- Reuse existing context calculations. Add gap trends or other derived facts only with
  valid timestamps, consistent opponent identity, reset/missing-data semantics, and replay
  fixtures. Unsupported calculations remain explicitly unavailable.
- Define a typed semantic proposal: factual request parts, per-part references,
  conversational acts, clarification or abstention, reason code, and model/calibration
  version. Allow mixed requests and preserve
  unsupported parts. Changing serialized conversation semantics requires a new schema
  version and migration or explicit rejection tests.
- Evaluate `MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli` on CPU first. Use bounded
  hypotheses for topics/acts and batched evaluation; account for work per hypothesis and
  context length. Confirm the model's token limit without silently losing the current turn.
- Evaluate `convaiinnovations/laya-multilingual` as a separate typed-decision candidate.
  Experiment with bounded per-label two-option `choice` questions for queries and acts;
  measure all passes/batches and validate joint consistency. Do not assume a single-choice
  result represents compound requests or that independent binary decisions form a coherent
  plan. Compare one multilingual resident model with Laya's higher-memory routed
  English-plus-multilingual mode.
- Treat Laya's upstream results as screening evidence only. Its multilingual checkpoint is
  322M parameters, publishes moderate Turkish MASSIVE intent accuracy, ships uncalibrated,
  and reports weak zero-shot typed-decision results. Pin local checkpoint/package revisions,
  disable runtime downloads, measure CPU memory/latency, and defer domain fine-tuning to
  CE-07 unless its zero-shot/calibrated result already earns promotion.
- Calibrate acceptance on separate bilingual data. NLI scores, embedding similarity, and
  model-reported probabilities are not trusted calibration. Unknown references, low
  evidence, and unsupported language/context must permit abstention.
- Validate schema, reference consistency, and complete mixed-request semantics through the
  same controller for every adapter. Valid requests can mention both ahead and behind;
  contradictory bindings within a request part require abstention or clarification.
- Use a bounded persistent local worker for selected small-model inference and the existing
  loopback server for Qwen. Measure startup/residency with ASR/TTS; normal fake tests must
  import no optional model packages. Keep all inference off the telemetry task.
- Add a separately versioned multi-turn evaluation with explicit answerability, partial
  replies, corrections, delivery outcomes, and equivalent acceptable interpretations.
  Audit split leakage and metric denominators as detailed in the design; retain the CE-02
  v1 report unchanged. Compare full-context candidates fairly, including a validated v2
  Qwen adapter rather than assuming the current factual-only fallback handles new acts.

Both initial choices are experiments rather than assumptions of race-domain accuracy.
[MiniLM model card](https://huggingface.co/MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli)
and [Laya multilingual model card](https://huggingface.co/convaiinnovations/laya-multilingual).

Exit: MiniLM and Laya comparisons with Qwen for each language and difficult-input category,
including accuracy-versus-coverage, total judge latency, resident memory, calibration, and
multi-turn limitations. A candidate may be rejected without blocking the other. No live
replacement is implied by merely obtaining faster inference.

### CE-05: Hybrid driver conversation

- Route accepted small-judge results directly to the existing fresh fact retrieval and
  deterministic renderer. Send uncertain/complex requests to Qwen once if enabled and
  within the remaining turn deadline; otherwise clarify or report unavailability.
- Expose fallback residency explicitly: disabled, loaded on demand, or kept warm. Account
  for the RAM/VRAM and cold-start cost. A small front-end judge does not reduce resident
  memory if Qwen remains loaded. Lightweight operation remains usable without Qwen, with
  clarification for cases outside the judge's reliable coverage.
- Add bounded bilingual acknowledgment/refocus acts for CONV-004, including mixtures of
  emotion and factual questions, following the calm-teammate behavior scenarios. Use brief
  acknowledgment and relevant help; avoid repetitive or unsolicited admonishment.
  Claims such as "We saw the contact" require actual supporting evidence; driver reports
  do not become independently verified events.
- Integrate delivery-aware dialogue state, partial answers with targeted clarification,
  correction, and explicit no-reply acts. Validate these through complete replay dialogues
  and bilingual listening review before live promotion.
- Extend reply refresh to preserve approved social/clarifying acts and per-part opponent
  references while retrieving new facts. Current factual-only refresh cannot carry these
  semantics. Feed idempotent, ID-tagged radio outcomes back to the owning session and
  distinguish displayed text from completed speech.
- Preserve explicit language preferences, critical interruption, speech expiry, bounded
  dialogue memory, and fact refresh after inference and before playback. Cancel unfinished
  inference when possible and always ignore late results. Avoid duplicate model stages
  on ordinary questions and duplicate speech after recovery.

Exit: replay and opt-in live routing pass the quality gates, all known ambiguity cases
remain represented, and no model can add race values or bypass scheduling. Failure of
both planners produces a bounded response; telemetry and critical calls keep running.

### CE-06: Proactive judgment and shadow evaluation

- Expose eligible noncritical candidates and context to a separate ranker. Deterministic
  eligibility rules remain authoritative; learned ranking cannot resurrect a forbidden
  candidate or veto a critical one.
- Define usefulness labels before training: useful now, potentially useful later, redundant,
  distracting, or unknown. Evaluate a deterministic baseline and calibrated logistic
  regression on typed features; consider boosted trees only if measured quality warrants it.
- Begin with synthetic/replayed, manually annotated event windows. Existing SQLite playback
  outcomes are operational observations, not usefulness labels: completed does not mean
  useful, cancelled does not mean unwanted, and silence does not mean approval.
- Run model evaluation on candidate/event changes, not every telemetry frame. Shadow work
  has a bounded queue and yields to driver requests; critical output never waits on it.
- Store versioned content-free shadow decisions, disagreement reasons, model/calibration
  versions, and timings separately from production outcomes, preserving retention and
  idempotence. Reproducing a decision uses explicit fixture/dataset IDs; feature snapshots
  for training need a separate opt-in dataset, not silent expansion of history storage.
- Define a later session-review/UI route for explicit feedback and provenance. Verbal
  feedback, inferred preference writes, online learning, and contextual bandits stay deferred.

Exit: shadow mode changes no audible decisions; annotated replay reports measure useful
calls retained, distracting calls suppressed, and timing/resource overhead. Promotion
requires independent labels, not simply agreement with the strict policy.

### CE-07: Train smaller judges and evaluate alternatives when needed

- Trial multilingual E5-small embeddings with logistic heads, then SetFit fine-tuning if
  needed. Compare before/after using the same locked test set and independent calibration.
  Begin with roughly 25-50 varied examples per label per language, adding data where
  learning curves and errors show gaps; that count is a starting point, not sufficient proof.
- Include real ASR errors only from supplied or explicitly consented local samples, with
  provenance and deletion rules. No default transcript/audio collection or automatic
  uploads; no assumption that operational SQLite history contains this training data.
- Preserve one encoded input with small task heads where useful. Evaluate ONNX/INT8 export
  for both quality and CPU latency, including a fresh calibration check after quantization.
- If Laya demonstrates useful bilingual signal but misses CE-04 gates, evaluate a
  race-domain decision-head fine-tune and recalibration here. Do not train against locked
  cases or promote it merely because its API already resembles the desired judge output.
- Keep mDeBERTa as an optional NLI quality comparison, FunctionGemma as a possible trained
  query router, and EuroMoE as a possible generative comparison. These are research options,
  not required installations. Require English/Turkish evidence and license review for each.

E5-small is a multilingual embedding candidate; SetFit supplies a training approach for
sentence-transformer classification. Neither is a drop-in conversational model.
[E5 model card](https://huggingface.co/intfloat/multilingual-e5-small)
and [SetFit implementation](https://github.com/huggingface/setfit).

Exit: promote only an alternative that improves the measured quality/resource trade-off
on the intended hardware. Retaining MiniLM, Laya, or Qwen is a valid outcome.

### CE-03: Optional acceleration and user controls

- Provide pinned, checksum-verified optional llama.cpp CPU, CUDA, and Vulkan runtime
  setup, with dependency/license notices. Installation is an explicit setup action.
- Benchmark zero, partial, and full offload with the same weights, prompt, and context
  budget. Test schema-constrained replies on each backend rather than assuming parity.
- Add the panel's CPU/GPU/Automatic selection, effective backend/device, and fallback
  status. Apply changes on the next start and expose advanced limits in configuration.
  Until promotion, CPU remains default and Automatic is labeled experimental.
- Test unsupported drivers/devices, insufficient memory, GPU-worker failure, bounded CPU
  recovery, external-server ownership, and missing optional packages.
- Keep ASR, the promoted judge, and Piper on CPU initially unless a separate experiment
  shows a benefit. Measure the combined CE-05/06 worker load rather than optimizing the
  conversational server alone.

Exit: CPU requires no CUDA/ROCm installation. A tested NVIDIA/CUDA path and an actual
AMD/Vulkan run have equivalent functional coverage and visible fallback. Without AMD
hardware evidence, retain that route as experimental and request community validation.

### CE-08: Race validation, promotion, and open-source readiness

- Test representative hardware: a mainstream 4-6-core/16 GB RAM CPU configuration with GPU
  inference disabled, an NVIDIA system with constrained VRAM, an AMD/Vulkan system, and
  the current high-end development machine. These are test targets, not published minimum
  requirements. Reduced thread counts on a powerful PC do not substitute for older hardware.
- Compare identical simulator workloads: game alone, current engineer, small-judge hybrid,
  and shadow ranking; compare CPU/partial/full offload where supported. Repeat runs and
  report variance, p99 frame time, 1% low FPS, memory, and telemetry gaps. Use a repeatable
  replay/practice workload before a real-race session; finish with an endurance run.
- Promote Automatic mode only for configurations supported by those measurements. Retain
  CPU override, visible fallback, and an easy rollback to the original planner/strict policy.
- Promote reactive routing and proactive ranking independently. A conversational pass does
  not authorize ranking to alter noncritical calls; the ranker needs its own live opt-in gate.
- Publish tested hardware/runtime versions, minimum/recommended requirements, limitations,
  offline setup, and model/voice/runtime attribution. Choose the repository license before
  presenting a distributable open-source release; model and voice licenses need separate
  review. Optional accelerated dependencies must not burden CPU-only installation.

Exit: documented CPU and AMD/NVIDIA results, no missing critical-call regression, stable
radio/telemetry during the workload, clear fallback, and a reproducible installation path.
An untested platform stays experimental. A release may retain strict proactive policy if
the learned ranker has not earned promotion.

## Acceptance and measurement gates

Correctness gates are mandatory. Numeric performance/quality goals below are initial
engineering targets, not achieved results or promises for all hardware. CE-02 freezes the
test protocol and any justified revisions before selecting a winner.

| Area | Gate / initial target |
| --- | --- |
| Critical behavior | All critical-call, expiry, interruption, session/reset, and missing-fact fixtures pass; no learned veto or generated numeric facts |
| Reactive quality | At least 95% correct complete plans among accepted small-judge results in each language; target at least 70% coverage of supported cases; report sample sizes and uncertainty |
| Difficult inputs | Dedicated ambiguity/unsupported/stale/social suites; no hidden removal of CONV-001 through CONV-003; assess combined router correctness as well as the small judge |
| Judge latency | Initial target warm CPU p95 at or below 300 ms on the selected mainstream reference PC; all heads, hypotheses, and context preparation included |
| End-to-end benefit | Target at least 25% lower warm PTT-release-to-first-audio p95 on common supported requests versus the same-hardware baseline; report full mixed workload and cold fallback separately |
| Simulator impact | Initial target no more than 5% worsening of p99 frame time or 1% low FPS versus game alone on the fixed workload, assessed over at least three runs with variance |
| Telemetry and endurance | No new inference-attributable telemetry-gap regressions or critical-call deadline failures; complete a 60-minute combined session with bounded memory/queues and cancellation checks |
| Proactive promotion | Zero critical suppression; on labeled noncritical cases retain at least 95% of useful-now calls while reducing distracting/redundant calls versus strict baseline; report per-category counts |
| Portability and failure | CPU works without GPU packages; real AMD/Vulkan and NVIDIA/CUDA evidence for their support claims; bounded failure recovery and visible degraded state |
| Privacy and reproducibility | No default audio/transcript/prompt storage; pinned artifacts, dataset/calibration versions, reproducible local reports, no network inference |

If a classifier abstains on everything, low error alone does not pass. If ASR/TTS dominate
latency, a faster planner may miss the end-to-end target; record that evidence and prioritize
the relevant speech work rather than claiming the judge solved speech latency. Fallback
latency and model residency are included in the evaluation, not hidden by warm benchmarks.

## Immediate next task and deferred decisions

CE-01 is implemented in `config.py`, conversation adapter/factory composition, the shared
server-launch service, `ui/runtime.py`, and `scripts/start_conversation_model.py`, with
targeted config/lifecycle tests. The shipped execution remains Qwen on CPU. CE-02's
versioned bilingual datasets, content-free report schema, and frozen promotion protocol
are implemented. The owned-runtime CPU baseline recorded 164/200 exact turns, 1.28-second
warm planning p95, and about 5.27 GiB peak working set. CE-04 compact context/judge
work proceeds next using the [revised architecture](ce04-architecture.md) and
[delivery sub-slices](conversational-core-design.md#reviewable-delivery-slices).
Implement CE-04b contracts, session/state, assembler and controller with a scripted judge;
review scenario expectations before freezing CE-04c, then compare Laya/MiniLM/v2 Qwen.
The v1 report does not establish multi-turn interaction quality;
the new suite must address its split and coverage-definition limitations. CE-03 acceleration
is deliberately deferred until immediately before CE-08; none is an automatic default
replacement.

Decide from evidence later: the production judge and calibration thresholds, model
residency, CPU/offload presets, exact minimum hardware, ASR acceleration/replacement, and
whether a trained ranker improves calls. Keep TTS-001 voice replacement and STT-001 capture/
recognition improvements separate unless timing/quality evidence makes them a prerequisite.
TTS-001 now tracks FreyaTTS-small as a Turkish-only evaluation candidate while retaining
Piper for English/fallback; no dependency or runtime change is selected. CONV-004 is
explicitly scheduled in CE-05. Real-race history/panel checks and genuine-blue IR-001
validation remain open and can run alongside these slices. M6 remains deferred.
