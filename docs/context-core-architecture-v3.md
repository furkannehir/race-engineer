# Context/Core Engineer architecture v3

Updated 2026-10-03. This is the authoritative design for the replacement conversational
intelligence path. It supersedes the CE-04 bounded semantic-query design and the associated
CE-05 plan. The existing conversational prototype remains available only as a migration
source and fallback reference.

## Product outcome

The driver speaks naturally. The system builds relevant evidence from live and historical
telemetry, reasons about what a race engineer should communicate, and lets Qwen produce the
actual conversation. Strict code protects evidence, freshness, radio priority and critical
calls; it does not define a command grammar or one answer template per question.

The first architecture proof must handle the same pipeline for:

1. “Where are we?” — direct current evidence.
2. “Why am I losing time?” — time-window analysis and an evidence-backed hypothesis.
3. “He has no idea about racing.” — natural teammate interaction without forcing a
   telemetry query.

## Runtime shape

    continuous
    iRacing -> normalization -> Telemetry Memory -> Context Engineer
                                                      ^       |
                                                      |       | evidence
    STT -> DriverTurn --------------------------------+       v
                                                 Core Engineer
                                                      |
                                                      | EngineerBrief
                                                      v
                                                    Qwen
                                                      |
                                                      | GeneratedResponse
                                                      v
                                          refresh -> Grounding Gate
                                                      |
                                                      v
                                                radio scheduler -> TTS

The Context Engineer is both continuous and query-aware. It incrementally maintains useful
time windows and race entities, then selects or derives evidence relevant to the current
utterance. The Core Engineer may later make a small bounded request for additional evidence;
an unbounded agent loop is not part of the first implementation.

These boxes are logical interfaces, not a requirement for a separate model per box. A
portable mode may combine Core Engineer reasoning and Qwen generation in one inference.
A stronger mode may run a continuous temporal model and a small separate Core Engineer.

## Boundary contracts

DriverTurn
: Transcript, language hints, session generation, receipt time and bounded recent dialogue.

EvidenceItem
: A measurement, deterministic derivation, learned inference or explicit unknown. Every
  known item carries a subject, metric, source sequence, timestamp, confidence and source
  fields. Inferences carry a claim; unknowns carry neither a fabricated value nor claim.

CapabilityDescriptor / CapabilityResult
: A value-free, typed calculation advertised to the Context Engineer and its deterministic
  execution result. Descriptors publish required inputs, outputs, freshness and uncertainty;
  results return sourced evidence or an explicit unavailable reason.

ContextPacket
: The Context Engineer's bounded evidence selection and situation tags for one turn. It
  cannot mix sessions, duplicate evidence IDs or cite future telemetry.

EngineerBrief
: The Core Engineer's communication decision: goal, language, tone, evidence IDs, guidance
  and confidence. It is not final speech and cannot cite evidence absent from the packet.

GeneratedResponse
: Qwen-authored natural speech. Volatile numerical values use explicit evidence
  placeholders, and every placeholder maps to one Core-approved evidence item. Qwen may
  produce evidence-free social conversation when the brief calls for it.

GroundedResponse
: Final text after refreshed evidence substitution and scope validation. Only this contract
  may enter the conversational radio path.

The initial contracts and orchestration live in core/intelligence.py and intelligence/.
They are additive and do not make the old query planner part of the new design.

## Telemetry Memory

Telemetry Memory retains bounded current and historical data independently from driver
turns:

- current normalized frame and session generation;
- recent high-frequency signal windows;
- lap and sector summaries;
- nearby opponent histories;
- normalized events and race phase; and
- driver baselines that are valid for the current car/track/session scope.

It exposes generic operations such as latest value, window delta, trend, comparison,
ranking, aggregation, change detection, correlation and event lookup. These operations are
typed numerical tools, not spoken intents. Simulator-specific SDK names stay behind the
normalization/schema boundary.

An LLM will not receive thousands of raw frames as JSON. Numerical processing belongs in
the memory/query substrate or a temporal model. Language models receive compact evidence.

## Context Engineer

The Context Engineer answers “what evidence is relevant and what does the data suggest?”
It may combine:

- direct measurements;
- generic deterministic time-series operations;
- learned temporal representations or anomaly detectors;
- semantic relevance selection for the current utterance; and
- explicit uncertainty when the simulator or history cannot support a conclusion.

The implementation is model-agnostic. Laya/MiniLM-style language encoders may help select
language-side relevance, but they are not treated as temporal telemetry models. Candidate
models must be evaluated against scripted evidence controls and replay data before live use.

INT-03 implements the first model-directed baseline. The planner receives the driver turn
and a catalog containing normalized numeric selectors, units and availability flags, but no
telemetry values. Local Qwen selects a bounded set of generic latest/window operations;
Telemetry Memory executes those operations deterministically and returns provenance-bearing
evidence. The planner may select no telemetry for a social turn and must name unsupported
analysis as an explicit unknown instead of substituting a nearby fact. This path contains no
English or Turkish keyword router and produces no spoken answer.

The INT-07 v4 repair binds raw query IDs to complete selectors from the current catalog;
the model cannot construct arbitrary source/signal/opponent triples. In the same planner
call it selects social, factual or mixed purpose independently of factual time scope.
Application-owned situation tags and missing-information reasons preserve those distinctions
for the Core Engineer. Stable value-free catalogs precede variable dialogue in the prompt
to permit prefix reuse; evidence and plans are never cached. One factual time scope per turn
is still a limitation for cross-scope compound questions. See the
[development evaluation and limitations](intelligence-evaluation.md).

The v5 planner first marks social content independently and lists bounded, transient requested
facts before selecting tools, in the same inference. Application code derives conversational
purpose; the inventory is not evidence and is discarded rather than becoming dialogue or driver
memory. This helps compound requests and distinguishes requested statistics without adding
phrase-based routing. Public evidence contracts and delivery-time refresh are unchanged.

The baseline exposes latest, window delta/mean/minimum/maximum/linear trend and current-field
count/mean/minimum/maximum. INT-06 adds a separate typed capability catalog for calculations
that need stable race semantics. Qwen selects a capability from its description; the application
executes it and returns evidence. The baseline calculates classification, fuel range, rolling
position change, current gaps, relative gap trend and constant-trend catch time. `is_last` and
relative direction are application-owned relationships, not spoken phrase commands or
model-authored comparisons. Pit/rejoin contracts remain explicit unavailable outcomes until
their real strategy inputs exist. Lap/sector comparison, event lookup and learned temporal
inferences remain schema/tool additions, not new spoken intents.

## Core Engineer

The Core Engineer answers “what should a good teammate communicate now?” It chooses whether
to inform, analyze, coach, acknowledge, clarify or remain silent. It separates facts from
hypotheses, considers race phase and radio load, and emits an evidence-backed brief.

The first version may use a local general model behind this interface. Later versions may
use a smaller distilled reasoner or policy model. Critical race-control calls remain on the
existing deterministic policy path and do not wait for this component.

The portable INT-04 baseline uses one local Qwen inference to produce both the Core decision
and the conversational response while preserving the two contracts in code. Its compact model
wire output contains only the communication goal and short English or Turkish speech. Evidence
is selected by short placeholders; application code expands those placeholders into known
evidence references and supplies the calm-teammate tone, empty guidance and default confidence
in the full brief. Social turns may be evidence-free. Unsupported facts remain unknown rather
than being replaced with a nearby measurement.

## Qwen and grounding

Qwen is the conversational generator, not merely an intent classifier. In portable combined
mode it receives the turn, compact evidence, personality and bounded dialogue context, then
writes the actual English or Turkish response while choosing the communication goal.

Generated speech cannot embed literal numeric telemetry; the model uses compact placeholders
such as `{{a}}` and `{{b}}` from the turn-specific binding table. The application reconstructs
the full evidence references rather than asking the model to repeat IDs and fields. Exact known
`[a]`-style aliases are normalized at this wire boundary; invented, malformed or repeated
aliases still fail closed. Before delivery, the Context Engineer refreshes cited evidence and
the grounding gate:

- checks turn, session, generation and language scope;
- requires the generator to use exactly the evidence approved by the Core brief;
- rejects missing, expired, unknown or nonscalar referenced evidence;
- substitutes refreshed values;
- rejects unresolved placeholders and overlong radio text; and
- hands only the grounded response to the radio scheduler.

This mechanism cannot prove every free-text statement true. Prompts, evidence-aware
evaluation and conservative handling of unverified incidents remain required. Precise
telemetry claims and numerical values have a mechanical provenance boundary.

## Runtime profiles

Portable
: Generic telemetry analytics plus a combined Core/Qwen inference. The current implementation
  performs one evidence-planning inference and one combined Core/response inference per driver
  turn.

Enhanced
: Continuous temporal context model, small Core reasoner and Qwen generator. Context work
  occurs incrementally so PTT does not trigger whole-session analysis.

CPU remains a complete supported route. Optional CUDA/Vulkan acceleration is a deployment
choice and does not change these contracts.

## Migration boundary

Carry forward:

- telemetry adapters, normalization, recording and replay;
- event derivation and critical strict policy;
- STT/TTS adapters, PTT controls and radio scheduling;
- process ownership, cancellation and timeout behavior;
- driver profiles, explicit preferences and privacy defaults; and
- session freshness and opponent identity concepts.

Replace as the primary path:

- closed RaceQuery/semantic-intent routing;
- Qwen used only as a classifier;
- one fact provider and one response template per supported question; and
- CE-04 candidate routing as the live conversational architecture.

The old fact renderer may later be ported as an emergency response fallback. It is not the
new Core Engineer or the normal Qwen response path.

## Implementation order

1. **INT-01 — implemented foundation:** contracts, component protocols, one-turn
   orchestrator, strict evidence grounding, and scripted coverage of the three architecture
   proof conversations.
2. **INT-02 — implemented foundation:** bounded Telemetry Memory with generic normalized
   signal selectors plus latest, delta, mean, minimum, maximum and trend operations.
   Session changes/rewinds reset history, ordering failures are rejected, missing or
   incomplete windows become explicit unknown evidence, and replay time drives windows.
3. **INT-03 — implemented baseline:** model-directed evidence selection over a value-free
   normalized signal catalog; bounded generic latest/window/field operations; evidence
   refresh; evidence-free social turns; and explicit unknowns without a spoken command
   grammar.
4. **INT-04 — implemented portable baseline, hardening in progress:** one local Qwen inference
   combines the Core
   decision and natural English/Turkish response, while application contracts keep their
   evidence and generation responsibilities separate. Driver-provided numeric claims are
   verified against current evidence, and unsupported projections produce a direct limitation
   rather than a request for unrelated telemetry. Driver-supplied numeric tokens are withheld
   from the generation boundary; application-provided rendering bindings are the only route for
   spoken telemetry numbers. The prompt receives a metric-aware structural example generated
   from the selected binding, so scalar signals do not need per-question response templates.
   Deterministic classification relationships protect ranking math, and one bounded repair
   inference is permitted only after an invalid draft. Broad real-model bilingual evaluation is
   still pending.
5. **INT-05 — integrated baseline, hardening in progress:** `voice-iracing` continuously feeds
   Telemetry Memory,
   uses the new Context/Core/Qwen path, refreshes selected evidence in the radio delivery
   slot, and retains existing cancellation, expiry and deterministic critical-call behavior.
   Replay diagnostics now distinguish an unreachable model from timeout and reachable HTTP
   failure. Typed temporal scope mechanically blocks current measurements from answering a
   future hypothetical when no projection capability exists. A real iRacing shakedown is pending.
6. **INT-06 — implemented baseline:** the typed deterministic race-capability registry is wired
   through planning, execution and delivery-time evidence refresh. Classification, fuel range,
   rolling position change, current gaps, relative gap trend and constant-trend catch time are
   implemented with explicit availability, provenance, freshness and uncertainty. Pit loss,
   stop duration and rejoin position are typed future capabilities that remain unavailable until
   their real strategy inputs exist. There are no phrase-specific answer branches and no
   model-authored race arithmetic.
7. **INT-07 — evaluation foundation in progress:** a versioned bilingual workload and
   content-free runner now compare temporal scope, typed capability/query selection,
   evidence availability and latency at the Context Engineer boundary. Qwen is the runnable
   portable baseline. The initial MiniLM/Laya adapters were screened and not promoted.
   Broader locked/multi-turn testing, a provenance-bearing temporal adapter, the enhanced
   profile and a measured promotion decision remain. See
   [INT-07 Context Engineer evaluation](intelligence-evaluation.md).

No old CE-04/05 component is ported merely because it exists. Each migration must satisfy
one of these new boundaries and have a focused test.
