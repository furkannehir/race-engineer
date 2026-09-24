# Conversational core: behavior and implementation design

Updated 2026-09-24. Planning companion to
[CE-04 and CE-05](context-engine-implementation-plan.md). This document defines proposed
behavior and implementation gates; it does not claim that this behavior is implemented.
The driver selected **calm teammate: acknowledge briefly, then help** as the initial tone.

Architecture revision 2 is recorded in [CE-04 architecture](ce04-architecture.md). It fixes
the component names and ownership: `DialogueSession` owns `DialogueState`, the
`ConversationContextAssembler` creates immutable per-turn input, and `SemanticJudge` is
the shared role evaluated with Laya, MiniLM, and Qwen adapters.

## Experience we are building

The engineer follows the conversation and the race together. It understands freely
phrased English/Turkish questions, remembers the relevant subject, accepts corrections,
responds naturally to a remark, and explains a limitation without guessing. Its timing
and restraint matter as much as its words. Internal labels must never become phrases the
driver has to memorize.

The first increment covers existing factual capabilities, conversational continuity,
clarification, and bounded social responses. New strategy calculations, incident blame,
unrestricted generated advice, persistent personality inference, voice preference writes,
and automatic learning remain outside this increment. A new fact still needs a validated
provider. A model cannot make an unavailable calculation available by describing it well.

## Behavioral contract

- Be concise and calm. Acknowledge frustration without judging another driver or claiming
  to have witnessed contact. Offer help when the context supports it; avoid repeating
  "focus on your race" after every complaint. Elaborate when the driver asks.
- Resolve meaning using the current utterance, a pending clarification, recent dialogue,
  and relevant race evidence. An explicit correction or topic change overrides prior
  context. Recency alone does not prove what an ambiguous pronoun means.
- Distinguish an unclear request, an understood but unsupported request, and a supported
  request whose data is unavailable. Each needs a different explanation or recovery.
- Preserve every requested part. Answer independent supported parts, then briefly explain
  unavailable parts or ask one targeted clarification. Never silently replace "fuel to
  finish" with "fuel remaining" or drop the difficult half of a compound question.
- Ask the smallest useful clarification. Permit an elliptical response such as "behind"
  or "arkadaki" to complete the pending question. Do not trap the driver in a clarification
  loop: after one unsuccessful follow-up, explain the limitation and allow a fresh request.
- Use explicit reply-language preferences first. In automatic mode, use the current
  utterance with recent language as context; one borrowed racing term must not force a
  language switch. Test both directions of code switching.
- A factual question normally receives an answer, clarification, or short unavailable
  response. Silence is reserved for a defined no-reply act or cancelled/expired output;
  failures remain visible in the panel. A closing "thanks" should not start a new exchange.
- Preserve critical-call priority, explicit communication preferences, and speech expiry.
  A conversational result cannot veto a critical warning or resurrect an expired reply.

### Example acceptance scenarios

Examples are proposed semantic expectations, not mandatory input phrases or fixed scripts.
P6 and a 2.4-second gap are illustrative synthetic values and require fresh fixture data.
Equivalent natural wording is acceptable. Review these scenarios before freezing tests.

| Situation and driver input | Expected behavior |
| --- | --- |
| Fresh session: "Position?" / "Kaçıncıyız?" | Brief factual answer, such as "P6" / "Altıncıyız." Short input alone is not invalid. |
| Gap behind was discussed: "Is he catching?" / "Yaklaşıyor mu?" | Resolve the previously established car; answer only if a valid trend exists for that same car. Otherwise say the trend is unavailable. |
| No opponent established: "Is he catching?" / "Yaklaşıyor mu?" | Ask "The car ahead or behind?" / "Öndeki mi, arkadaki mi?" Accept the short follow-up. |
| Fuel was discussed: "And now?" / "Peki şimdi?" | Refresh the fuel answer while that topic is still valid; use current data. |
| Fuel was discussed: "Actually, our position?" / "Aslında kaçıncıyız?" | Switch to position immediately. |
| "No, I meant the guy behind." / "Hayır, arkadakini sordum." | Correct the referent and recompute the intended answer. Acknowledge briefly; do not defend the earlier interpretation. |
| "That was dirty." / "Bu yaptığı hiç hoş değildi." | Neutral acknowledgment, such as "Copy" / "Anladım." No invented contact observation, blame, penalty, or diagnosis of the driver's mood. |
| "That was dirty. Gap behind?" / "Bu yaptığı hiç hoş değildi. Arkayla fark?" | Brief acknowledgment plus the verified gap. Keep the factual answer prominent. |
| "Position, and is he catching?" with no referent | Answer position, then clarify the opponent; retain the unresolved portion only. |
| "Can we finish on this fuel?" when prediction is unavailable | State that the finish estimate is unavailable. Fuel remaining may be offered as explicitly different information. |
| An answer was interrupted before its numeric part; driver says "Say again." | Do not assume the answer was heard. Repeat the intended information using fresh time-sensitive facts, or explain if unavailable. |
| "Thanks." / "Sağ ol." after a delivered answer | Close the exchange without another question; proposed default is no extra radio reply. |

Some cases legitimately admit more than one interpretation. For example, "Where are we
now?" after discussing fuel can refer to fuel or position. Annotate acceptable clarification
and context-specific answers before evaluation; do not redefine the old CONV-001 benchmark
expectation merely to accommodate a model result.

## Responsibilities and contracts

Use the existing session, context builder, fact providers, and shared radio boundaries.
The [architecture](ce04-architecture.md#names-and-ownership) defines the authoritative
component names. Keep the existing v1 planner interface while introducing the versioned
`SemanticJudge` boundary so model adapters can be evaluated independently.

| Component | Responsibility and boundary |
| --- | --- |
| Dialogue Session | Own lifecycle, turn IDs, cancellation and state commits; consume delivery events without blocking telemetry. |
| Conversation Context Assembler | Build immutable bounded context from capabilities, evidence, freshness, race features, preferences and a dialogue snapshot. No session lifecycle or history mutation. |
| Transient Dialogue State | Current topic, anchored opponent, pending clarification, bounded recent utterances/meanings, and delivery outcomes; RAM owned by the session. |
| Semantic Judge | Propose requested meanings, response acts, per-part reference bindings, unresolved parts, and evidence requirements. Laya and MiniLM are alternative adapters; Qwen is a baseline/possible fallback. |
| Judge Router | Run the selected primary and at most one eligible Qwen fallback; use the same deterministic acceptance checks for both. |
| Dialogue Controller | Validate the complete proposal, resolve answerability, apply correction/clarification rules, and propose state transitions for the session to commit. Ordinary application code. |
| Fact retrieval and renderer | Read fresh facts for resolved queries and produce short bilingual replies, including bounded acknowledgments and explicit limitations. |
| Radio and delivery feedback | Enforce priority and expiry, revalidate before playback, record completed/interrupted/cancelled/expired outcomes in transient dialogue state. |

The current `conversation-plan.v1` allows either factual queries or one clarification.
It cannot express an acknowledgment plus a query, a partial answer plus clarification, or
a deliberate no-reply decision. `ConversationRequest` carries questions/history but no
compact race context. Current history remembers questions/plans before playback; it does
not establish which engineer words were delivered. These are CE-04/05 design gaps, not
capabilities gained automatically by swapping the model.

Define versioned successor contracts before adding adapters:

- Input: turn ID, session/generation, explicit language preference, bounded dialogue view,
  compact evidence/capability view, and deadline. ASR metadata is optional and must not be
  treated as a calibrated certainty score without validation.
- Semantic proposal: requested query parts, conversational acts, scoped references, unresolved
  parts, reason codes, and model/calibration version. Separate semantic confidence from
  telemetry availability. Validate the joint result, including incompatible label choices.
- Response decision: answered/partial/clarify/acknowledge/unavailable/no-reply, grounded
  fact references, selected wording, expiry and pending-state update. No-reply is explicit,
  not an empty string forced through the existing nonempty reply contract.
- Delivery event: turn and response IDs, generation, started/completed/interrupted/cancelled/
  expired/failed state, and delivery extent if actually known. Playback completion is only
  an operational observation; it does not prove that the driver heard or understood it.

Preserve or explicitly reject old serialized versions with tests. Keep the existing v1
Qwen baseline runnable; a v2 Qwen adapter needs its own validation before being used as
fallback for new social or mixed-act semantics. Schema validity alone is insufficient.
Treat transcripts and external names as input data, never authority to change preferences,
disable safeguards, write memory, or execute tools.

## Memory, state, and timing

Keep three distinct scopes: bounded transient dialogue, current-session race evidence,
and durable explicit driver preferences. No transcript/audio storage or inferred emotion,
personality, incident, or preference writes are introduced by this work.

- Let the session commit all dialogue updates; the controller proposes transitions and
  the assembler and model adapters receive immutable snapshots.
- Track the driver's received request separately from a proposed engineer response and
  from actual delivery. An unanswered driver question can still establish a topic; an
  undelivered engineer answer cannot establish that the driver received its information.
- A pending clarification has its originating turn, unresolved request, allowed choices,
  and expiry. A new explicit request supersedes it. Specify topic/reference/clarification
  lifetimes in CE-04b and test clock boundaries rather than leaving them implicit.
- Anchor an opponent reference to stable identity when available. If identity changes or
  cannot be established, clear or clarify the reference; never silently transfer "he" to
  a different car occupying the same relative position. Derived trends need the same rule.
- Clear session-scoped state on session change, connection generation change, replay seek,
  and explicit reset. Retain explicit profile settings. Discard late results by generation
  and turn ID even if cancellation of the underlying model is unavailable.
- Recheck facts after inference and before playback. Define separately whether a stale
  session may receive a neutral, fact-free acknowledgment; the proposed initial rule is
  to retain the existing unavailable response on stale/disconnected input. Relaxing that
  rule requires separate tests and must not reopen stale factual answers.
- A newer request or critical interruption must not create a backlog of obsolete replies.
  Fix queue/deadline/resume rules in CE-04b using existing radio limits. Keep inference off
  the telemetry path. Run interpretation per driver turn and proactive judgment per
  eligible candidate change, not per telemetry frame.
- Ask once for a repeat when input is unintelligible; do not report a hardware/model failure
  as though the driver spoke unclearly. ASR, semantic uncertainty, missing data, and model
  timeout have distinct outcomes and diagnostics.

## Evaluation before choosing a model

The CE-02 report remains a descriptive v1 baseline. Its locked dataset has 200 independent
turns and no multi-turn sequences. The current scorer defines supported cases by
non-clarification expectations, which also includes unsupported/unavailable requests.
Family validation compares identifiers; it does not establish semantic separation. A
2026-09-24 audit also found one exact question shared by calibration and locked groups.
Do not silently alter that dataset or retrospectively reinterpret its recorded score.

Build a separately versioned dialogue suite and report for CE-04:

1. Start with paired English/Turkish scenarios covering the examples above, ordinary facts,
   negation, compounds, ambiguous references, missing facts, short/noisy ASR text, and code
   switching. Include the same utterance under different topics and race states, repeated
   "and now?" turns, opponent replacement, interruptions, and session resets. Preserve
   CONV-001 through CONV-003 as visible regressions with documented v1/v2 semantics.
2. Annotate acceptable meanings, acts, references, factual availability, clarification
   targets, state transitions, and forbidden claims. Allow predeclared equivalent outcomes
   for genuinely ambiguous inputs. Score whole dialogues as well as individual turns.
3. Split by underlying scenario/paraphrase family, keeping translations together. Audit
   literal and semantic leakage before freezing. A family ID must be independent of its
   split/language; merely prefixing IDs does not establish independence. Label classes
   should still appear across splits through different scenarios. Context-dependent repeated
   wording is legitimate within a dialogue and needs turn IDs, not blanket text uniqueness.
4. Distinguish understood, currently answerable, unsupported, clarification-needed, and
   fact-unavailable cases. Report factual answer coverage on answerable requests; report
   social handling, useful clarification, compound completeness, and fallback separately.
   The whole accepted proposal must be correct, including acts and references. Freeze v2
   denominators before model runs; publish alongside, not as a silent replacement of, v1.
5. Compare all candidates with the same information and scenario budgets. Keep legacy Qwen
   results separate from full-context v2 Qwen results. Calibrate on calibration data only;
   freeze thresholds/prompts before locked evaluation. Repeated tuning requires new held-out
   evidence, not reuse of a now-development test set.

Keep the plan's per-language 95% accepted-plan accuracy and 70% answerable-case coverage
targets, documenting the v2 coverage definition. Report counts and uncertainty, plus
per-category and full-dialogue failures. Require zero fabricated facts, wrong-car factual
answers, stale playback, and critical-call interference in the safety scenarios; this
finite test result is not a guarantee about all races. Test a candidate that abstains on
everything to ensure it cannot pass through accuracy alone.

Measure total interpretation work, context preparation, all hypotheses/heads, fallback,
timeouts, resident memory, and cold starts. Preserve the CPU p95 target of 300 ms for the
small-judge stage and the separate end-to-end targets. Laya's binary decisions and MiniLM's
hypotheses must compose into a coherent plan; batching does not imply free work or a single
classification sufficient for a whole turn. Neither candidate is preselected.

Use deterministic fake adapters for the controller and radio tests first. Then compare
local models offline. Finally conduct a small bilingual listening review, initially over
replay: correctness, continuity, usefulness, tone, and timing are separate ratings from
voice quality. Real audio uses deliberately supplied samples; no new race is required
for the initial dialogue work. TTS-001 and STT-001 remain separate issues.

## Reviewable delivery slices

These sub-slices refine the existing CE numbering; they do not reorder the roadmap.

| Slice | Deliverable | Completion evidence |
| --- | --- | --- |
| CE-04a | Behavior contract, architecture revision 2 and bilingual scenario review | Design and component ownership documented; calm-teammate tone recorded; scenario expectations reviewed before freezing |
| CE-04b | Versioned contracts, session/state, context assembler and controller | Scripted SemanticJudge and fake-clock tests for corrections, clarification/reference expiry, valid compounds, delivery state, text-only mode and resets; exact bounds documented |
| CE-04c | Dialogue evaluation protocol, fixtures and split audit | Distinct v2 metrics and leakage checks; frozen calibration/test scenarios; old baseline preserved |
| CE-04d | MiniLM, Laya and v2 Qwen evaluation adapters | Same-context CPU comparisons, calibrated joint decisions, bounded local-only execution; no implicit live replacement |
| CE-04e | Evidence-backed routing decision | Select the reliable scope for each route, fallback/residency cost, or retain Qwen if alternatives do not earn promotion |
| CE-05 | Opt-in live integration and calm-teammate responses | Fresh grounded replies, delivery-aware memory, partial answers, bounded social acts, graceful failure, radio and listening-review gates |

The next implementation task is CE-04b using architecture revision 2 and the CE-04a
scenarios; complete their review before freezing the CE-04c evaluation. Contract,
state, and evaluation work can proceed without choosing the winning model. CE-06 proactive
usefulness remains a separately labeled/shadow-evaluated task. Sharing context does not
make a question interpreter qualified to decide when to interrupt the driver.

Still evidence-driven: model choice, exact confidence/lifetime settings, fallback residency,
and minimum hardware. Broader personality controls and freer generated wording can follow
once grounded dialogue and timing are dependable. CE-03 remains immediately before CE-08.
