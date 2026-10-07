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

**Candidate review, 2026-09-24:**
[FreyaTTS-small](https://github.com/freyavoiceai/FreyaTTS) is one Turkish-specific option
for a later comparison, not a preferred or selected replacement. Its published model is a
183M-parameter, character-level Turkish synthesizer with a deterministic default voice,
48 kHz mono output, and Apache-2.0 code and weights. The authors report 8.0% WER / 3.0%
CER on their 495-sentence Turkish evaluation set, RTX 4090 RTF 0.10-0.11 with roughly
1.5 GB VRAM, and Apple M3 CPU RTF 0.70. Their seven-rater study favors Freya's naturalness,
although the top MOS confidence intervals overlap and Piper achieves lower WER. These are
upstream measurements, not Race Engineer or Windows/iRacing results.

Freya is Turkish-only and depends on PyTorch plus the VoxCPM2 AudioVAE. Its current local
model path still fetches that VAE through Hugging Face, and its dependency versions are
not pinned. Integration must therefore use separately pinned, checksum-verified local
model/VAE assets in an optional isolated worker with network access disabled. Keep Piper
for English and as the lightweight fallback; route by validated reply language behind the
existing `ConversationSpeaker` protocol. Do not add Freya/PyTorch to the default install.

Before promotion:

- audit and retain Freya, AudioVAE, package, dataset, and voice/model distribution terms;
- compare Piper and Freya with blind driver listening on short race calls, acknowledgments,
  Turkish characters, driver/track terms, positions, lap numbers, fuel, and decimal gaps;
- verify comma-decimal and unit normalization explicitly: Freya's current digit expansion
  handles integer/dot runs but does not define Race Engineer's Turkish `2,4 saniye` form;
- pin the canonical voice seed and treat failed/collapsed synthesis as an error or Piper
  fallback instead of silently switching to a different speaker seed;
- measure cold start, synthesis and release-to-first-audio p50/p95, real-time factor, peak
  RAM/VRAM, cancellation, and combined load with iRacing, ASR, and conversation inference;
- test Windows CPU and NVIDIA CUDA locally. Upstream publishes no Windows AMD GPU path, so
  retain CPU fallback and make no AMD acceleration claim without a separately validated
  runtime; and
- require fully offline startup after explicit setup, bounded worker failure, no hidden
  downloads, and no regression to radio priority, expiry, or freshness checks.

Upstream references: [model card](https://huggingface.co/freyavoice/Freya-TTS),
[technical report](https://arxiv.org/abs/2607.09530), and
[evaluation set](https://huggingface.co/datasets/freyavoice/freya-tr-eval).

## INT-001: Unsupported pit projection requests irrelevant telemetry

**Status:** Fixed in English/Turkish synthetic real-model replay - live validation pending

**Observed:** 2026-09-29, synthetic intelligence replay

"Where would I be if I pit this lap?" asked the driver to confirm current position and field
size even though the replay already contained those values and the application had no pit-loss,
stop-time, traffic, or rejoin projection capability. INT-04 now requires the Context Engineer to
classify the request as `future_counterfactual`. The INT-07 planner now rejects a draft that
requests current raw facts under that scope rather than silently pruning it. Accepted plans
supply an explicit unavailable projection. The Core Engineer states that limitation
rather than requesting unrelated input. INT-06 now exposes a reusable typed rejoin capability,
but it returns `missing_rejoin_projection_model` until pit loss, service duration, and field
trajectory inputs exist. Current telemetry alone is never presented as a strategy calculation.

## INT-002: Driver telemetry claim can fail structured generation

**Status:** Fixed in English/Turkish synthetic real-model replay - live validation pending

**Observed:** 2026-09-29, synthetic intelligence replay

"I'm P6 in 10 cars grid" caused a reachable local server to return HTTP 500. The likely failure
path was an attempted literal-number response conflicting with the numeric grounding grammar.
INT-04 now directs the planner to retrieve current position evidence, redacts unverified numeric
tokens before generation, supplies canonical rendering bindings, and gives the Core Engineer a
deterministic classification relationship. The reply must refresh every input to that relationship.
Replay diagnostics no longer advise starting Qwen for an HTTP response from a server that is
already reachable. Broader paraphrase coverage and live stability still require validation.

## INT-003: Generic scalar telemetry can fail structured generation

**Status:** Fixed in English synthetic real-model replay - broader validation pending

**Observed:** 2026-09-30, synthetic intelligence replay

Three consecutive requests for the fuel situation reached the Core Engineer and failed with
`engineer_model_http_500`. Context selection was correct: it returned current fuel plus an
explicitly unavailable fuel-trend item. The Core prompt had a canonical numeric-placeholder
example for classification but no equivalent for an arbitrary scalar, so constrained generation
could dead-end while attempting a literal value.

INT-04 now derives a structural rendering example from the selected evidence binding and its
metric/unit metadata. This is not a fuel phrase handler: the same path was validated with fuel,
speed and lap evidence, and the application still performs the final numeric substitution.

## INT-004: Context planner completeness, relevance and social-purpose errors

**Status:** Open - measured on local Qwen development replay, 2026-10-06

The repaired v4 planner passed 24/30 bilingual development cases: mixed-purpose turns lost
social intent, compound fuel/speed turns omitted speed, and historical speed averages selected
the wrong operation or capability. V5 adds a bounded request inventory and separate social flag
within the same inference. The latest saved run passes 41/50 (12/12 original, 15/18 additional
controls, 14/20 composition). The historical statistic controls now pass; both original
fuel/speed questions retrieve both facts. Remaining failures:

- Five compound turns add unrelated or redundant evidence despite retrieving requested facts.
- English venting is mistaken for a fuel question: a regression from v4.
- One English hypothetical question is incorrectly marked mixed rather than factual.
- English/Turkish gratitude-plus-speed questions lose their social purpose.

These drafts are schema-valid, so schema enforcement alone cannot detect semantic mistakes.
Do not add phrase-specific answer branches or silently reinterpret the requested statistic.
Keep these controls in development, expand disjoint paraphrases and actual sequential dialogue,
and re-evaluate relevance and completeness before a model-promotion decision. The planner
currently also permits only one factual temporal scope per turn; cross-scope compound requests
need an explicit contract extension. See [INT-07 evaluation](intelligence-evaluation.md).

V5 is currently wired into this development branch's live/replay path; there is no automatic
v4 fallback or release-quality claim. The request inventory is transient model guidance,
not authoritative evidence or proof of completeness. Planner median rose to 4.06 s in the
broader 50-turn run (earlier v4: 2.21 s over 30 turns); full radio latency still needs measuring.

## INT-005: Silent Core response uses an invalid empty speech string

**Status:** Fixed in synthetic local-model replay - live validation pending

**Observed:** 2026-10-06, full text pipeline smoke test after the planner repair

The Context Engineer correctly selected an evidence-free social turn, but the Core model
twice emitted `goal="silence"` with `speech_template=""`. Application validation requires
JSON `null` for silence, so the otherwise legitimate decision failed closed with
`engineer_model_response_invalid`. The schema now separates silent and spoken branches:
silence carries null speech, no guidance and no references; speech is nonempty. Python
validation remains in place for providers that bypass decoding constraints.

A Turkish social replay also exposed llama.cpp's structured-output parser error. Compact
bilingual evidence-free output examples restored valid replies in the tested cases, without
another model call or weaker validation. The final repeat gives short acknowledgments in
both languages; silence remains a valid explicitly tested option. These are structural fixes
and small synthetic checks, not proof of ideal teammate behavior or broad language quality.

## INT-006: Unrelated retrieved comparisons reject an otherwise grounded reply

**Status:** Dependency-check fix implemented - real-race validation pending

**Observed:** 2026-10-07, synthetic English fuel-range-plus-speed replay

The planner selected the requested fuel/speed plus unrequested classification. Core generated
a correct two-fact template with valid references, but application validation required all
retrieved relationship inputs, including unused classification. Both attempts were rejected
as `engineer_model_response_invalid`. The initial failed replay took 56.56 seconds.

Dependency closure is now checked per used relationship. A reply using unrelated fuel/speed
may omit classification; a reply referencing part of classification must still cite all its
comparison inputs. Evidence-free comparison drafts remain rejected. Deterministic numeric
substitution and delivery-time refresh remain unchanged. Five regression tests cover unused,
partially used, evidence-free and multiple independent relationships. This fixes the validation
coupling, not the planner's extra-evidence bug (INT-004).

The post-fix real-Qwen repeat spoke both correct facts, while a ranking control still cited
both comparison inputs. The compound reply took 43.08 seconds on CPU, so latency acceptance
is explicitly pending; this correctness repair is not a race-readiness claim.

## INT-007: Turkish compound reply intermittently fails Core validation

**Status:** Open - observed in the full text latency benchmark, 2026-10-07

On the second `compound-tr` attempt, Context selected the exact expected fuel-range-plus-speed
plan, but both Core output attempts failed application validation. Both model HTTP requests
completed; the final error was `engineer_model_response_invalid`. No reply was delivered, and
the failed turn took 21.90 seconds. The first attempt completed with both facts in its reply.

The timing report deliberately excludes raw model output and therefore does not establish the
specific validation defect. Investigate the two synthetic Core drafts before changing schema
constraints or attribution; this is not automatically the repaired unrelated-comparison bug.
The [latency baseline](evaluations/intelligence-latency-cpu-baseline-2026-10-07.json) retains
per-attempt numeric metrics and the failing stage. Keep this as a correctness failure when
comparing faster candidates or runtime settings, rather than dropping it from the workload.

## CONV-004: Add contextual race-engineer acknowledgments and reassurance

**Status:** Implemented in the new intelligence path - live validation pending

**Requested:** 2026-09-21

Conversation should respond naturally to remarks as well as factual questions. For
example, "That's dirty" could receive a concise acknowledgment or "Focus on your race
now", rather than being forced into a position/fuel query or a generic unsupported reply.
The requested feel is a supportive race engineer, not simply a spoken telemetry lookup.

The INT-04 portable Core/Qwen adapter now supports evidence-free acknowledgment, coaching,
clarification or silence with the selected calm-teammate style. Driver claims remain separate
from telemetry evidence, and the response still passes through the radio scheduler. Remaining
validation work:

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

STT-001 and TTS-001 remain deferred quality items. CONV-004 now changes the new live runtime,
but needs real-race English/Turkish validation before it can be considered accepted.

## CONV-001: Vague fuel follow-up switches to position

**Status:** Legacy-prototype defect - re-evaluate on the new live path

**Observed:** 2026-09-20, local Qwen3-4B-Instruct-2507 Q4_K_M, CPU

After "How is fuel looking?", "Where are we now?" returns current position rather than
staying with fuel or clarifying the topic. The position value is grounded, but the intended
topic is lost. Reproduced by `fuel-context` in `fixtures/conversation/cases.json`.

## CONV-002: Ambiguous opponent reference can be guessed

**Status:** Legacy-prototype defect - re-evaluate on the new live path

**Observed:** 2026-09-20, same local-model setup

With no preceding opponent context, "Is he pulling away?" can select the car ahead instead
of asking which car. Gap trends are unsupported, so it does not invent a trend, but the
reference resolution is wrong. Reproduced by `ambiguous-reference` in the model evaluation.

## CONV-003: Unsupported part of a compound question can be omitted

**Status:** Legacy-prototype defect - re-evaluate on the new live path

**Observed:** 2026-09-20, additional paraphrase evaluation

"Tell me the overall position and recommended tyre pressures" returns position but omits
the explicit acknowledgement that tire advice is unavailable. It does not invent tire
pressures. Reproduced by `partially-supported-compound` in
`fixtures/conversation/holdout.json`.

CONV-001 through CONV-003 were reproduced with typed input, independently of transcription.
Keep the failing evaluation cases rather than weakening their expected behaviour.

## IR-001: False blue-flag call immediately after race start

**Status:** Fixed in normalization - live validation pending

**Area:** iRacing telemetry normalization and flag-event derivation

**Observed:** 2026-09-19, session `iracing:1:0`, source sequence `3732`

During the M3 live speech validation, the system announced "Blue flag" roughly 2.7 seconds
after the green-flag call. The driver confirmed that no blue-flag condition existed. The
policy, language, and TTS stages behaved correctly for the event they received. Inspection
of the recorded state showed the player starting lap 1, becoming P1, and no active opponent
with a greater completed-lap count. The defect was the normalizer accepting the raw blue
bit without establishing that it could apply to the player.

Race-session normalization now requires a valid player classification and an active,
non-pace-car opponent with a greater completed-lap count before exposing blue. The raw bit
continues to determine whether iRacing considers traffic close enough; the additional check
only corroborates that lapping traffic exists. Practice and qualifying preserve iRacing's
blue signal once player position is valid because completed-lap comparisons do not carry
the same meaning there.

The reduced regression fixture at
`fixtures/iracing/ir001_false_blue_start/raw_samples.jsonl` proves both that the captured
start transition produces no event and that a lapping-opponent control does produce one.
A live race with a genuine blue flag is still required before closing the issue completely.
No policy cooldown or timing workaround was added.
