# Known issues and deferred improvements

## STT-001: Short and single-word questions are sometimes misrecognized

**Status:** Chatterbox migration implemented; driver listening/game-load validation pending

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

The earlier voice felt unnatural/off, particularly in Turkish. On 2026-10-25 the
user selected local Chatterbox Multilingual 0.1.7's built-in voice for all speech.
Both old engines and per-language voice settings were removed. See
[speech setup and limitations](text-to-speech.md). The implementation is not proof
that naturalness, Turkish pronunciation or racing latency is accepted.

The September Freya notes below are historical research, not current routing or
fallback instructions. No Piper fallback remains after the Chatterbox migration.

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

## TTS-002: Chatterbox can append speech-like gibberish after the supplied sentence

**Status:** Reproduced in saved authored audio; generation fix pending

**Observed:** 2026-10-29, live v8/FP16 session. The driver heard extra gibberish
after `I don't have the historical data needed for that.` The logged TTS input
contains only that sentence; actual live audio is not retained. Another nine-word
fallback produced 12.12 seconds of audio during the same session.

An isolated local probe used the unchanged production default-voice synthesis
function and installed Chatterbox 0.1.7, with paired seeds in BF16 and FP32. It
saved eight WAVs per precision without playback or live configuration changes.
Local ASR flagged an added tail after the exact historical-data sentence in
BF16 seed 1 (`You as Tang said.`), and similar additions in FP32 samples. These
are ASR-assisted observations, not authoritative transcriptions of gibberish.
For the other fallback, BF16 seed 1 generated 11.68 seconds and a long unrelated
ASR transcript. A stop token was present in every saved sample, so the presence
of EOS alone does not establish clean speech or rule out a bad tail.

Artifacts: ignored `data/diagnostics/chatterbox-tail-probe-20261029/`, containing
the probe scripts, per-precision `report.json`, WAVs, native logs and
`transcripts.json`. The original race's random state was not recorded, so this is
a reproduction of the failure pattern, not a reconstruction of that exact audio.
The small sample does not establish error rates or isolate precision as the cause.

Follow-up: compare supported decoding/alignment controls against these saved
failures and EN/TR controls, then add a measured bounded rejection/retry policy if
needed. Do not treat output duration, finite PCM or an EOS token as proof that the
audio contains only the requested words. No live model, precision or voice change
has been promoted from this probe.

November 7 follow-up: a V3 source race candidate is installed separately and selected
explicitly by `data/trials/chatterbox-v3-live.toml`. Both EN/TR reply and automatic
playback paths completed smoke checks through one CUDA/BF16 worker, including the
previous failure sentences. This verifies initialization/routing/playback, not the
absence of unwanted words. Keep TTS-002 open pending listening and race acceptance.
See [the V3 trial](text-to-speech.md#v3-source-race-trial-2026-11-07) for setup,
runtime/model pins, timing/memory observations and rollback.

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

V5 was wired into this development branch's live/replay path before the October 7 v6 switch;
there is no automatic v4 fallback or release-quality claim. The request inventory is transient
model guidance,
not authoritative evidence or proof of completeness. Planner median rose to 4.06 s in the
broader 50-turn run (earlier v4: 2.21 s over 30 turns); full radio latency still needs measuring.

The `qwen-context-v6` planner now covers the English vent, hypothetical pit,
and compound-selection regressions with fresh bilingual controls. On the same 58-turn set it
scored 53/58 exact plans versus v5 at 45/58; the eight new contrast cases passed, and extra
evidence turns fell from nine to two. The 30-turn full-pipeline sample improved exact plans
from 18/30 to 30/30, but median total latency increased from 8.01 s to 10.26 s. V6 is now the
live/replay development default by user decision; v5 remains available for explicit comparison.
Five older follow-up/composition failures remain, and full spoken quality and release promotion
still need broader evaluation. See the candidate comparison in
[INT-07 evaluation](intelligence-evaluation.md).

**Live follow-up and current priority, 2026-10-12:**

The October 11 GPU sessions showed that faster inference does not resolve request
understanding. A greeting was mapped to lap telemetry; factual position requests
produced `planner_empty_selection`; damage questions were treated as social content
or substituted with speed. Direct typed probes on October 12 reproduced greeting/
presence checks as classification, a one-word position request as a gap request,
an unsupported starting-grid request as an empty plan, and damage as social content.
These probes used synthetic telemetry and no STT, microphone or TTS, so this is not
solely a speech-recognition problem. They do not recreate the live telemetry catalog
or establish a general accuracy percentage.

The unchanged v6 planner was also rerun against the four existing bilingual
development datasets on a fresh owned CUDA server: 53/58 exact turns passed
(English 27/29, Turkish 26/29), with zero transport/model/validation errors.
This is consistent with the earlier development result, but those datasets do
not adequately represent the everyday short greeting/presence/unsupported cases
above. Planner-only median was 285 ms and sample p95 534 ms on this idle host;
these are not microphone-to-audio or game-load measurements. No prompt, model,
decoder contract or live routing was changed during these checks.

The next slice is Qwen interpretation correctness: expand bilingual development
coverage for short natural utterances, social/presence checks, available and
unavailable facts, and recent-dialogue references; compare a separately evaluated
prompt/output-contract candidate with v6 before live adoption. Do not introduce
phrase-specific routing, weaken evidence checks, silently substitute current rank
for starting grid, or infer that a fast/schema-valid plan is semantically correct.
Empty plans need bounded recovery or an honest failure response, not an invented
telemetry answer. Race-start briefings remain deferred under RACE-001.

October 12 implementation: `qwen-context-v7` was initially an opt-in candidate,
not the live default. It uses one social-kind decision plus a factual request inventory in the
existing Qwen planner call, without transcript keyword routing or another model.
Application validation still owns catalog IDs, query/scope legality, evidence and math.
Forty-four bilingual development turns now cover greetings, radio checks, short
classification questions, unsupported damage/grid facts, compounds and dialogue.
Twelve disjoint post-freeze paraphrases are also available for development validation.

On the same 102-turn CUDA development workload, common-contract exact credit rose
from v6's 67/102 to v7's 95/102. This excludes social-subtype checks for a fair
comparison: v6 has no such output field. V7 passed 21/22 subtype-bearing turns.
Existing 58-turn coverage improved from 53/58 to 57/58; new coverage improved from
14/44 to 38/44. The seven remaining failures are an invalid English three-fact
draft, Turkish relief misread as facts, one Turkish original-grid paraphrase mapped
to live classification, incomplete/extra retrieval on two damage-plus-speed turns,
and two unresolved references guessed as a gap behind. These are not fixed.
Bounded planner repair and an audible failure path remain follow-ups, not implemented
by this candidate; unsupported facts that are correctly interpreted now carry a
stable unknown instead of an empty factual plan.

Twenty selected candidate text-pipeline checks completed with matching plans and
evidence. The final separately saved 12-paraphrase CUDA replay also passed 12/12
with no Core repairs: median 0.609 s, sample p95 0.877 s, maximum 1.066 s. These
exclude microphone/STT/TTS, server startup and racing load. Manual review found
honest grid/damage limitations but awkward Turkish phrasing (including the wrong
speaker perspective in a mixed radio check). Qualitative factual truth is not
mechanically proven by numeric grounding. On October 25, the user selected v7 as
the development/live-test default because it outperformed v6 in the matched tests.
V6 remains an explicit rollback. This rollout decision does not fix the seven
misses or establish real-catalog/game-load acceptance. Details
and repeat commands are in [INT-07 evaluation](intelligence-evaluation.md#october-12-understanding-candidate-v7).

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

**Status:** Fixed in synthetic local-model replay - live validation pending

On the second `compound-tr` attempt, Context selected the exact expected fuel-range-plus-speed
plan, but both Core output attempts failed application validation. Both model HTTP requests
completed; the final error was `engineer_model_response_invalid`. No reply was delivered, and
the failed turn took 21.90 seconds. The first attempt completed with both facts in its reply.

The original timing report deliberately excluded raw model output and could not establish the
specific defect. The compact-Core performance slice reproduced it under manual reply review:
Qwen selected the correct short evidence aliases but sometimes wrote `[a]` and `[b]` instead of
the required `{{a}}` and `{{b}}` placeholders in Turkish. A strict repeat rejected the same
delimiter error on both the initial and repair drafts. This was a wire-format serialization
failure, not a telemetry calculation or evidence-selection failure.

The Core wire response now contains only `goal` and `speech`. Exact known square-bracket aliases
are canonicalized to the corresponding trusted placeholders before references are reconstructed.
The application still rejects invented aliases, malformed/repeated placeholders, literal numeric
telemetry and incomplete deterministic relationships. Sanitized rejection reasons are logged
without model or driver text. The final 12-turn CPU replay completed 12/12 with zero repairs;
both Turkish compound replies cited and grounded fuel range and speed correctly. The
[post-change report](evaluations/intelligence-latency-core-compact-cpu-2026-10-07.json) retains
the per-stage evidence. Real-race validation remains pending.

## INT-008: Live engineer stops after five completed conversations

**Status:** Fixed in simulated live-loop regression - real-race validation pending

**Observed:** 2026-10-07, local panel shakedown

The failed run completed five replies before `panel_error` reported `ValidationError`.
The live buffer retained six complete exchanges (twelve dialogue entries) by default, while
`DriverTurn.recent_dialogue` accepted only eight entries. The sixth question therefore tried
to construct a turn containing ten entries and stopped the worker. Restart cleared the buffer.
The log omitted validation-field detail, but a simulated live-loop regression reproduced this
exact failure before the fix.

Configuration and dialogue contracts now share a maximum of twelve complete exchanges
(twenty-four entries). The default remains six exchanges, and live/replay buffers derive
their entry count from the validated configuration. Evaluation history uses the same bound.
Regressions cover zero, default and maximum history, continuing beyond buffer rollover while
retaining the most recent complete exchanges. Oversized contract history remains rejected.

## INT-009: Q8 KV cache can confuse Turkish starting-grid and current-position requests

**Status:** Open — blocks Q8 cache promotion; live FP16 cache is unchanged

**Observed:** 2026-10-28, offline matched Qwen v7 cache comparison on RTX 5080

The full development comparison with the same Q4_K_M weights, 8192 context,
full CUDA offload and verified Flash Attention found a Q8-only regression for
`Yarışa kaçıncı başladık?` (`understand-starting-grid-tr`, question ID
`de1d737d172e2788`). FP16 correctly treated the starting-grid information as
historical/unavailable. Q8 selected `current_classification` and described the
current P5 as the starting position. Do not count this as a wording-only change.

Local artifacts: `data/benchmarks/qwen-cache-20261028T133127Z/report.json` and its
per-cache `review.jsonl` files. The targeted follow-up
`data/benchmarks/qwen-cache-20261028T133459Z/report.json` reproduced the same
question's failure under Q8 in both cache orders (FP16 combined checks 8/8, Q8 6/8).
Reproduce the complete matched sequence with:

```powershell
.\.venv\Scripts\python.exe scripts\benchmark_qwen_cache.py --full-suite --rounds 1
```

A short representative subset did not catch this; preserve full paraphrase and
seeded-history checks. Source/input fingerprints and effective cache/attention
settings were verified. The observation is not proof of universal Q8 failure or
of deterministic reproduction under a different prefix/order. Investigate temporal
selection and reply grounding before enabling Q8; do not shrink context or alter
evaluation goldens to make a performance comparison pass. See
[cache trial details](qwen-cache-trial.md). Existing shared baseline failures and
combined racing-load acceptance remain separate work.

V8 follow-up (2026-10-28): an experimental time-first prompt plus opt-in temporal
evidence metadata passed the targeted starting-grid cases under both caches and
both orders. The expanded 84-question matrix also exposed the old error under
v7/FP16, so this is a latent interpretation weakness, not exclusively Q8 behavior.
V8 is not promoted: remaining exact-plan and manual reply failures are recorded in
[the v8 comparison](qwen-cache-trial.md#v8-results-2026-10-28-not-promoted).

## INT-010: Core can turn unavailable damage information into an unsupported negative claim

**Status:** Observed negative claim mitigated in experimental v8; wider semantic validation open

**Observed:** 2026-10-28, authored offline v8 FP16/Q8 comparison

For `Do I have any damage, and what is my speed?`, v8/FP16 selected supported
speed plus an unavailable-analysis reason and passed the plan/evidence checks,
but replied with the speed followed by `No damage has been reported in the telemetry.`
V8/Q8 similarly said `No damage has been reported at this time.` Missing damage
telemetry does not justify that assertion. This is distinct from a parser failure
or merely omitting the unsupported half of a compound request.

Artifacts: `data/benchmarks/qwen-cache-20261028T142016Z/report.json` and its two
`round-01-*/review.jsonl` files, case `understand-partial-damage-speed-en`.
The FP16 run is a concrete example of automatic plan/evidence passes not proving
radio-response correctness. The new temporal guard checks referenced evidence
timing; it does not semantically validate every free-form clause. Retain explicit
unknowns and investigate unsupported-claim handling without phrase blacklists,
inventing damage readings, or weakening evaluation goldens. Live v7/FP16 remains
unchanged; no new damage-analysis capability is implemented by this experiment.

October 28 follow-up: v8's `grounded-limitations.v1` now binds application-owned
EN/TR limitations to explicit unknown IDs, preserves supported answers in partial
requests, and refreshes unavailable inputs before grounding. Fully unavailable
requests skip Core inference. Unbound punctuation clauses are removed from partial
Core bodies; an omitted supported answer has one bounded repair. The 92-turn
FP16/Q8 run `qwen-cache-20261028T153414Z` reproduced the original question with
an explicit limitation and measured speed under both caches. Damage/tyre-only
and damage-plus-fuel replies were also reviewed without a healthy/no-damage claim.
The structural guard cannot prove assertions sharing a clause with a valid
placeholder or repair an incorrect planner interpretation. Keep this limitation
open for wider semantic/live acceptance; v7/FP16 is still the live default. See
[the safeguard and results](qwen-cache-trial.md#experimental-unavailable-data-handling).

## INT-011: Turkish speed wording can lose or mix the evidence unit

**Status:** Open — manual response-quality blocker for v8 promotion

**Observed:** 2026-10-28, authored offline v8 FP16/Q8 replay

For `Hasar var mı ve hızımız kaç?`, guarded v8/FP16 correctly states the missing
data limitation but describes a 51 m/s speed value as `51 metredir`, omitting the
time unit. `Anlık hızımızı ve lastik aşınmasını söyler misin?` produces
`51 metre per second` under both caches. The numeric value is grounded; free-form
unit wording is not. Do not treat numeric substitution or a limitation guard as
proof that a sentence preserves the evidence's meaning.

Artifacts: `data/benchmarks/qwen-cache-20261028T153414Z/round-01-*/review.jsonl`,
cases `understand-partial-damage-speed-tr` and `unsupported-tyres-speed-tr`.
Follow-up: validate metric/unit meaning and bilingual realization without changing
the underlying value or introducing a fixed driver command set. Live defaults
were not changed by this experiment.

## INT-012: Prepared replies expire silently while the panel says Speaking

**Status:** Delivery/UI fix implemented; live race retest pending

**Observed:** 2026-10-28, v8/FP16 run `3ea38d8308604d4093379454efecdae8`,
18:56–18:58 Istanbul. Six transcribed driver turns produced one spoken reply,
four `tts_reply_expired` outcomes and one Core validation failure. Failed TTS
stages lasted 3.1–6.3 seconds. One automatic announcement also completed.

The callback compared the pre-synthesis frame's timestamp with the three-second
telemetry freshness threshold, even as new telemetry arrived. The scheduler also
declared Speaking before synthesis, explaining the misleading panel status.

October 29 fix: conversational queueing/preparation uses the separate 15-second
`reply_delivery_timeout_s` budget. Playback still requires fresh latest telemetry
and the original session generation; automatic intent deadlines are unchanged.
The worker receives the remaining preparation budget and acknowledges cancellation
before reuse. The panel distinguishes preparation from playback dispatch and
shows unplayed-answer notices. Tests cover six-second generation with a fresh
stream, stale/disconnected telemetry, session changes, timeout, mute, failure and
interruption. Spoken values still describe the snapshot retrieved before synthesis;
they can age within this delivery budget. See [the delivery boundary](live-conversation.md#freshness-and-session-isolation).

## INT-013: Unheard replies and redacted history values can contaminate later answers

**Status:** Delivery-aware history and marker guard implemented; broader semantics open

In the same run, the planner correctly categorized `Hey, yo, are you there?` and
`Hello.` as social, but Core returned position text containing `[unverified_number]`
with no referenced evidence. Generated replies entered history before playback,
including an earlier position response the driver never heard. Numeric redaction
markers could then be copied as ordinary non-numeric prose.

October 29 fix: live intelligence commits a reply only after completed playback
(or explicit text-only CLI delivery); failed, muted, expired, interrupted and silent
answers are not stored as spoken exchanges. Session generations clear this memory.
Core rejects the internal marker and has one bounded repair. V8 radio checks and
greetings omit old dialogue from Core's payload, while the Context planner keeps
history for interpretation; a social-only reminder prohibits historical factual
claims. EN/TR radio checks and greetings after an earlier position answer were
manually reviewed under both caches without position contamination. This is not
semantic proof of every evidence-free sentence.

## INT-014: Last-lap timing requests can select classification or an unrelated query

**Status:** V8 interpretation improved; timing capability absent and Turkish edge case open

The live run interpreted `What is my last lap?` as current classification. No
completed-lap duration exists in the current normalized telemetry catalog. Correct
behavior is an honest unavailable-data reply, not a position answer or fabricated
lap time. Adding real lap timing remains separate telemetry/capability work.

October 29: v8 distinguishes completed-lap duration, lap count and rank without a
spoken-command router. Both English replay paraphrases now state the historical
data limitation under FP16 and Q8. One Turkish paraphrase also does so, but
`Bir önceki turumun süresi neydi?` fails FP16 query validation; Q8 instead retrieves
unrequested speed data and mentions it. Keep these failures open. Artifacts:
`data/benchmarks/qwen-cache-20261028T234957Z/round-01-*/review.jsonl` and `report.json`.

## INT-015: Experimental LFM native selector can ramble instead of requesting a tool

**Status:** Open - native LFM is now the selected English MVP path

**Reported and reproduced:** 2026-11-08, typed English synthetic telemetry trial

After the sequence `where are we`, `position`, `pit strategy`, `damage report`,
and `gap ahead`, two consecutive `gap behind` turns failed with
`context_model_response_incomplete`. The model spent its full 768-token selection
budget generating conversational prose, including repeated thinking tags, and
returned no tool call. No telemetry calculation or Core reply was executed for
those failed turns. The actual `gap_behind` capability was available and returned
**1.6 seconds** at synthetic sequence 203. The model's discarded prose invented
2 seconds; do not salvage it as a fallback or increase the budget to hide the loop.

An isolated comparison explicitly sent `reasoning_effort="none"` plus
`chat_template_kwargs.enable_thinking=false` on every request. It reproduced both
failures; this did not establish a fix. No application settings or token budgets
were changed by the diagnostic. The user's running server remained untouched;
the separate probe server on port 8093 was closed afterwards.

The same sequence exposed two separate relevance problems: a broad race overview
selected only current classification, and `are we catching` selected the future
catch-time projection instead of the existing current relative-pace capability.
The fixture supports `closing` at 0.05 s/s; 20 seconds is a conditional catch-time
estimate if that trend continues. A valid projected number is not the same as
answering the requested current relationship. Presentation also exposes internal
labels such as `field classified_cars` and `future counterfactual`.

Next acceptance checks: fresh and history-bearing gap requests must finish with
validated calls; current closing/losing-ground questions must use relative pace;
future catch-time questions must retain their assumption; broad overviews must
retrieve relevant available facts or explain their limits. Preserve natural
phrasing, bounded execution and evidence validation rather than adding a phrase
lookup or accepting unvalidated model prose.

**Wording update, 2026-11-08:** The native Core now receives a compact fact view
with human descriptions, preserving original evidence references for validation
and refresh. The reported gap/damage/availability exchange produced short,
distinct replies, and position answers stopped reading the internal reported-car
count in the inspected examples. Limitation framing is model-selected from safe
choices; unsupported claims remain application-bounded. The unchanged 26/50
interpretation score, looping gap selector and broad overview/catch-time selection
issues remain open. This is a presentation improvement, not resolution of INT-015.

**Ranking/history update, 2026-11-08:** Native selection now receives quoted
delivered history rather than prior assistant replies. Core keeps the current
question verbatim and excludes old spoken measurements. Separate first/last
operations use application-calculated claims, refreshed alongside the supporting
position. This fixes the reported repeated `are we last` / `are we first`
redaction-marker failures. The exact sequence and eight current classification
checks passed against the real local model; gap/damage follow-ups also completed.
A bare gap now retains its direction/unit from evidence metadata. No driver
phrase router or unvalidated free-text fallback was added.

The latest unchanged 50-case development run completed all turns, with 29 exact
plans, 35 evidence matches and zero Core repairs (322/473 ms typed median/p95).
The remaining 21 plan mismatches, variable follow-up wording and earlier
loop/scope/relevance failures keep this issue open. No live race, microphone,
audio playback, Turkish or VR-load acceptance is implied. Final ignored artifacts:
`data/benchmarks/lfm3-ranking-20261108-r3/` and
`data/diagnostics/lfm3-ranking-20261108/`.

Synthetic diagnostic artifacts remain ignored under
`data/diagnostics/lfm3-native-adapter-20261108/`: `gap-turns-wire.jsonl`,
`gap-turns.log`, `gap-turns-no-reasoning-wire.jsonl`, and
`gap-turns-no-reasoning.log`. See [the LFM trial](lfm3-trial.md).

## RACE-001: Announce starting grid and an expected finishing position

**Status:** Deferred feature - after Qwen request-understanding work

**Requested:** 2026-10-12

At the start of a race, the engineer should brief the driver on their actual
starting-grid position and an evidence-backed expected finishing position or range.
Starting grid is a distinct, session-scoped fact, not a synonym for current live
classification. The finishing estimate requires a separately validated prediction
method, suitable inputs and explicit uncertainty; do not derive or invent it merely
from current position. Missing inputs must produce an honest limitation.

Later implementation should define the briefing window, retain the authoritative
grid snapshot, prevent duplicate briefings on reconnect, and respect critical-radio
priority. No grid capability, prediction model or automatic briefing is enabled by
this backlog entry. Improve Qwen's understanding first (INT-004).

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
