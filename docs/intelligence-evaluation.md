# INT-07 Context Engineer evaluation

INT-07 compares local Context Engineer candidates at one stable boundary: given a driver
turn, a value-free signal catalog and the deterministic race-capability catalog, produce a
typed `ContextPlan`. The application—not the candidate—executes telemetry queries and race
arithmetic. This makes a Qwen planner, a small semantic model and a later temporal model
comparable without allowing any of them to invent race values.

## What is implemented

- Versioned development, calibration and locked seed datasets under
  `fixtures/intelligence-evaluation/`.
- English and Turkish paraphrases covering every INT-06 capability, generic raw-signal
  retrieval and evidence-free social turns.
- Exact scoring of temporal scope, capability selection, normalized telemetry queries,
  evidence availability and required stable unknown reasons.
- Additional required-selection coverage and extra-request counts, without relaxing exact
  scoring. Rejected plans retain catalog-only diagnostic selections and rejection counts;
  they never execute or earn accepted-plan credit.
- Separate first/subsequent-call planner, deterministic execution and total latency
  distributions. Call order does not establish model startup or prompt-cache state.
- Optional numeric-only server prompt/generation timings and cached-token counts; absent
  provider metrics remain null rather than being inferred from wall-clock latency.
- Explicit purpose scoring and 18 additional bilingual development controls for social,
  mixed, compound, historical raw-query, unavailable, authored-follow-up and changed-catalog cases.
- Twenty bilingual composition controls: two/three requested facts, reordered clauses,
  contrastive average/peak/minimum/net-change requests and social/factual combinations.
- Content-free reports: test utterances are represented by hashes; model replies and live
  driver transcripts are never written. The authored fixture questions remain in the
  version-controlled dataset itself.
- Reproducibility metadata for candidate/model revision, input and implementation fingerprints,
  Git state and local machine hardware.

The original 32-turn workload plus 38 development controls exercise the contract and harness
(50 development turns; the calibration and locked seeds remain unchanged).
They are not large enough to justify model promotion. A promotion-grade locked set needs at least 100
English and 100 Turkish turns, mixed/multi-turn cases and adversarial scope controls before
its scores become a release gate.

## Candidate profiles

| Profile | Current state | Intended use |
| --- | --- | --- |
| `portable` | Runnable with `qwen-context-v5` | One local general model selects catalog-bound evidence; deterministic code executes it |
| `enhanced` | Reserved, fails closed | A learned temporal component may add typed signals/capabilities before the same planning boundary |

MiniLM and Laya remain evaluation candidates, not dependencies or selected architecture.
Each needs a pinned local checkpoint, license review and an adapter that emits the same
complete `ContextPlan`. A classifier that can only return one intent is not equivalent to the
planner: it must support compound questions, abstention, temporal scope and capability/query
selection or be evaluated only as one component of an enhanced candidate. A learned temporal
model likewise exposes provenance-bearing typed evidence; it never sends raw model-authored
numbers directly to speech.

## Run the portable baseline

From the repository root, start the configured local llama.cpp/Qwen server in terminal 1:

```powershell
.\.venv\Scripts\python.exe scripts\start_conversation_model.py --config config\default.toml
```

Wait for loading to complete and keep that terminal running. In terminal 2, run:

```powershell
.\.venv\Scripts\python.exe -m race_engineer evaluate-intelligence `
  --config config\default.toml `
  --dataset fixtures\intelligence-evaluation\development.json `
  --dataset fixtures\intelligence-evaluation\development-controls.json `
  --dataset fixtures\intelligence-evaluation\development-composition.json
```

Use the development split while changing prompts or adapters and calibration only when
choosing thresholds. Run the locked split after the candidate and thresholds are frozen:

```powershell
.\.venv\Scripts\python.exe -m race_engineer evaluate-intelligence `
  --config config\default.toml `
  --dataset fixtures\intelligence-evaluation\locked.json
```

The pinned small-model screening candidates use the existing local STT Python environment
as an isolated PyTorch worker and do not change the live application dependencies:

```powershell
.\.venv\Scripts\python.exe -m race_engineer evaluate-intelligence `
  --candidate minilm-nli-v1 `
  --dataset fixtures\intelligence-evaluation\development.json

.\.venv\Scripts\python.exe -m race_engineer evaluate-intelligence `
  --candidate laya-multilingual-v1 `
  --dataset fixtures\intelligence-evaluation\development.json
```

The command does not download or install anything. For Qwen, start the configured llama.cpp
server first; an unreachable server is counted as a per-turn model error so failures remain in
the denominator. MiniLM and Laya workers start and stop inside the explicit evaluation command
with offline model-loading flags. The report is printed as JSON; redirect it to a deliberately
chosen local file when a result should be retained.

## First portable baseline

The October 4 results below describe the historical `qwen-context-v2` contract. The current
`qwen-context-v5` adapter uses a different model-facing contract; old reports are preserved,
not relabelled or retroactively rescored.

The first CPU-only Qwen run is recorded in
[`evaluations/intelligence-qwen-cpu-seed1-2026-10-04.json`](evaluations/intelligence-qwen-cpu-seed1-2026-10-04.json).
On development plus calibration, 17 of 24 turns hit the 30-second timeout and none produced
an exact plan. The seven completed plans over-selected capabilities or duplicate raw queries.

The model-facing plan was then reduced on development data only: application code now assigns
query/request IDs, the payload uses a compact value-free catalog, and output is limited to 256
tokens. Mean planner time fell from 29.5 seconds to 11.8 seconds and no turn timed out, but the
candidate still passed 0 of 12 exact turns. It selected relevant capabilities more often while
continuing to add duplicate evidence, unrelated capabilities or invented signals. The compact
contract is retained as an efficiency improvement; Qwen has not earned promotion as the
Context Engineer.

Pinned MiniLM and Laya were then evaluated through the same local dynamic-catalog adapter; the
comparison is recorded in
[`evaluations/intelligence-small-candidates-seed1-2026-10-04.json`](evaluations/intelligence-small-candidates-seed1-2026-10-04.json).
The adapter scores value-free capability descriptions, normalized signals and a social option;
it contains no utterance phrase router. MiniLM passed 1/12 development turns with a 93 ms warm
p50. Laya passed 3/12 with a 1.20-second warm p50. Both passed 0/6 Turkish turns. Neither was run
on calibration or locked data because neither cleared development.

These results reject the naive zero-shot single-topic adapters, not the model families for every
possible fine-tuned/component role. MiniLM may still be useful as a trained retrieval component;
Laya may still be evaluated with calibrated domain questions. Neither can replace the Context
Engineer today, and neither changes live routing.

## October 6 repair: catalog-bound queries and diagnostic scoring

The first repair slice keeps the same model, one planner call, live pipeline and internal
`ContextPlan` contract. The model-facing query now contains `signal_id`, `operation` and
`window_s`. Each ID binds a complete advertised selector; the model no longer constructs a
source/signal/opponent tuple. Dynamic schema enums restrict direct/window/field operations to
their actual catalog IDs. Application validation still rejects forged IDs or invalid operations
if a provider bypasses the schema. Capability requests and raw queries can coexist for compound
questions. This is an evidence API, not a spoken-phrase router.

Reports now use `intelligence-eval-report.v2`:

- `failure_stage` distinguishes transport, response, plan validation and evidence execution.
  `model_errors` counts transport/response failures, not rejected evidence plans.
- `rejected_draft` retains only recognized catalog selections, a validated scope and rejection
  counts. Unrecognized model identifiers, free-text reasoning and transcripts are not copied.
  A rejected draft never becomes `observed_plan`, never executes and cannot pass.
- `selection` counts matched, missing and extra expected requests. Duplicates count as extras.
  `required_evidence_passed` requires all expected requests, correct scope, the expected evidence
  availability and unknown reasons, and successful execution; it tolerates extra requests.
  It is a diagnostic metric, not a promotion gate or proof of equivalent alternative reasoning.
  Exact plan/turn scoring and the promotion rules below remain unchanged.
- Unrecognized model-authored unknown text is hashed in reports and still fails comparison;
  a generic `unknown` on a social turn is not silently excused.
- `call_state` and `planner_first_call` / `planner_subsequent_calls` replace the misleading
  cold/warm labels. The external server may already have cached any of these prompts.

The catalog-bound v3 intermediate run is retained in
[`evaluations/intelligence-qwen-v3-development-2026-10-06.json`](evaluations/intelligence-qwen-v3-development-2026-10-06.json).
It eliminated invalid signal selections and retrieved required selections on all ten factual
questions, but still passed 0/12 exact turns: duplicate evidence and generic unknown markers
remained. Catalog binding alone did not solve understanding.

## October 6 repair: purpose, minimal evidence and prompt reuse

The v4 planner separates **purpose** (`social`, `race_information`, `mixed`) from factual time
scope. Pure social turns use no scope, evidence or unknown reason. A mixed turn retains both
the social remark and the factual request; multiple factual requests alone are not "mixed".
Application-owned situation tags carry that distinction into the existing Core/response step.
There is no extra inference layer or phrase-to-answer router.

The model now selects a small `missing_information` enum rather than writing free-form
unknowns. Matching unavailable capabilities supply their own authoritative reasons. Scoped
JSON alternatives prohibit social evidence and current facts masquerading as projections;
Python checks enforce the same rules even if a provider bypasses the schema. Balanced examples
teach the output structure and minimum required evidence, not an accepted vocabulary.

The stable, value-free catalogs precede language, recent dialogue and transcript in the prompt.
This permits llama.cpp prefix reuse while changed catalogs invalidate the affected prefix.
Neither telemetry values nor evidence plans are cached: each turn binds the current catalog,
and delivery-time evidence refresh remains unchanged. The same Qwen model and CPU runtime are
retained in configuration. Optional server metrics distinguish prompt processing from token generation;
first/subsequent call order is not evidence of a cold/warm server.

The supplemental `development-controls.json` uses fresh memory for each case, including an
early frame without fuel history and a later frame with enough history. Follow-up controls
provide **authored prior dialogue**, not actual earlier model replies. Purpose expectations
are scored only where annotated; the original seed's expectations were not rewritten.
History strings are passed to the candidate but omitted from reports. Both full exact scores
and required-evidence diagnostics require the annotated purpose to match.

The current contract still has one factual time scope per turn. Same-scope capability-plus-raw
questions are representable; simultaneous current and historical/future requests need a later
per-request-scope design. These controls are development data, not an independent release gate.

### Real local-Qwen development results

All October 6 repair runs used the existing local Qwen3-4B-Instruct-2507 Q4_K_M server and
the CPU-configured eight-thread profile on the Ryzen 7 9800X3D machine. Reports describe
configuration, not independent verification of an externally launched server's GPU offload.
No model weights, runtime settings, calibration questions or locked questions were changed.

| Development iteration | Original seed | New controls | Interpretation |
| --- | --- | --- | --- |
| v3 catalog binding | 0/12 | Not run | Valid evidence IDs alone were insufficient |
| v4 first balanced examples | 11/12 | 11/18 | Better factual/social separation; compound and follow-up gaps |
| v4 final contract examples | 12/12 | 12/18 | 24/30 exact overall; six substantive failures remain |

The first v4 runs are retained as
[seed report](evaluations/intelligence-qwen-v4-development-2026-10-06.json) and
[controls report](evaluations/intelligence-qwen-v4-controls-2026-10-06.json).
The [final combined development report](evaluations/intelligence-qwen-v4-development-final-2026-10-06.json)
records the current prompt's implementation fingerprint. Several intermediate prompt/schema
experiments were rejected; these are development iterations, not independent holdout results.

Final exact accuracy is 80% in each language. All 30 turns have correct temporal scope and
expected availability/unknown handling, with no transport, response, validation or execution
errors. This does **not** mean all facts were selected correctly: both mixed controls lose the
social-purpose tag, both fuel-plus-speed requests omit speed (Turkish also mislabels purpose),
English historical average speed selects maximum, and Turkish historical speed selects position
change. Full exact scores retain these failures. Purpose accuracy is 15/18 on annotated controls.

The combined run's planner median was 2.21 seconds, subsequent-call median 2.19 seconds and
first call 13.44 seconds. A changed-catalog call took 9.76 seconds; p95 overall was 6.93 seconds.
Server prompt/generation metrics and cached-token counts confirm reuse on repeated catalog
prefixes, but this sequential synthetic workload favors reuse. It excludes STT, the second
Core/response inference, TTS, racing load and model startup. Alternating live prompts may
reuse less; these numbers are not a radio-response latency promise.

A [repeat of the same 30 turns](evaluations/intelligence-qwen-v4-development-repeat-2026-10-06.json)
reproduced the same six failures and 24/30 score. With server prefix reuse already established,
planner median was 2.18 seconds and p95 3.06 seconds. Repetition measures this workload's
consistency and cache behavior, not generalization to unseen questions.

Qwen remains an **unpromoted development baseline**, despite the improvement over the original
0/12 result. Preserve the difficult controls and broaden the workload before judging generality.

### Integration checks after the planner runs

The full text pipeline exposed a separate Core decoding defect: a silent social decision
returned an empty speech string instead of JSON null, even after its existing retry. Silent
and spoken outputs now have separate schema branches; application validation is not relaxed.
A Turkish social sample also hit llama.cpp's structured-output parser error. Compact bilingual
social output examples restored valid replies in the checked samples without another inference
layer. These fixes are not proof of general conversational quality.

Final four-case synthetic text smoke results, with the same local server:

- Classification: correctly reported P5 and last classified position P6 (14.05 s).
- English frustration: short, neutral refocus acknowledgment (5.67 s).
- Turkish frustration: short, neutral refocus acknowledgment (5.09 s).
- Hypothetical pit rejoin: explicitly unavailable, no invented projection (6.69 s).

All four completed through Context, Core/response and grounding without errors. The ranking
reply supplied the numbers without an explicit yes/no, so directness still warrants review.
No microphone, TTS playback or live iRacing was involved. The 2.2-second planner median must
not be presented as the complete response time.

Reports above preserve their original implementation fingerprints. Subsequent numeric-metrics
hardening and this Core schema fix were checked separately rather than retroactively relabelling
planner scores. Automated verification: 372 tests pass, Ruff lint passes and mypy passes for all
90 source files. Existing unrelated repository-wide formatting differences were left untouched.

## V5: complete requests and distinct historical statistics

The model-facing contract now starts with two independent fields:

- `social_comment`: a strict boolean indicating social content anywhere in the turn;
- `requested_facts`: up to eight short descriptions of separately requested quantities,
  statistics and time windows, each bounded to 120 characters.

Only then does the model select temporal scope, capabilities, raw queries and unavailable
information. The application derives social/factual/mixed purpose from the boolean and scope,
instead of asking the model to confuse multiple factual requests with mixed social/factual
purpose. Pure social drafts require an empty inventory and no factual scope; factual scopes
require a nonempty inventory. Contradictory metadata is rejected before execution.

This is **one planner inference**, not another interpreter or model. The response budget rises
from 256 to 384 tokens to accommodate the bounded inventory. The inventory is model-authored
planning metadata, not evidence: it is discarded after validating the draft and never copied
into the evidence packet, driver memory or evaluation report. It is not proof that the model
listed everything correctly. Exact comparison against independently authored expectations is
still required; no coverage credit is awarded just because the model claims completeness.

Prompt guidance explicitly distinguishes quantity from time scope, average from peak/minimum,
and net change from rate of change. Capability and raw selections remain complementary. There
is no English/Turkish keyword router, new spoken command vocabulary or model-authored arithmetic.
The public `ContextPlan`, evidence refresh, Core and speech boundaries remain unchanged.

The new composition set was authored and run against v4 before the v5 changes: v4 passed 9/20.
Its [baseline report](evaluations/intelligence-qwen-v4-composition-2026-10-06.json) is retained.
A [prompt-only v5 attempt](evaluations/intelligence-qwen-v5-initial-development-2026-10-06.json)
passed 35/50 but regressed social turns; it was rejected in favor of the structured inventory.
The new set is development data, not a holdout, and does not satisfy the promotion gate.

### V5 measured results and current limits

The [first inventory-contract run](evaluations/intelligence-qwen-v5-development-2026-10-06.json)
passed 40/50. After adding contrast examples, the
[final October 6 development run](evaluations/intelligence-qwen-v5-development-final-2026-10-06.json)
passed **41/50** (19/25 English, 22/25 Turkish):

| Development set | Exact passes |
| --- | --- |
| Original | 12/12 |
| Additional controls | 15/18 |
| Composition/statistics | 14/20 |

The earlier v4 reports total 33/50 across separate runs (24/30 plus 9/20), not a single
matched 50-turn benchmark. V5 now selects the correct historical mean/maximum/minimum/delta
in all authored statistic controls. Both original fuel-range-plus-speed cases retrieve both
requested quantities; the English case still adds unrequested classification and fails exact
scoring. Original complaint-plus-question controls now retain mixed purpose.

Nine exact failures remain: five compound turns request unrelated or redundant evidence;
English venting incorrectly requests fuel data; one English hypothetical question gains
unrequested social purpose; and gratitude-plus-speed loses social purpose in both languages.
The venting error is a regression from v4, not an acceptable success just because the selected
fuel capability can execute. Six turns contain extra evidence, including that social regression.
Required-selection scoring reaches 46/50 but does not replace the 41/50 exact result.
There were no transport, response, plan-validation or execution errors in this run.

Planner median was **4.06 s**, p95 **10.58 s**, first call **12.68 s** on the configured CPU
profile. The earlier v4 30-turn median was 2.21 s; workload and cache differences mean this
is not an isolated estimate of the inventory's overhead. These are planner-only measurements,
not STT-to-audio latency. The inventory and larger output budget have not earned a performance
claim or release promotion.

V5 is the planner currently wired into replay and live conversation **on this development
branch**; "unpromoted" is an evaluation status, not a runtime feature flag or automatic v4
fallback. INT-07 remains open. Do not treat a race using this branch as an acceptance-tested
release. Calibration and locked seed datasets were not changed or used to tune v5.

October 7 verification: 395 automated tests pass; Ruff lint and mypy (90 source files) pass.
Tests cover strict inventory validation, catalog boundaries, complementary evidence selection,
distinct historical arithmetic and inventory exclusion from packets/reports. Scripted-model
tests verify contracts, not Qwen's semantic reliability.

### V5 full text-pipeline verification, October 7

Six synthetic replay questions were also exercised through Context, Core/response and
delivery-time grounding, using the configured CPU server. Five completed initially:

- English and Turkish historical averages: correct 50 m/s over the requested twenty seconds
  (10.03 s and 10.05 s).
- Turkish compound fuel/speed: both 12 fuel laps and 51 m/s were spoken (13.55 s), although
  Turkish phrasing remains awkward.
- English mixed frustration/fuel: correct 12 fuel laps, but no emotional acknowledgment
  (8.83 s). Correct planner purpose does not guarantee teammate-like wording.
- English venting: a short refocus acknowledgment without incident claims (5.92 s).
  This isolated success does not erase the venting failure in the planner benchmark.

The English fuel/speed question failed after 56.56 s. A diagnostic repeat showed a correct
fuel/speed template and references on both attempts, rejected because the planner also
retrieved classification and Core required its unrelated comparison inputs to be spoken.
Core now checks dependency completeness for each relationship actually referenced rather than
requiring every retrieved comparison. Partial comparisons and evidence-free comparison drafts
remain rejected; selected numeric evidence is still refreshed and substituted by application
code. There is no phrase router, extra model call or relaxed telemetry source requirement.
Five regression tests cover the corrected boundary. See INT-006 in
[known issues](known-issues.md).

Post-fix local-model checks:

- The same English compound question completed with both correct facts despite the same
  extra classification: 51 m/s and 12 fuel laps. It took **43.08 s**; this is far too slow
  for normal in-race conversation and is not a performance acceptance pass.
- A ranking control completed with both comparison references, correctly reporting P5 and
  field extent P6 (14.05 s), but still lacked an explicit yes/no answer to "Am I last?".

The managed CPU server used for these checks was stopped afterward; no user-owned server was
stopped. These are individual samples, not latency percentiles or proof of general reliability.

The saved 41/50 planner report predates this Core-only fix and retains its original source
fingerprint. It has not been relabelled as a post-fix whole-pipeline benchmark. No microphone,
TTS playback, racing load or real-race acceptance was tested here.

## Measure complete typed-reply latency

Run this from the repository root; iRacing and a microphone are not needed:

```powershell
.\.venv\Scripts\python.exe scripts\benchmark_intelligence.py `
  --show-replies `
  --output data\benchmarks\my-intelligence-baseline.json
```

The script starts the configured local model if necessary, or reuses an existing server.
It stops only its own child afterward. An external server's effective compute backend is
reported as unknown rather than inferred from the configuration. No models are downloaded.
Close the engineer/control panel's running session while measuring the isolated baseline.
Use a fresh output filename: existing reports are never overwritten.

By default it runs six existing development cases twice (12 turns): English/Turkish compound
fuel/speed, historical average speed, and social venting. Each repetition uses a seeded shuffled
order. Every question starts with fresh fixture/dialogue state; server caches are retained.
Planner and Core calls alternate naturally through the production adapters and orchestrator.
This is not a sequential-dialogue evaluation. The fixture is paused; it does not simulate
changing telemetry or race traffic.

For a longer run, add `--repeats 5`. Choose individual groups with repeated `--case` arguments,
for example `--case compound-en --case compound-tr`. An explicit alternate `--dataset` may
select another development dataset; calibration and locked splits are rejected by this tool.
The benchmark does not change prompts, inference settings or race decisions.

Each turn records:

- Context analysis, Core reply, response materialization, evidence refresh, grounding and
  total text-response wall time. Context/Core stages include their model calls; these nested
  timings must not be added twice.
- Each model attempt's wall time, prompt/generation time, prompt/completion token count and
  cached tokens where the provider exposes them. Core's existing repair call is counted.
- Exact planner-selection/purpose match, evidence availability/reason match, completion or
  sanitized failure stage, and reply action/word count.

The summary separates completed-turn latency from all-turn latency (including failures),
reports median/p95/max, and counts completed responses below five/ten seconds and at least
thirty seconds. First-in-run is a position in the run, not proof of a cold or warm cache.
Server startup/healthcheck is measured separately and excluded from reply latency. The
benchmark stops after a timeout or unreachable server and retains the partial report; it
does not queue more samples behind potentially unfinished model work.

Saved reports contain numeric timings, case/question hashes and reproducibility metadata,
not transcripts, reply text or request inventories. `--show-replies` displays the authored
fixture question and reply only in the console for manual review. A completed pipeline or
correct plan is not an automatic conversational-quality pass; directness, full answer coverage,
naturalness, uncertainty and social acknowledgment still require reviewing the reply.

Start with this isolated run, then repeat the same seed/questions while iRacing practice is
running to measure game-load interference. Label that run separately. This still uses fixture
telemetry. Actual live voice acceptance must additionally measure PTT release to first reply
audio, including STT, radio queueing and TTS; those stages are explicitly excluded here.

### First complete CPU timing baseline, October 7

The [saved 12-turn baseline](evaluations/intelligence-latency-cpu-baseline-2026-10-07.json)
used the managed CPU runtime with eight threads and zero GPU layers. Source/config/fixture
fingerprints did not change during the run. Server startup took 3.07 s separately.

- 11/12 pipelines completed, with 8/12 exact planner matches and one Core retry.
- Completed-turn median: 12.97 s; sample p95: 28.05 s; slowest: 32.32 s.
- Five completed turns were below ten seconds, none below five, and one exceeded thirty.
- Subsequent-in-run attempts ranged from 6.32 to 23.78 s, including one failed turn.
  Subsequent does not automatically mean warm.

The first English historical question took 22.70 s in Context and 9.63 s in Core.
Provider timings attributed 18.84 s and 5.64 s respectively to prompt processing.
Later planner requests commonly reused about 3,120 tokens despite the intervening Core call;
this run does not support an assumption that all prompt reuse is lost between the two roles.

The repeated English compound request dropped from 23.78 to 14.17 s. On the second attempt,
both model prompts were almost entirely cached: prompt processing totaled about 0.15 s,
while generation totaled 13.75 s for 102 planner tokens and 97 Core tokens. Cache improvements
alone cannot remove that generation cost. Refresh and grounding remained below a millisecond.

Manual review also matters: the final English vent produced an unrelated fuel answer; three
compound turns had non-exact plans even though the spoken replies included fuel and speed.
The second Turkish compound turn had an exact plan but failed Core validation after its repair
attempt, returning no reply after 21.90 s. Its exact validation cause is not established by the
content-free timing report. See INT-007 in [known issues](known-issues.md).

The script returned exit code one because a pipeline failed and still saved its complete
report. Completion counts are not answer-quality scores. This small sample describes this
workload, not a stable latency guarantee, and excludes STT/TTS/game load. Automated verification:
404 tests pass, Ruff lint passes, and mypy passes for source plus the benchmark entry point.

## Promotion rules

These rules become binding only after the promotion-grade workload is frozen:

- 100% correct temporal scope and authoritative-unavailable handling on safety controls;
- at least 95% exact-plan accuracy overall and 90% in each language and material category;
- at least 95% supported coverage, with accepted-plan accuracy reported separately;
- zero schema, invented-signal and unknown-capability escapes;
- zero model errors in a controlled replay run; and
- a measured latency/resource tradeoff against the Qwen portable baseline on the same
  machine and runtime profile.

A faster candidate does not earn promotion by losing compound, Turkish, social or
counterfactual behavior. A higher-quality enhanced candidate may use more resources, but its
profile and measured cost must remain visible. Full PTT-to-first-audio latency, STT and TTS
quality are separate end-to-end evaluations rather than Context Engineer scores.

## Next INT-07 work

1. Address the remaining development failures without relaxing exact scoring; add disjoint
   paraphrases and real sequential dialogue tests, including multi-scope compound requests.
2. Benchmark repeated prompts and catalog changes under the complete alternating planner/Core
   workload and iRacing load. Planner-only cache reuse is not full-radio latency.
3. Expand and review the locked bilingual workload without tuning against it, then freeze a
   candidate before running the release evaluation.
4. Decide whether domain-labeled training/calibration is worth pursuing for MiniLM or Laya;
   their naive zero-shot portable adapters are rejected.
5. Define one provenance-bearing learned temporal feature adapter for the enhanced profile.
6. Validate full PTT-to-audio quality and latency in a real race before any promotion claim.
