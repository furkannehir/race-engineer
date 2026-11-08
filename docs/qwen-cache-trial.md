# Offline Qwen KV-cache comparison

This is a retained comparison tool for the explicit `config/qwen-legacy.toml`
profile. The current source default is the English LFM MVP; this page records the
earlier Qwen experiments. The benchmark leaves the current default configuration
unchanged, and its shared fixture/runtime helpers also support the LFM benchmark.

## Experimental v8 temporal interpretation

V7/FP16 is the Qwen rollback baseline. `qwen-context-v8` is opt-in for evaluation,
typed replay or explicit custom configuration; adding it does not promote Q8.
Its wire schema remains v7's. The authored prompt interprets quantity and time
together before tool choice, qualifies the old unconditional position-to-current
classification mapping, and adds bilingual present/past-event contrasts. Driver
text is not keyword-matched or rewritten, and no additional LLM call is added.

Only v8 preserves the planner's requested time scope in `ContextPacket`. The
application attaches evidence timing from executed query operations or capability
descriptors, carries it through refresh/rendering bindings, and includes it in the
Core payload. Final grounding rejects referenced evidence with missing or
incompatible timing; orchestration rejects a refresh that changes/drops the
requested scope. V7 Core input omits the new optional metadata and its authored
prompt is frozen by a hash test, preserving the comparison baseline.

These guards check structured temporal compatibility. They do **not** parse the
meaning of free-form speech, independently reinterpret the transcript, distinguish
every historical event/window, or rescue a planner that labels both the request
and selected evidence incorrectly. Original starting-grid/fuel snapshots remain
unsupported; this change does not add those telemetry capabilities. Bilingual
paraphrases, correction histories and actual reply review remain necessary.

Run the four configurations without changing the live settings:

```powershell
.\.venv\Scripts\python.exe scripts\benchmark_qwen_cache.py --planner qwen-context-v7 --full-suite --validation --rounds 1
.\.venv\Scripts\python.exe scripts\benchmark_qwen_cache.py --planner qwen-context-v8 --full-suite --validation --rounds 1
```

Each invocation compares both cache precisions. `--validation` adds the existing
understanding-validation dataset, ten authored EN/TR temporal contrasts
(unavailable original lineup/fuel, current-after-past and past-after-current
follow-ups), and eight unsupported/partial-data questions about damage and tyre
condition. Their questions are not copied into the candidate prompt, and no
existing goldens, calibration or locked questions are modified. The resulting
matrix now has 100 turns per prompt/cache combination (including October 29's lap
timing and social-history controls), with seeded histories rather
than self-generated long conversations. The v8 candidate must be assessed against
v7 as well as against its own FP16/Q8 pair; a cache speed-up cannot excuse new
interpretation errors.

### Experimental unavailable-data handling

V8 also opts into `grounded-limitations.v1`; v7 keeps its existing Core prompt,
model wire schema and rendering behavior. The application binds a limitation to
explicit unknown IDs separately from numeric/claim evidence. Unknown does not
become zero, false, no reported damage, or a healthy car.

If all requested information is unavailable, the application supplies a short
English/Turkish limitation without another Core inference. Wording distinguishes
historical facts from future predictions; an ambiguous-reference reason instead
asks for clarification. These small fallback statements do not replace natural
generation for supported race answers or add a driver phrase router.

For partial answers, Core generates the supported body. The application retains
punctuation-separated clauses that cite supplied evidence placeholders, removes
unbound clauses, and prepends the application-owned limitation. Removed-clause
counts are logged without driver/model text. A body with no supported answer
requires one bounded repair, then fails visibly if still invalid. At playback
grounding, unavailable query/capability IDs are refreshed as well as spoken known
values. If an unknown becomes available, the stale limitation is rejected; this
slice does not automatically regenerate that answer.

This is a structural safeguard, **not semantic proof of every sentence**. A false
assertion sharing a clause with a valid placeholder, wrong units, an incorrect
planner selection, or incorrectly labeled unknowns can still escape it. Generic
fallbacks can also be less specific than a human explanation. Actual reply review
and further live acceptance remain required; no damage/tyre capability is added.

Reports identify the policy and per-turn model requests. Fully unavailable turns
now bypass Core, so latency changes against an older v8 policy must not be
attributed solely to cache precision. Compare FP16/Q8 within the same policy/run.

### V8 results (2026-10-28; not promoted)

The targeted starting-grid replay completed all eight semantic checks under v8
for each cache, including the previously failing Turkish question in both cache
orders (`qwen-cache-20261028T141345Z`). A subsequent four-way comparison included
all 84 development/validation questions per configuration:

| Configuration | Completed | Exact plans | Evidence checks | Combined checks | Median typed response |
| --- | ---: | ---: | ---: | ---: | ---: |
| v7 / FP16 | 83/84 | 75/84 | 72/84 | 71/84 | 646 ms |
| v7 / Q8 | 83/84 | 75/84 | 71/84 | 71/84 | 542 ms |
| v8 / FP16 | 84/84 | 78/84 | 76/84 | 74/84 | 669 ms |
| v8 / Q8 | 84/84 | 76/84 | 75/84 | 72/84 | 562 ms |

Reports: `qwen-cache-20261028T141806Z` (v7) and
`qwen-cache-20261028T142016Z` (v8), under ignored `data/benchmarks`. Both source/
input validity checks passed. The 8192 context, model weights, full offload and
cache allocations were verified. Actual game/STT/TTS timing is not included.

V8 corrected the starting-grid and past-event-fuel interpretations in this run,
including follow-ups; v7's expanded sequence also exposed the starting-grid error
under FP16. Thus the underlying weakness is not exclusive to Q8. V8 still had
new exact-plan mismatches: some target/time corrections were marked mixed/social
despite selecting the correct factual tool, and Q8 additionally retrieved an
unrequested ahead-distance signal for a damage-plus-speed request. Those mismatches
are retained rather than weakening the evaluation goldens.

Manual reply inspection found a separate blocker even when automatic checks
passed: v8/FP16 answered a damage-plus-speed request with the correct speed but
said no damage had been reported, although damage information was unavailable.
V8/Q8 showed the same unsupported negative assertion. Some historical limitation
wording also incorrectly called the unavailable past fact a "projection". Temporal
metadata does not prove free-form claims or unknown-state wording. See INT-010 in
[known issues](known-issues.md).

Decision: retain v7/FP16 live. V8 is an implemented, testable candidate with better
temporal behavior, not universally correct speech or an automatic Q8 promotion.
Next review unknown/unsupported factual claims and the remaining correction/
compound selection mismatches before a live-default decision.

### Unavailable-data safeguard results (2026-10-28; not promoted)

The revised v8 Core policy was exercised against all 92 development/validation
turns per cache, using the same paired order and seed. Report:
`data/benchmarks/qwen-cache-20261028T153414Z/report.json`. Source and input hashes
remained unchanged during the run, and effective cache/runtime settings verified.

| Configuration | Completed | Exact plans | Evidence checks | Combined checks | Median / p95 typed response |
| --- | ---: | ---: | ---: | ---: | ---: |
| v8 guarded / FP16 | 92/92 | 83/92 | 81/92 | 79/92 | 1109 / 1637 ms |
| v8 guarded / Q8 | 91/92 | 80/92 | 80/92 | 76/92 | 1007 / 1783 ms |

The original English damage-plus-speed reply now says:
`I can't assess that part from the available data. Your current speed is 51 metres per second.`
Both caches produced that reply instead of the unsupported negative damage claim.
Reviewed damage/tyre-only and damage-plus-fuel questions also explicitly acknowledged
unavailability; original-grid/fuel answers now use historical wording rather than
calling past data a projection. Neither cache made a healthy/no-damage assertion
in those reviewed cases. This is a targeted mitigation, not an overall semantic
accuracy score.

Remaining failures are retained: both caches retrieve extra facts for some partial
questions and guess ambiguous references. Q8 additionally failed Context schema
validation for `understand-relief-en`, changed the purpose of the English target
correction, and retrieved extra data for English damage-plus-speed. Its Turkish
compound answer needed one successful bounded Core repair. FP16 needed no Core
repairs. Turkish speed wording still sometimes drops "per second" or mixes in
English; see INT-011. Free-form claims within a bound clause remain unproven.

Core inference was bypassed on 26 fully unavailable turns per cache. Cache buffers
were still 1152 MiB (FP16) and 612 MiB (Q8), saving 540 MiB. Median/p95 above use
completed turns and exclude STT/TTS/audio. **iRacing was running**, with about
11.5 GiB already allocated before the trial server; game load was not controlled.
This is synthetic question replay alongside the game, not a live voice/race test
or an isolated timing comparison with older reports. Do not infer an overall
speed-up from this run. All trial-owned servers were closed afterward.

Validation: 556 tests passed; two Chatterbox tests skipped because NumPy is absent
from the main test environment. Ruff lint, formatting of the changed guard files,
and strict mypy passed. A repository-wide format check still reports pre-existing
formatting differences in unrelated files; those were not rewritten.

Decision: keep v7/FP16 live. The v8 unavailable-data safeguard is implemented and
testable, while planner selection, free-form semantics, Turkish units and Q8
regressions remain promotion blockers. Chatterbox BF16, PTT cues and live settings
are unchanged.

### October 29 live-delivery follow-up

The last live v8/FP16 run exposed a delivery issue separate from planner accuracy:
four answers were synthesized, then expired against the three-second age of their
pre-synthesis frame. The scheduler prematurely displayed Speaking and stored
unheard replies in history. The fixes and their recency tradeoff are documented
in [live conversation](live-conversation.md#freshness-and-session-isolation),
INT-012 and INT-013. They apply to normal live delivery too; the normal planner
remains v7/FP16. The separate local live-test profile selects v8/FP16.

V8 also isolates social radio-check/greeting Core generation from old factual
dialogue, rejects internal numeric-redaction markers, and distinguishes completed-lap
duration from race rank. Eight EN/TR controls were added without modifying older
goldens, bringing the suite to 100 questions per cache. Original last-lap timing is
not present in the telemetry catalog and should produce an honest limitation.

Real local replay report (October 29 Istanbul, filename uses UTC):
`data/benchmarks/qwen-cache-20261028T234957Z/report.json`.

| Configuration | Completed | Exact plans | Evidence checks | Combined checks |
| --- | ---: | ---: | ---: | ---: |
| revised v8 / FP16 | 98/100 | 90/100 | 87/100 | 86/100 |
| revised v8 / Q8 | 99/100 | 90/100 | 86/100 | 86/100 |

Source/input checks remained valid. The four new social-history questions returned
appropriate EN/TR acknowledgments under both caches, without leaked position or
redaction markers. Both English lap-duration questions returned an honest historical
limitation. One Turkish timing paraphrase also passed, while the other failed FP16
query validation and caused unrequested speed retrieval under Q8. English relief
still failed Context validation under both caches; compound/ambiguity and unit
issues remain. One Turkish compound needed a successful Core repair per cache.
These are retained failures, not an overall reply-accuracy score or a Q8 promotion.
This changed suite/order cannot directly establish improvement over earlier runs.

Validation: 571 tests passed, two NumPy-dependent tests skipped in the main environment;
Ruff lint and strict mypy passed. Virtual-clock live tests exercise six-second
preparation with continuing fresh telemetry, timeout, disconnect, stale telemetry,
session change, mute, generation failure, interrupted output and delivered history.
Existing critical-priority, cancellation, shutdown and PTT tests still pass.

An actual audio smoke test completed EN/TR replies and automatic calls with one
unchanged CUDA/BF16 Chatterbox worker. The English reply prepared in 3009.5 ms and
played; Turkish prepared in 2613.2 ms and played. Activity advanced through
preparing, speaking and idle. Artifact:
`data/diagnostics/live-delivery-smoke-20261029.txt`. This verifies device playback
completion, not human listening quality or combined race acceptance. All test-owned
workers were closed. Another live v8/FP16 session is still required.

## Baseline cache experiment

This is an opt-in memory/latency experiment, **not a live configuration change**.
It compares FP16 K/V with Q8_0 K/V using the same installed Qwen3-4B
Instruct-2507 Q4_K_M file, v7 Context/Core prompts, 8192-token context,
single slot, full CUDA offload, eight configured CPU threads and existing
request timeout. Chatterbox BF16, STT, PTT cues and the normal launcher are untouched.

## Run without iRacing

Close the engineer/game and other GPU workloads for an isolated comparison.
No race, model download, hosted API, microphone or audio playback is needed.
The local Qwen CUDA setup must already be installed. From the repository root:

```powershell
.\.venv\Scripts\python.exe scripts\benchmark_qwen_cache.py
```

The default run uses 20 representative authored questions: ten English and ten
Turkish. They cover radio checks, short position requests, venting, unsupported
starting-grid questions, corrections/topic changes with seeded dialogue history,
compound requests, future requests, historical telemetry and follow-ups. Each
cache runs those questions twice, with opposite trial order:

1. FP16, then Q8.
2. Q8, then FP16.

Each trial has a fresh owned server and excluded bilingual warm-up, followed by
the same shuffled question order for its paired round. Greedy production request
sampling and the same server seed are used. A prompt-prefix/KV cache can still
be reused within a trial, just as in production; numeric provider timing/cache
metrics are retained in the report. The history controls use authored dialogue,
not a long self-generated conversation. STT/TTS/end-to-end radio timing is excluded.

For a short pilot:

```powershell
.\.venv\Scripts\python.exe scripts\benchmark_qwen_cache.py --rounds 1 --case understand-radio-check-en --case understand-radio-check-tr --case compound-en --case compound-tr
```

`--full-suite` uses all questions in the two development datasets instead of
the representative subset. Calibration/locked sets and real recordings are not
used. `--port` changes the separate trial loopback port (default 8097).
`--timeout-s` bounds each trial (default 900 seconds).

## Verify actual runtime settings

An initial FP16/Flash-Attention-auto preflight records how the current runtime
resolves `auto`. Both measured cache candidates explicitly request Flash
Attention **on**, so attention mode cannot masquerade as a cache optimization.
The pinned installed b10964 CUDA 13.3 runtime's own logs and `/props` must verify:

- Effective Flash Attention enabled, not merely a requested `auto` flag.
- K and V both using the requested precision.
- Context still 8192, parallelism still one, matching model and server seed.
- Every model layer offloaded and nonzero CUDA model/KV allocations.

Unverifiable/unsupported settings fail visibly. There is no CPU fallback or
external-server reuse. The tool refuses an occupied trial port and never stops
an unrelated server. `ConversationTrialOptions` supplies typed diagnostic-only
launch arguments; the normal `ConversationServer` command remains unchanged
when no trial is supplied. Timeout/Ctrl+C close the owned process and retain
partial per-trial reports.

Exit code zero means the controlled trials completed with unchanged inputs/source,
not that every semantic check passed or that Q8 is approved. Inspect the paired
regressions, pipeline failures and manual replies even after a successful run.

## Outputs and interpretation

Each invocation creates a new ignored directory at
`data/benchmarks/qwen-cache-<UTC timestamp>`. Existing output directories are
never overwritten. `--output` can name a new subdirectory under `data/benchmarks`.

- `report.json`: aggregate metrics, paired regressions/improvements/shared failed
  checks, model/runtime/source/input fingerprints and comparison-validity flags.
- `preflight-auto/runtime.log`: requested/effective attention/cache allocation proof.
- `round-*/runtime.log`: native runtime allocation/inference diagnostics.
- `round-*/result.json`: incremental timings, effective settings and automatic checks.
- `round-*/review.jsonl`: authored question, seeded history and grounded reply for
  manual review. This is synthetic content, never microphone or race-session text.

The automatic checks compare planner purpose/time/requests and expected evidence,
and distinguish Q8-only regressions from cases already failing FP16. Successful
pipeline execution is **not** semantic correctness. Reply wording, arithmetic,
yes/no contradictions and completeness still need manual review; automatic reply
accuracy is deliberately unset. Paired outcomes are not an automatic promotion gate.

Native CUDA model/KV/compute buffer sizes are exact runtime-reported allocations,
not total process VRAM. Whole-device `nvidia-smi` samples include other programs
and exclude unsampled peaks; WDDM per-process attribution is not assumed. GPU
samples are taken outside per-turn timers. Source/input changes invalidate the
comparison, even though partial reports remain available. Small p95 samples and
isolated GPU tests do not establish racing-load latency or long-session stability.

Only the cache changes: Q8 must not halve the claimed memory of the entire
four-bit model. Promote a candidate separately only after reviewing bilingual
failures/replies and testing with Chatterbox and iRacing running.

Runtime option reference: [official llama.cpp server documentation](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md).
The benchmark additionally validates the installed version's flags and actual logs.

## Initial results (2026-10-28)

The installed RTX 5080 / b10964 CUDA 13.3 runtime verified Flash Attention already
enabled under `auto`. Both controlled candidates retained all 37 GPU-offloaded
layers and the same 2375.91 MiB CUDA model buffer. K/V buffers were 1152 MiB for
FP16 and 612 MiB for Q8: **540 MiB saved**, not a halving of the whole model.

The balanced representative run (`qwen-cache-20261028T132840Z`) performed two
opposite-order rounds, 40 questions per cache:

| Measurement | FP16 cache | Q8 cache |
| --- | ---: | ---: |
| Median typed pipeline latency | 669 ms | 579 ms |
| p95 typed pipeline latency | 819 ms | 731 ms |
| Pipeline completed | 40/40 | 40/40 |
| Exact planner checks | 40/40 | 40/40 |
| Evidence checks | 36/40 | 36/40 |

The shared evidence misses were the two future-pit-projection controls repeated
across rounds; both cache variants abstained from inventing a projection. This
does not make their evidence checks passes. Some Turkish replies changed wording
and remained awkward. The longest reported prompt in this subset was 5925 tokens,
not a near-8192-token quality stress test.

A broader development run (`qwen-cache-20261028T133127Z`) tested all 62 questions
per cache, including additional paraphrases and seeded-history controls:

| Measurement | FP16 cache | Q8 cache |
| --- | ---: | ---: |
| Median completed-turn latency | 615 ms | 531 ms |
| p95 completed-turn latency | 815 ms | 741 ms |
| Pipeline completed | 61/62 | 61/62 |
| Exact planner checks | 56/62 | 55/62 |
| Evidence checks | 52/62 | 51/62 |

There were ten paired cases failing the combined checks in both configurations,
including a shared Turkish vent parse failure. Additionally, Q8 alone failed
`understand-starting-grid-tr`, question ID `de1d737d172e2788`:

> Yarışa kaçıncı başladık?

FP16 treated this as historical/unsupported starting-grid information and replied
that the starting position was unknown. Q8 requested current classification and
incorrectly described the current P5 as the starting position. This is a real
observed reply regression, not merely different wording. Requested cache settings
and the unchanged capacity were verified in both native logs. Both report source/
input validity flags were clean.

The targeted follow-up (`qwen-cache-20261028T133459Z`) repeated all four EN/TR
starting-grid paraphrases in both trial orders. FP16 passed 8/8 combined checks;
Q8 passed 6/8, reproducing the same Turkish question's regression in both rounds.
Run that diagnostic with:

```powershell
.\.venv\Scripts\python.exe scripts\benchmark_qwen_cache.py --full-suite --rounds 2 --case understand-starting-grid-en --case understand-starting-grid-tr
```

**Decision: keep the live FP16 cache unchanged.** Q8 is a promising memory/latency
candidate, not an accepted live default. Reproduce the temporal/starting-grid
regression, review the existing common failures and reply wording, then rerun a
separate comparison before promotion. Do not adjust goldens/prompts during a
performance run or conceal baseline failures by reporting only completed turns.
