# Conversation evaluation protocol

Updated 2026-09-23. This protocol freezes the CE-02 workload and promotion gates before a
smaller context judge or accelerated runtime is selected. It evaluates semantic planning;
telemetry values still come from deterministic application code.

## CE-04 planning caveat (2026-09-24)

The v1 workload and recorded results below remain unchanged. A planning audit found one
exact question shared by calibration and locked groups; family-name validation alone does
not establish semantic split separation. The locked workload contains independent turns,
with multi-turn sequences only in development. Its non-clarification-based supported
coverage also includes unsupported/unavailable requests. Treat the report as a descriptive
baseline, not proof of general conversational quality or leak-free promotion evidence.

CE-04 will add a separately versioned dialogue suite, split audit, and explicit answerability
metrics before model selection. See the
[conversational-core evaluation design](conversational-core-design.md#evaluation-before-choosing-a-model).
Do not silently repair this frozen dataset or rescore old reports under new definitions.

## Workloads and separation

All committed datasets use `conversation-eval.v1`. A case declares its language, exact
query plan or clarification, expected reply status, paraphrase family, category, tags, and
synthetic fixture. Groups contain independent paraphrases with fresh history. Sequences
retain bounded dialogue history and may advance the replay frame.

| Split | File | Size | Use |
| --- | --- | ---: | --- |
| Development | `fixtures/conversation/cases.json` | 20 turns | Existing difficult cases and prompt development |
| Development | `fixtures/conversation/holdout.json` | 11 turns | Historical post-tuning smoke cases; retained for regression continuity |
| Calibration | `fixtures/conversation/calibration.json` | 60 turns, 30/language | Threshold selection only |
| Locked | `fixtures/conversation/locked.json` | 200 turns, 100/language | Final model/routing comparison; never prompt or threshold tuning |

Paraphrase families cannot cross splits. Dataset validation rejects duplicate case IDs,
families, questions, or expected queries. The locked split requires at least 100 English
and 100 Turkish turns. Any intentional content change requires a revision bump; do not fix
a model by silently editing a locked expectation after seeing its result.

The suites include short and one-word forms, negation, Turkish/English code switching,
compound requests, ambiguity, unavailable strategy, unsupported preference/emotional
remarks, and multilingual factual questions. Multi-turn refresh, topic carry-over,
language switching, prompt injection, and missing facts remain in development sequences.
Stale/future snapshots, disconnects, and session changes are deterministic orchestration
guards in `tests/test_conversation.py`, `tests/test_live_conversation.py`, and
`tests/test_iracing_adapter.py`; they are not sent to the language model because inference
must not run for stale input and an in-flight invalidation must discard its answer.

## Running the explicit local suite

The normal test suite uses scripted planners. It never downloads a model, starts a
microphone, or performs network inference. A real-model run is an explicit local action:

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_conversation.py `
  --config config\default.toml `
  --manage-server `
  --dataset fixtures\conversation\locked.json `
  --output data\evaluations\qwen-cpu-locked.json `
  --run-label qwen-cpu-locked `
  --quiet
```

`--manage-server` uses the same bounded launcher as the panel. If a matching server is
already listening, the report marks it external/unmanaged because its backend cannot be
verified. Stop that server first for a reproducible owned-runtime startup measurement.
Exit code 1 means at least one expectation failed; the JSON report is still written.

Reports use `conversation-eval-report.v1` and identify the git state, model and revision,
quantization, runtime build, configured/effective backend, threads, context limit, hardware
and driver, dataset revisions, and process resources available from the OS. Per-turn rows
contain a one-way question ID, expected/observed plan, reply status, failure reason, and
timings. They never contain the question, transcript, prompt, reply text, or microphone
audio. `data/` is git-ignored by default.

`cold` means the first planner request after the process lifecycle began; later requests
are marked `warm`. Planner time is measured independently. Application time is fresh fact
retrieval and deterministic rendering after subtracting planner time. Startup and wall
time are separate. Typed replay cannot measure capture, ASR, TTS, radio wait, or first
audio, so those stages are explicitly listed as unmeasured rather than reported as zero.
End-to-end speech evaluation requires deliberately supplied audio in a later workload.

## Frozen promotion gates

These gates were fixed before the CE-02 locked run. The Qwen CPU run is a descriptive
baseline, not automatically a release pass.

- All deterministic freshness, session/reset, missing-fact, critical-policy, and radio
  regression tests pass. A model never supplies telemetry numbers or bypasses scheduling.
- For a small judge, exact accepted-plan accuracy is at least 95% in each language and
  supported-case coverage is at least 70% in each language. Report both counts; abstaining
  on everything cannot pass.
- Expected opponent/topic clarifications achieve at least 90% exact accuracy. Unsupported
  and injection cases must remain bounded to their expected plan; model errors count as
  failures rather than disappearing from the denominator.
- A promoted hybrid's combined exact-turn accuracy must meet or exceed the same-hardware
  Qwen baseline in each language and must retain every deterministic safety guard.
- The small judge's warm CPU planning p95 target is at most 300 ms on the selected reference
  machine. The later speech workload targets at least a 25% reduction in warm
  PTT-release-to-first-audio p95 against the same-hardware Qwen baseline.
- Hardware comparisons use identical weights, prompt, context, datasets, and runtime
  settings. CPU remains a required path; CUDA/Vulkan support needs its own real report.

Accuracy is exact plan plus reply-status agreement. Accepted accuracy counts turns where a
non-clarification plan was returned. Supported coverage is the proportion of expected
non-clarification turns receiving such a plan. Clarification accuracy is reported
separately. p50/p95 use linear interpolation over all relevant turns, including failures
and timeouts.

## Reproducible CI checks

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_conversation_evaluation.py `
  tests\test_conversation.py
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m mypy
```

The evaluation tests validate split isolation, bilingual locked-set size, content-free
results, percentile calculations, fresh history for paraphrase groups, and retained history
inside multi-turn sequences.

## Qwen CPU baseline — 2026-09-23

The owned llama.cpp b10964 CPU runtime ran Qwen3-4B-Instruct-2507 Q4_K_M with eight
threads, an 8192-token context, one parallel slot, and zero GPU layers. The reference
machine was Windows 11 with a Ryzen 7 9800X3D, 64 GB RAM, and an RTX 5080 present but unused
for inference. The complete aggregate is in
[`evaluations/qwen-cpu-locked-2026-09-23.json`](evaluations/qwen-cpu-locked-2026-09-23.json);
the content-free per-turn report remains git-ignored under `data/evaluations`.

| Metric | Result |
| --- | ---: |
| Exact turns | 164/200 (82%) |
| English exact turns | 83/100 (83%) |
| Turkish exact turns | 81/100 (81%) |
| Accepted-plan accuracy | 84.13% |
| Supported-case coverage | 96.67% |
| Opponent clarification accuracy | 5/20 (25%) |
| Warm planner p50 / p95 | 1,023.54 / 1,284.60 ms |
| Cold first planner turn | 4,519.47 ms |
| Fresh retrieval/render p50 / p95 | 0.089 / 0.129 ms |
| Runtime startup | 3,068.65 ms |
| Peak working set | 5,660,618,752 bytes (about 5.27 GiB) |
| Model errors / compute fallbacks | 0 / 0 |

Position, fuel remaining, fuel consumption, and gap-behind cases were perfect. The main
measured gaps were opponent clarification (25%), gap-ahead interpretation (65%),
fuel-to-finish interpretation (70%), and unsupported/social handling (75%). The evaluator
was run twice without tuning the locked set and returned the same 164/200 decisions; warm
p95 varied from about 1.25 to 1.28 seconds. This establishes the Qwen CPU comparison point,
not a claim that it meets the later small-judge gates.

The report records a dirty git tree because CE-02 had not yet been committed. Its exact
source/config/dataset/fixture SHA-256 fingerprints are therefore part of the committed
aggregate. Re-running after commit should produce a clean git identity; model decisions
are expected to remain the same if all recorded fingerprints and runtime settings match.
