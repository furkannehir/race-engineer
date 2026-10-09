# Isolated Laya radio-interpreter experiment

Started 2026-10-09 on `experiment/laya-interpreter`, after committing radio diagnostics
and the deferred research backlog as `ee50f5b` and fast-forwarding/pushing `main`.

## Scope and boundaries

This is an offline component experiment, not a live Context Engineer replacement. It asks
Laya typed questions about a driver message's purpose, time scope and independently requested
topics. No generated text, phrase router, model-authored arithmetic, telemetry execution,
Core inference, speech recognition or synthesis is involved. Natural wording is permitted;
the topic list is an internal diagnostic vocabulary, not a spoken command set.

The pilot contains 28 authored development examples, 14 English and 14 Turkish. It covers
presence checks (including after a position conversation), short position questions, fuel,
gaps, frustration, mixed/compound requests, fuel follow-ups, ambiguous requests, hypothetical
pit rejoin, historical speed, broad race overviews and unrelated requests. Expected labels
were authored before either run; they were not relabelled to match candidate outputs.
Each case is repeated twice in a seeded shuffled order. Repeats are not independent examples.

These decisions do **not** specify opponents/entities, windows, query operations, capability
arguments, uncertainty policy or executable evidence requests. Passing this component test
would not establish a complete `ContextPlan` implementation or promotion eligibility.

## Pinned existing runtime

No package installation, model download or training was performed:

- isolated Python: `data/stt-prototype/runtime/Scripts/python.exe`;
- package: local `data/dialogue-prototype/packages`, Laya **0.3.20**;
- checkpoint: `convaiinnovations/laya-multilingual`, revision
  `e4e9ddf21a7b1903b7acffd8814ad4307bf63a67`, as recorded in local provenance;
- model directory: `data/dialogue-prototype/laya`;
- effective device: CPU, eight PyTorch threads;
- PyTorch **2.14.0+cpu**, Transformers **5.17.0**;
- Hugging Face offline flags enabled before loading.

This is the already-downloaded checkpoint/package, not a claim to have evaluated the newest
upstream release or every Laya checkpoint. Existing October 4 results remain unchanged:
the old single-topic catalog adapter passed 3/12, with 0/6 Turkish. Its complete-plan scoring
is different from this component test, so the scores must not be compared as a regression.

## Run without iRacing or Qwen

Use the isolated runtime, not the desktop `.venv`:

```powershell
.\data\stt-prototype\runtime\Scripts\python.exe scripts\benchmark_laya_interpreter.py `
  --state-format text `
  --repeats 2 `
  --output data\benchmarks\laya-radio-my-run.json
```

Use `--state-format structured` for the JSON-state comparison. Both formats preserve authored
history and distinguish the current message; plain text omits JSON wrapping for cases without
history. Each run keeps the same questions, expectations, seed and checkpoint. Progress is
printed per turn. Choose a new output filename: existing files are never overwritten.

Exit code **0** means every component expectation matched, **1** means at least one failed
(the report is still saved), and **2** means setup could not proceed. An exact component
match is not automatic model promotion. The experiment never starts or stops a Qwen server.
Stop racing/live voice processing while repeating a resource comparison; the report is not
a simultaneous-game-load benchmark.

## First CPU measurements

| Measure | Structured state | Plain conversation text |
| --- | ---: | ---: |
| Unique development cases | 28 | 28 |
| Attempts (two repeats) | 56 | 56 |
| Runtime errors | 0 | 0 |
| Exact purpose + scope + topics | 0/56 | 4/56 |
| English exact matches | 0/28 | 4/28 |
| Turkish exact matches | 0/28 | 0/28 |
| Purpose matches | 14/56 | 20/56 |
| Time-scope matches | 24/56 | 28/56 |
| Exact topic-set matches | 16/56 | 20/56 |
| Inference median | 599 ms | 537 ms |
| Inference p95 | 678 ms | 636 ms |
| Slowest inference | 732 ms | 666 ms |
| Model/import startup, excluded above | 8.20 s | 7.08 s |

Local, ignored reports:

- `data/benchmarks/laya-radio-decisions-2026-10-09-pilot1.json`;
- `data/benchmarks/laya-radio-decisions-2026-10-09-text1.json`.

The plain-text passes are the English mixed relative-pace and race-overview cases, each twice.
Both formats still fail full presence-check expectations and many basic requests. The model
often assigns a current race time scope to non-factual turns and retrieves irrelevant topics.
The valid-response shapes and zero runtime errors do not count as semantic success.

The runs demonstrate sub-second, comparatively tight CPU timings for this small input and
eight typed questions. They **do not** establish acceptable interpretation accuracy, reliable
abstention, or live-race performance. Scores use a fixed **0.5** topic threshold for screening;
no threshold tuning was done. Raw probabilities are uncalibrated and cannot authorize routing.

## Reports, privacy and next investigation

Reports contain case IDs, hashed inputs, expected/observed labels, numeric scores/timings,
runtime revisions, source/input fingerprints and machine metadata. They do not store question
text or dialogue. Fixture wording is explicitly authored and version controlled; no driver
audio or private session transcript was imported. Detailed reports remain under ignored `data/`.

Next investigate the question interface and checkpoint/package compatibility, including any
newer upstream release, before judging the model family. If another checkpoint is evaluated,
pin it separately and retain these results. Do not silently update the STT or desktop runtime.
No training, confidence calibration, fast acknowledgement or live routing change is included.
MiniLM/SetFit interpreter training and Core specialization remain future research under
RES-01/RES-02 in [the roadmap](roadmap.md).
