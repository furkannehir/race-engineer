# CE-04 dialogue evaluation and candidate adapters

CE-04c/d infrastructure is implemented. The separately versioned
[dialogue protocol](../fixtures/dialogue/README.md) covers 118 synthetic typed turns
in paired English/Turkish scenarios. Its oracle and always-abstain controls validate
the harness; they are **not model-quality results**. CE-02 data/results are unchanged.

## Boundaries

`SemanticJudge` receives immutable per-turn context. All three candidates use exactly
the same compact projection: current utterance, explicit/default language, topic,
pending clarification, the last two meanings, last response/delivery status, capabilities,
opponent IDs, battle state and bounded event categories. No expected labels, numeric
answers, generated wording, memory writes or tool execution are exposed to the judge.
The deterministic controller remains responsible for fresh facts and session guards.

Qwen-v2 uses the existing owned llama.cpp CPU runtime over loopback with schema-constrained
JSON. This is separate from the live v1 planner. MiniLM and Laya each use one owned,
persistent CPU subprocess. Normal imports/tests need no optional ML packages. There is
one inference request at a time and no worker queue; timeout/cancellation terminates the
owned worker. Oversized context is a per-turn error, never silent truncation. Worker
stderr and provider bodies are not propagated into normal reports.

The initial encoder experiment evaluates 29 hypotheses: 14 query/reference combinations,
five modes, three social acts, two languages, three reference-only completions and two
clarification choices. MiniLM uses three-way NLI entailment scores, treating neutral as
uncertainty. Laya uses its binary `noul` heads. Both batch eight rows per CPU forward and
include **all** tokenization, heads/batches, composition and IPC in turn timing. Joint
acceptance checks every selected/absent query, choice threshold/margin, and semantic
contract; conflicting combinations abstain. These are internal meanings, not required
spoken phrases. Failure of this head composition does not prove every possible use of
either model would fail.

The initial context is intentionally compact enough for MiniLM's 512-token limit;
requests exceeding the candidate limit remain errors and count in evaluation. Laya has
a 1024-token evaluation cap and reserves space for heads. This compact semantic view
is not a replacement for the controller's full identity/freshness checks.

`JudgeRouter` supports at most one configured fallback after abstention or invalid
semantics. Understood-but-unsupported facts or genuinely missing references never trigger
another model. Both proposals undergo the same validation and shared deadline. The router
is not attached to the live panel; CE-04e can legitimately leave it unused.

## Reproduce locally

Normal CI: run pytest; no downloads, microphones or model servers are used. Large-model
evaluation is an explicit separate action. Use a fresh output filename; reports are not
overwritten. This machine's existing STT Python contains compatible CPU PyTorch and
Transformers. Laya's optional package is isolated under ignored `data/`, so the speech
environment's installed packages are not modified.

```powershell
# Explicit, one-time package/checkpoint download. All evaluation inference is offline.
.\data\stt-prototype\runtime\Scripts\python.exe -m pip install --no-deps --target data/dialogue-prototype/packages laya==0.3.20
.\data\stt-prototype\runtime\Scripts\python.exe scripts/setup_semantic_models.py

# Fixture controls, with no heavyweight dependency.
.\.venv\Scripts\python.exe scripts/evaluate_dialogue.py --candidate oracle --split locked --output data/evaluations/oracle-new.json
.\.venv\Scripts\python.exe scripts/evaluate_dialogue.py --candidate abstain --split locked --output data/evaluations/abstain-new.json

# Calibrate each encoder, then freeze its operating point before locked evaluation.
.\.venv\Scripts\python.exe scripts/evaluate_dialogue.py --candidate minilm --split calibration --output data/evaluations/minilm-cal-new.json
.\.venv\Scripts\python.exe scripts/evaluate_dialogue.py --candidate minilm --split locked --calibration data/evaluations/minilm-cal-new.json --output data/evaluations/minilm-test-new.json
# Repeat those two commands with --candidate laya and distinct laya filenames.

# Qwen has no score threshold to fit; use the frozen full-context prompt.
# An already-running external server is not accepted as controlled CPU evidence.
.\.venv\Scripts\python.exe scripts/evaluate_dialogue.py --candidate qwen --split locked --port 8091 --output data/evaluations/qwen-test-new.json
```

Use `--python` for a separate compatible CPU ML environment; the application environment
does not need PyTorch, Transformers or Laya. Pinned checkpoint SHAs live in the explicit
setup script. Reports record installed library versions, CPU threads, checkpoint/runtime
revisions, source and dataset fingerprints, hardware, startup time and process peak
working set. Report timing includes errors/abstentions, not just successful answers.
Peak working set is a process lifetime observation, not combined simulator/ASR/TTS RAM.
The screening host may also run ordinary development checks; this is not a controlled
idle-system or in-race benchmark. Treat timings as screening measurements, not guarantees.

Do not benchmark candidates concurrently. The held-out benchmark runs them sequentially
and closes each owned worker/server; all processing stays local. Model downloads are not
part of normal application startup.

## Promotion and remaining evidence

`evaluation/routing.py` records failed gates, including missing sample volume, language
accuracy, coverage, clarification, inference errors and small-judge CPU p95. It never
changes live configuration. `scripts/summarize_dialogue.py` requires matching dataset,
adapter revision and source fingerprints, then publishes a content-free screening report.

This initial suite can reject a candidate or retain the existing Qwen route. It cannot
establish release qualification: only 34 judged turns per scenario language (35 English/
33 Turkish expected replies after language overrides), correlated bilingual
scenarios, a single high-end CPU, and no listening/audio/co-residency or frame-time tests.
The larger independent holdout, one-word ASR recordings, mixed-act playback integration,
critical-radio interruptions and representative hardware remain explicit gates. No new
live replacement or automatic fallback is enabled by CE-04.

Upstream API references: [MiniLM model card](https://huggingface.co/MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli),
[Laya repository](https://github.com/NandhaKishorM/laya), and
[Laya multilingual checkpoint](https://huggingface.co/convaiinnovations/laya-multilingual).
Upstream throughput/quality claims are not used as local performance evidence.
