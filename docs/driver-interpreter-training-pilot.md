# Driver-action training pilot

Status, 2026-10-11: **two local CPU pilots completed; experimental social live trial enabled;
neither checkpoint promoted as a full interpreter**. The first
used the 800-row draft; the second uses the expanded 1,400-row v2 draft. The owner
authorized each exploratory training run and subsequently requested a race trial.
This does not approve their annotations or replace full telemetry interpretation.
The first-run results are preserved
below; [v2 results](#second-pilot-v2-2026-10-11) are recorded separately.

## What was trained

The pinned `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` sentence
encoder feeds a 12-output sigmoid classification head: the corpus's 11 driver actions
plus `needs_clarification`. Multiple actions can be predicted. This is a **plain
PyTorch pilot, not SetFit**. We reused the existing optional STT Python environment
without upgrading or installing training packages or changing live dependencies.
The checkpoint was downloaded once; training and inference use local files only.

Inputs contain only the observed driver text and supplied role-separated dialogue.
Gold meanings, request focuses, tags, IDs and synthetic-ASR `intended_text` are never
model input. This classifier does **not** learn the full request parser: fuel versus
position, first versus last, opponent references, time windows, calculations and
generated replies remain outside this experiment. Correctly identifying an
information request does not mean we have understood its entire question.

## Protocol

- Seed 17, group-disjoint train/validation/test: **552 / 126 / 122** examples.
  Every split contains equal English and Turkish counts. Related bilingual variants
  stay together. Group assignments are in a separate ignored manifest; the corpus
  itself remains `draft` and `unassigned`.
- Actions represented by one group are train-only. Radio checks, acknowledgments,
  out-of-scope messages and greetings have **no positive test coverage**. Corrections
  have train/test coverage but no positive validation coverage.
- Baseline: frozen, unadapted embeddings compared with fixed bilingual action
  descriptions by cosine similarity. This is a small similarity baseline, not the
  earlier Laya/mDeBERTa benchmark or a strong zero-shot NLI baseline.
- Warm a classification head for 400 minibatches. Compare 100 additional head-only
  updates against 100 matched minibatches updating the head and last two encoder
  layers. Sampling, initialization, loss and head learning rate are matched for
  that comparison. Encoder dropout is disabled in both branches, including during
  adaptation; the adapter branch still computes gradients.
- Batch 16, eight CPU threads, float32, masked normalized mean pooling, maximum
  128 tokens. Left truncation preserves the current turn. No corpus input exceeded
  that limit. Weighted binary cross entropy uses **training-only** class weights.
- Encoder learning rate 0.00002; warm-head rate 0.02; matched-head rate 0.001.
  Adaptation checkpoints at 50 and 100 steps; one global threshold selected from
  0.3/0.4/0.5/0.6/0.7 using validation exact match. Candidate selection is also
  validation-only, ties favoring the frozen encoder. Test predictions are obtained
  after those decisions; no hyperparameters were changed after inspecting test results.
- Additional no-input sanity check: predict the constant majority vector fitted on
  training labels. Here that is always `request_information`. This check was added
  after the first run to expose class imbalance; it did not change candidate selection.
  The training runner includes it in future reports, and a regression test reproduces
  its 52/122 exact matches on this split. It is not in the original saved report.

## Results

Exact match requires **all action bits and the clarification bit** to match the
draft annotation, with no extra labels. Macro F1 below averages only the eight
targets with positive test support; it is not an all-label deployment score.

| Candidate | Validation exact | Test exact | Test micro F1 | Supported-label macro F1 |
| --- | ---: | ---: | ---: | ---: |
| Unadapted similarity baseline | Not used to select trained candidate | 6.56% | 0.1210 | 0.1090 |
| No-input majority sanity check | Not used to select candidate | 42.62% | — | — |
| Frozen encoder + trained head | 62.70% | 50.00% | 0.5840 | 0.2800 |
| Last-two-layer adaptation + head | 63.49% | 50.00% | 0.5833 | 0.2828 |

The adapter at step 50 was saved as the selected playground model because of its
validation score, not because it won on test. Its validation edge is **one example**
out of 126; test gives no convincing advantage over the frozen encoder.
Selected-model test exact match: English **31/61 (50.82%)**, Turkish **30/61 (49.18%)**.

Selected-model action F1: information requests **0.8551**, car-problem reports
**0.6452**, venting **0.7619**. Corrections, preference changes, temporary back-off
feedback, unclear messages and clarification have **zero true positives** in this
test. There are also false radio-check/acknowledgment/greeting predictions. Do not
treat missing positive coverage as passing those categories.

Warm single-message classification, 40 test inputs on this machine:
**9.225 ms median, 16.686 ms p95**, including tokenization, encoder and head.
This excludes model loading, STT, Qwen reasoning, telemetry retrieval and TTS.
The recorded training/evaluation duration was 27.04 seconds, excluding download and
Python startup. These timings are not portable performance guarantees.

Training shows a modest useful signal over the majority check, and the small
classifier is fast here. It does **not** establish reliable driver understanding.
Eight hundred paraphrases are not eight hundred independent situations. Every
label is still an assistant-authored draft, some families were previously probed,
and grouping is provisional. Compound families can still overlap semantically.
This is exploratory synthetic holdout evidence, not a locked acceptance test.

## Try it in PowerShell

The pilot is already trained on this workstation. From the repository root:

```powershell
.\data\stt-prototype\runtime\Scripts\python.exe scripts\try_driver_interpreter.py
```

Or inspect one message:

```powershell
.\data\stt-prototype\runtime\Scripts\python.exe scripts\try_driver_interpreter.py --prompt "Are you there?"
.\data\stt-prototype\runtime\Scripts\python.exe scripts\try_driver_interpreter.py --prompt "Beni duyuyor musun?"
```

Both returned `radio_check` in smoke tests. Both phrases are **in the training
corpus**, not independent accuracy evidence. Try new wording and compare with your
own intended meaning. Scores are uncalibrated; an empty action list is possible.
The tool prints interpretations, not engineer replies; it records no entered text.

Messages are independent by default. For supplied context, use `/history on`, then
`/engineer The car behind is closing.` and another engineer reminder before a driver
turn such as `I know what I'm doing.`. Recent history is bounded to six turns.
`/reset` clears it; `/history off` clears and disables it; `/quit` exits.
History support is an experiment interface, not proof the model reasons correctly.

## Reproduce the run

The optional runtime needs the project's installed package, CPU PyTorch,
Transformers and Hugging Face Hub/safetensors. Those are already present here.
The ordinary project `.venv` is for helper tests and dataset inspection, not model
execution. Setup downloads about 470 MB of pinned safe tensor/tokenizer files.
Training refuses drafts without an explicit flag and refuses an existing output
directory. Pick a fresh output name rather than overwriting the first result.

```powershell
.\data\stt-prototype\runtime\Scripts\python.exe scripts\setup_driver_interpreter.py
.\data\stt-prototype\runtime\Scripts\python.exe scripts\train_driver_interpreter.py --allow-draft-labels --output data\driver-interpreter-pilot\run-seed17-repeat
.\data\stt-prototype\runtime\Scripts\python.exe scripts\try_driver_interpreter.py --model-dir data\driver-interpreter-pilot\run-seed17-repeat
```

Base revision: `e8f8c211226b894fcb81acc59f3b34ba3efd5f42`. Run artifacts are under
`data/driver-interpreter-pilot/run-20261011-seed17/`: `report.json`,
`split-manifest.json`, `model.json`, and separate frozen/fine-tuned/selected
safetensors adapters. The pinned base is under `data/driver-interpreter-pilot/base/`.
All models and run artifacts are ignored by Git. No unsafe pickle checkpoint,
live configuration change, automatic annotation approval or model promotion occurs.

## Next, separately

### Owner playground feedback, 2026-10-11

The owner supplied these English terminal results with no dialogue history. These
are observed development failures, not a new independent accuracy benchmark or
approval of complete annotations. The proposed action corrections below are ours;
the owner identified the original interpretations as wrong. No automatic training,
threshold change, corpus rewrite or checkpoint promotion was performed.

| Supplied driver message | Observed prediction | Proposed action | Relevant scores |
| --- | --- | --- | --- |
| `such a dirty move` | `report_problem` | `vent` | report_problem 0.8271; vent 0.2746 |
| `how's the weather today` | No actions | `request_information` | request_information 0.2768; greeting 0.2984 |

Both outputs had `clarification_needed: false`, threshold 0.5 and no history.
Owner-reported inference durations were 8.606 ms and 7.971 ms respectively. These
numbers are supplied observations, not separately reproduced measurements.

- The first utterance expresses an opinion about perceived unfair driving. It
  does not describe a mechanical fault or establish that an incident occurred.
  A car-problem report should require an actual driver report of a car problem,
  not merely negative wording. Contrast examples should distinguish emotional
  complaints from concrete faults, and include mixed complaint-plus-question turns.
- The second is an information request. Under the corpus's race-radio assumption,
  track/session weather is the default working interpretation, but `today` alone
  does not specify current measurements versus a forecast. That distinction belongs
  to later request extraction; this pilot only predicts the conversational action.
  Missing weather telemetry would limit the answer, not turn the question into a
  greeting or erase its information-request action.
- Inspection of the original seed-17 split found `incident-vent` and
  `current-weather` entirely in **validation**, not training. The training set's
  12 vent-positive examples all belong to `gap-ahead-family` and combine venting
  with an information request. Weather has ten bilingual corpus examples but
  none from that weather family were fitted. This supports a narrow training-coverage
  explanation; it does not prove the model's internal reason for either prediction.
- Lowering the global threshold to 0.25 would still retain the wrong car-problem
  label on the first message, and would add a greeting on the second. Do not treat
  a threshold tweak or memorizing two phrases as sufficient repair.
- Keep these supplied phrases as already-exposed development cases. Add independent
  situations and natural English/Turkish wording before the next documented pilot,
  with positive train/validation/test coverage for each target where feasible.
  Group close paraphrases together; do not move existing validation families into
  training and then reuse their variants as a supposedly fresh test.

### Follow-up scope

The first targeted expansion is now drafted: [v2 coverage notes](driver-meaning-dataset.md#targeted-v2-expansion-2026-10-11)
describe 600 added rows across 60 scenarios, with positive action/clarification
coverage in all candidate partitions. It was subsequently used for a separately
authorized second run; the original checkpoint remains unchanged.

Review disputed labels and the new **scenario families**, especially clarification,
corrections, communication feedback and preferences. Reserve fresh English/Turkish real-world wording for an
independent evaluation. A future pilot can compare learning methods (including
SetFit), class balancing and dialogue representation on a new documented protocol.
Do not repeatedly tune against this now-inspected test. Full request extraction
and any production routing/fallback decisions need separate work and evaluation.

The training runner now accepts `--dataset`; its default remains the original v1
corpus to reproduce this first experiment. For a separately authorized v2 pilot,
select both the new corpus and a new run directory explicitly:

```powershell
.\data\stt-prototype\runtime\Scripts\python.exe scripts\train_driver_interpreter.py --dataset fixtures\driver-meaning\review-expanded.v2.json --allow-draft-labels --output data\driver-interpreter-pilot\run-v2-seed17
```

That command has now completed in the separately authorized second training task,
not during dataset expansion. It refuses to overwrite the existing run directory.
The playground continues to load `run-20261011-seed17` unless `--model-dir` is supplied.
Its answers and accuracy do not improve merely because the source corpus grew.

## Second pilot: v2, 2026-10-11

Completed on the owner's explicit request to train once more. New ignored artifacts:
`data/driver-interpreter-pilot/run-v2-seed17/`. The corpus is
`fixtures/driver-meaning/review-expanded.v2.json`, normalized SHA256
`8d669630af11653ed6d7ff81f8ef54141e4348c4b95469028b9a311a8833ec8f`.
The original selected adapter's file SHA256 was checked before and after and is
unchanged: `0C79E4FB6C4DE65AD6F2C2CE8ACBD275B5A8E8BF634C511FBBC775CDB91A2D81`.

Same pinned base and training settings as v1: CPU, eight threads, seed 17,
batch 16, 400 warm-head updates, 100 matched head-only/encoder-adaptation updates,
last two encoder layers adapted, identical learning rates and validation-only
checkpoint/threshold/candidate selection. This is a new fit from the pinned base,
**not continued training of v1**, and not SetFit. Because there are more training
rows but the same update budget, examples receive fewer average sampled exposures.

Split: **948 training / 216 validation / 236 test**, with language counts
474/108/118 per language. Every one of the 12 targets has positive support on each
side; no unsupported-positive label is omitted from v2 macro F1. All annotations
remain authored drafts and source splits remain unassigned. No fitting, threshold
change or candidate switch was performed after seeing these test results.

| Candidate | Validation exact | Test exact | Test micro F1 | All-12-target macro F1 |
| --- | ---: | ---: | ---: | ---: |
| Unadapted similarity baseline | Not used to select trained candidate | 6.78% | 0.2363 | 0.1686 |
| No-input training-majority baseline | Not used to select candidate | 12.71% | 0.1594 | 0.0242 |
| Frozen encoder + trained head | 51.39% | 96/236 (40.68%) | 0.5731 | 0.4288 |
| Last-two-layer adaptation + head | 54.63% | 89/236 (37.71%) | 0.5940 | 0.4395 |

Validation chose the adapted checkpoint at step 100, with threshold **0.7**.
The frozen control uses its own validation-selected threshold 0.6. The adapted
checkpoint is still the saved playground selection, despite its worse test exact
match: changing selection now would use test feedback. It has slightly higher
micro/macro F1 but **seven fewer fully correct messages** than the frozen control.
There is no clear overall fine-tuning win. Selected-model exact match is English
40/118 (33.90%) and Turkish 49/118 (41.53%).

Do **not** interpret the old 50% and new 37.71% as a controlled regression, or as
proof that adding data made the model worse. The grouped partitions and class
distributions differ. V1 tested eight positive targets with a 42.62% constant
majority baseline; v2 tests all 12 and its constant baseline is 12.71%. Both sets
are already-inspected synthetic development material, not fresh release acceptance
evidence. Two fixed candidates on the same v2 split can be compared directly;
cross-corpus percentages cannot establish a causal data-expansion effect.

### What is working and what is weak

Selected-model test F1: radio checks 0.6250, information requests 0.7308,
car-problem reports 0.7619, venting 0.7500 and greetings 1.0000. The last result is
only ten positives from a held-out group, not proof greetings always work.
Preference changes reach 0.4651, unclear messages 0.3784 and clarification 0.3125.
Corrections are 0.1818 and communication feedback is 0.0690. Acknowledgments and
out-of-scope requests have **zero true positives**. Thirty-six test messages produce
an empty prediction. Extra problem/information/vent labels also hurt exact match.
The full report preserves per-label TP/FP/FN and failure IDs.

Warm classification on 40 messages: **5.882 ms median / 8.956 ms p95** on this
workstation, including tokenization, encoder and head, excluding loading, STT,
Qwen, telemetry and TTS. Total recorded run duration is 40.57 seconds, excluding
Python startup. No corpus inputs were truncated. The latency samples differ from
v1; this is not evidence that model adaptation made the architecture faster.

### Previously reported development cases

These were checked after the fit and test evaluation, using the saved selection
without changing its threshold or weights. They are known development probes,
**not an independent additional test set**. Their exact messages are absent from
the authored corpus; closely related wording and meanings were already exposed.

| Supplied message | Earlier owner-reported output | Saved v2 output |
| --- | --- | --- |
| `such a dirty move` | `report_problem` | `vent`, no clarification |
| `how's the weather today` | No actions | `request_information`, no clarification |

For the dirty-move case, v2 vent score is 0.8778 and problem-report score is still
0.6312; the validation-selected 0.7 threshold retains only vent. For weather,
information-request score is 0.8551. These scores are uncalibrated. Matching these
two proposed corrections does not establish that their entire scenario family is
reliably repaired. The earlier reported and new measured timings are not a controlled
latency comparison.

The repeated-phrase history interface was also smoke-tested: `Can we make it?`
without history returns information request with clarification; after the driver
asks about fuel to finish and the engineer mentions fuel range, it returns the same
action **without** clarification. This is a known corpus/development context contrast,
not evidence of general dialogue reasoning. The classifier still does **not** output
fuel/catching topics, full question meaning, arithmetic or engineer replies.

### Try the v2 checkpoint

The existing command without `--model-dir` still uses v1. Select v2 explicitly:

```powershell
.\data\stt-prototype\runtime\Scripts\python.exe scripts\try_driver_interpreter.py --model-dir data\driver-interpreter-pilot\run-v2-seed17
```

At the time of that training run, no default model, live configuration, annotation
approvals or production routing changed. The subsequently authorized bounded live
trial is documented below. Before another large data expansion or any promotion, inspect the remaining
label confusions, family diversity and training-budget/sampling choices. New
experiments must select settings on validation or new development partitions, not
repeatedly optimize this now-observed test. Real-world independent evaluation and
the full request-understanding architecture remain separate requirements.

## Experimental live race trial (2026-10-11)

The owner requested testing the trained v2 pilot in an actual race. The repository's
`config/default.toml` now opts into `[conversation.driver_interpreter]` with the
preserved `run-v2-seed17` checkpoint. Bare `ConversationConfig()` remains disabled;
packaged `.exe` installations explicitly disable this source-only experiment. No
checkpoint was retrained, annotation approved, or model marked `promoted`.

The trained CPU worker loads and warms once before the microphone is armed, alongside
the existing model startup sequence. It reuses the optional STT Python environment;
the main application does not import Torch. It reads pinned local base/adapter files
only, has no network/downloads, and closes its owned hidden process when the engineer
stops or a replay exits. Cancellation and timeouts kill and reap the worker. If it
cannot load or an inference fails, Qwen planning takes over for the rest of that run.

Routing is deliberately limited. Only a singleton `radio_check`, `greeting`, or
`vent` prediction can produce an evidence-free social context plan. The action must
exceed both the checkpoint threshold and a declared 0.85 guard, with no information
score at/above 0.2 or clarification/unclear/correction/preference/feedback/out-of-scope
score at/above 0.35. All other active actions veto this route. Truncated model inputs
and unrecognized history formats also fall back. These guards are uncalibrated
experimental policy, **not a reliability guarantee**: missed mixed clauses remain a
race-test risk. They were not tuned against a release holdout.

Qwen still generates the actual response on the social route; there are no canned
replies and accusations are not incident evidence. Information requests, mixed
requests, problem reports and feedback retain the entire original Qwen planner ->
telemetry/capabilities -> Qwen Core/response -> refresh/grounding path. A fuel query
therefore normally needs two Qwen calls; a qualifying social turn normally needs one
(existing invalid-response repairs can add another request). The classifier
does not silently invent topics or change driver preferences. Existing STT, TTS,
automatic critical calls, session/generation checks and freshness checks are unchanged.

History uses the identical role-separated training format and the most recent six
entries. Full configured history still reaches Qwen on fallback. Oversized tokenized
inputs cannot take the social shortcut, avoiding a factual clause being discarded by
the encoder's 128-token truncation.

### Smoke measurements, not an end-to-end speed claim

The actual persistent v2 worker warmed in **6.42 seconds** on this machine. Five
sequential development probes had **8.77–9.63 ms** client round trips (tokenization,
encoder/head and local IPC), with **7.82–9.19 ms** worker inference. These exclude
STT, Qwen generation, TTS, game load and startup. `such a dirty move` took the social
route; fuel and mixed vent/fuel probes retained Qwen planning. `Are you there?` and
`Beni duyuyor musun?` still produced weak/uncertain predictions and fell back. Do not
expect every radio check, or every race question, to become faster. This is known
development evidence, not fresh independent accuracy evaluation.

### Run the race test

From the repository, start the existing panel and engineer normally:

```powershell
.\start-panel.cmd
```

Wait for radio ready (startup includes the one-time classifier warmup). Try naturally
worded English/Turkish radio checks, greetings, venting, fuel/position questions, and
mixed messages such as `That was dirty. How much fuel do we have?`. Record approximate
button-release-to-first-audio delay and whether the response addressed the whole turn.
Do not change thresholds or retrain during this comparison. For an A/B check, keep
the same machine/runtime settings and similar prompts, disable the section's `enabled`
flag, then **stop/restart the engineer** and repeat. Game CPU load, cache state, prompt
length and recent dialogue can affect timings, so compare several turns rather than
one fastest reply.

The existing `logs/control-panel.jsonl` receives `driver_interpreter_ready` (startup),
`radio_driver_actions` (uncalibrated scores/inference time), `radio_interpreter_route`
(`social_fast_path` or `qwen_fallback`, with reason), and a `driver_interpreter` stage.
Existing context/core/model, ASR and TTS timing events remain correlated by radio trace.
Classification labels/scores contain no transcript; existing opt-in radio-text privacy
settings still govern text logging. Inspect the latest run, not accumulated runs:

```powershell
Get-Content logs\control-panel.jsonl -Tail 500 |
    Select-String 'driver_interpreter|radio_driver_actions|radio_interpreter_route|radio_stage_finished|radio_turn_finished'
```

Rollback is `[conversation.driver_interpreter] enabled = false` in the config followed
by an engineer restart. No model files or existing checkpoints need deletion. Next
full-interpreter work remains telemetry-request extraction/domain review and independent
evaluation; this race trial only measures a bounded social shortcut's benefit and errors.
