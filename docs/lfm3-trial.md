# LFM3 English MVP and trial record

LFM3-1.7B Q4_K_M is the selected conversational model for the current English MVP.
`config/default.toml` now selects the native LFM planner and CUDA b12088 runtime;
the usual `start-panel.cmd` loads it without an extra config argument. One resident
model handles tool selection and reply wording. Qwen3-ASR remains the separate CPU
speech recognizer. This selection does not establish general semantic acceptance:
lap-time requests, invalid tool arguments and follow-up interpretation need work.
Turkish conversation/training remain deferred. No training was performed.

On a fresh source checkout, install the model/runtime assets without creating
comparison profiles:

```powershell
.\.venv\Scripts\python.exe scripts\setup_lfm3_trial.py --backend gpu --runtime-only
```

Already installed assets are verified and reused. Use `--backend both` to provision
an explicit CPU option too. Speech setup is separate. The default TTS configuration
is unchanged; the VR test's Chatterbox V3 configuration remains available through
`--config data\trials\lfm3-native-gpu.toml` in this workspace.

Qwen weights/runtimes are retained for explicit rollback and legacy bilingual
experiments: `start-panel.cmd --config config/qwen-legacy.toml`. Legacy Qwen setup
and cache benchmarks default to this rollback file. Existing executables must be
rebuilt to adopt LFM; component set 3 points to the matching model/CPU runtime.

The diagnostic history below records earlier checkpoints before this adoption.

## Installed candidate

- Model: `LiquidAI/LFM3-1.7B-GGUF`, official `LFM3-1.7B-Q4_K_M.gguf` (1,099,627,808 bytes).
- Model revision: `cd0c74c575b5c416f3011b5b4c298077f26ca8c2`.
- Model SHA256: `3793e1e6e2266919d6e015287221b41100175a98eddaba92e111aed3b05c93c4`.
- Runtime: separate llama.cpp `b12088`, source commit `d1044bc82712fbf08996080bef6298b24a27e9c7`.
- Windows CPU and CUDA 13.4 x64 archives are pinned by size and SHA256 in setup.
- One model/server is reused for Context and Core. There is no Qwen fallback.
- Thinking is explicitly disabled through the chat template, context is 8192, parallelism 1.
- Assets and generated profiles/reports stay in ignored `data/`; existing Qwen is retained.

Liquid's [model card](https://huggingface.co/LiquidAI/LFM3-1.7B) lists 16 languages,
excluding Turkish, and describes a custom LFM Open License v1.0. Its
[llama.cpp guide](https://docs.liquid.ai/deployment/on-device/llama-cpp) requires
b11960 or later for LFM3. The existing b10964 Qwen runtime was left intact.

## Native adapter: setup and terminal trial

Already provisioned in this workspace. To reproduce from the same V3 speech profile:

```powershell
.\.venv\Scripts\python.exe scripts\setup_lfm3_trial.py --native-tools --config data\trials\chatterbox-v3-live.toml
```

On a new checkout, use an available source config instead. Setup defaults to
`config/default.toml` and preserves its speech/policy settings. Downloads and
extraction are verified; existing differing files/profiles are not overwritten.
No driver, system toolkit or Python environment is replaced.

Start interactive English chat against synthetic telemetry (server starts/stops automatically):

```powershell
.\.venv\Scripts\python.exe scripts\try_lfm3.py --native-tools
```

Type freely; `/clear` resets delivered dialogue and `/quit` exits. This trial uses
English explicitly, independent of saved driver reply-language preferences. It
does not record microphone audio, speak replies or alter driver memory. The
fixture remains fixed; no race is required and history contains only successful
responses. Failed questions are reported visibly.

A single question:

```powershell
.\.venv\Scripts\python.exe scripts\try_lfm3.py --native-tools --question "Am I last?"
```

Use `--config data\trials\lfm3-native-cpu.toml` for an explicit CPU run. The CUDA runtime
successfully detected the RTX 5080. Neither profile silently changes backend.

The native profiles and source default use `context_planner = "lfm3-native-v1"`
on port 8092. Run one at a time. They never borrow or terminate the original
custom-planner comparison server on port 8091.
No trained interpreter or Qwen model is loaded alongside LFM for this path.

The adapter performs two bounded inference stages through one resident model:

1. LFM uses llama.cpp's native `tools`/`tool_calls` interface to choose functions.
   `get_current_race_position`, `check_if_driver_is_first` and
   `check_if_driver_is_last` map to the existing `current_classification`
   capability; calculations stay in Python. Other functions cover the capability
   catalog, current raw signals, recent window statistics, current field
   aggregates, social acknowledgment, unavailable information and clarification.
2. After validating the whole selection, the existing Context Engineer executes
   the requested telemetry calculations. LFM submits a short `goal`/`speech`
   draft through the existing schema-constrained JSON API, using evidence
   placeholders. The native tool selector and the JSON reply client address the
   same configured model alias/port; there is no second set of LFM weights. The
   existing Core materializer, evidence refresh and strict grounder validate the
   result. There is no unvalidated free-text fallback. This adapts native calls
   to Pitward's Context/Core boundary rather than running an unrestricted agent.
   Core retains its existing limit of one repair attempt after a rejected draft.

The earlier reply-as-function experiment copied literal telemetry into speech
despite the advertised argument schema. The native reply path now sends a compact
root-object schema in `response_format`, with `inform` for factual speech and
`acknowledge` for social speech. Its speech pattern excludes digits, JSON string
delimiters, escapes and line breaks. Clarification and unavailable-data handling
retain their application-owned paths. This experimental reply interface does not
offer the full Core silence/goal selection available to Qwen. The pinned
[llama.cpp server documentation](https://github.com/ggml-org/llama.cpp/blob/b12088/tools/server/README.md)
describes this schema-constrained JSON API. Python still validates speech and
evidence references; decoding is not a substitute for grounding.
Rejection warnings include sanitized reason codes in terminal
text as well as structured fields, without model or driver text.
Native calls reject unoffered functions, invalid arguments, duplicate selections,
unknown signals and mixed temporal scopes before executing any selection.
Delivered dialogue is quoted context data for the native selector, not a replay of
its own assistant messages. Core receives the current factual question verbatim,
descriptive evidence placeholders and current facts, excluding earlier spoken
measurements. HTTP
requests run off the event loop, have bounded sizes/timeouts, close connections,
and cannot publish late metrics after cancellation. Existing server ownership and
shutdown apply. Nothing changes in speech, PTT, playback or driver preferences.

This version explicitly rejects Turkish turns. Requests combining current and
historical/future information still hit the existing single-scope plan boundary.
Grounding validates evidence use; it cannot prove that the selected evidence
answers the driver's actual question. English semantic acceptance remains open.

The earlier custom-planner profiles remain available on port 8091. Create them
with setup **without** `--native-tools`, then run `--compact-planner`. The
`qwen-context-v8` identifier in those TOMLs retains the existing schema,
telemetry execution and grounding policy. Actual weights are LFM3, as the model
field and server logs show. `--compact-planner` overrides only the planner's
system prompt with `lfm3-english-compact-v1`; the catalog, driver input, output
schema, token budget and application grounding are preserved. Core still uses
the existing prompt. Without that flag, the original v8 planner prompt is tested.
The compact override currently lives only in the trial command, not the panel.

### Live iRacing and VR shakedown

The existing native GPU profile can also launch the source control panel against
real telemetry. It selects LFM on CUDA, Qwen3-ASR on CPU and the speech settings
preserved when the profile was created. In this workspace those speech settings
are Chatterbox V3 on CUDA/BF16. The profile and required local component files
were checked before the combined race/VR measurement summarized in the roadmap.

Exit terminal trials with `/quit` and close any other running engineer before
starting. A separate LFM trial occupies the same port, and an unrelated Qwen
server would distort the memory comparison. Set the panel's reply language to
English for this English-only trial; retain the usual saved microphone, wheel
button and output device. Launch from the repository root:

```powershell
.\start-panel.cmd --config data\trials\lfm3-native-gpu.toml
```

The panel starts its own model. Click **Start** and wait for models and iRacing
connection readiness. The normal launcher without `--config` now selects LFM
with the default speech setup.

For a GPU trace, run this in a separate PowerShell before starting the engineer:

```powershell
$gpuTrace = "logs\lfm3-vr-gpu-$(Get-Date -Format 'yyyyMMdd-HHmmss').csv"
nvidia-smi --id=0 --query-gpu=timestamp,utilization.gpu,memory.used,memory.total --format=csv -l 1 > $gpuTrace
```

Stop the sampler with Ctrl+C after the run. These are whole-device, once-per-second
samples, including the game and other applications; they do not measure isolated
model memory, transient peaks or VR frame timing. Record the headset's frame timing
or reprojection view separately. Compare a similar driving scene with Pitward
stopped, then running, then generating speech.

Try ordinary questions, social remarks and follow-ups over several laps. Include
position after an overtake, first/last, gaps ahead/behind, fuel and a radio check.
Compare the first response with later ones, and note missed/incorrect answers or
stutters. Existing `logs/control-panel.jsonl` and rotated siblings correlate STT,
native selection, Core, queueing, speech preparation and playback outcomes. The
profile enables local transcript/final-reply logging without recording microphone
audio or full model prompts. Review these logs together with the GPU trace after
the run. The terminal benchmark's sub-second time excludes STT, TTS and playback;
this shakedown measures the complete radio path under game load.

### Preliminary native tool-format diagnostic

The driver's terminal test confirmed failures on greetings and current-position
questions. A subsequent isolated diagnostic tested short natural user messages
through LFM3's native function-calling API, without the custom planner JSON
contract. All twelve existing capabilities were advertised as functions; no
telemetry tool was executed and no generated reply passed application grounding.
The user's running trial server was left untouched; a separate owned server used
port 8092 and was closed afterwards.

Using existing capability names matched 7/9 expected tool selections. Descriptive
public names for two tools (`get_current_race_position` and
`get_recent_race_position_change`, mapped to the same internal capability IDs)
matched 9/9. An expanded development probe including twelve additional questions
matched **20/21**, with median native request latency **91 ms**. The remaining
mismatch selected pit-stop duration rather than total pit-loss projection.

This is evidence for trying a native tool adapter, not an accepted race engineer
or a comparison against the 50-turn pipeline score below. There was no history,
tool execution, evidence refresh, final-answer grounding, STT, TTS or VR load in
this probe. Social wording remained generic. The later adapter described above
executes and validates the full pipeline, with materially lower selection
accuracy on the larger suite. Tool aliases describe functions, not recognized
driver phrases; there is no driver-utterance keyword router.

Artifacts: `data/diagnostics/lfm3-native-20261108/report.json`,
`report-clear-names.json` and `report-expanded.json`.

## Initial results — 8 November 2026

Both comparisons used the same 50 authored English development questions,
including paraphrases, history, social/factual mixtures and unavailable data.
One shared V3 Chatterbox worker stayed resident; models ran sequentially. No STT,
game/VR or physical audio playback was exercised. The model/runtime versions
differ, so timing differences are not a pure model-size experiment.

| Candidate | Pipeline completed | Exact plans | Evidence expectation matches |
| --- | ---: | ---: | ---: |
| LFM3, unchanged v8 prompt | 0/50 | 0/50 | 0/50 |
| LFM3, compact English planner v1 | 8/50 | 6/50 | 7/50 |
| Qwen4B/v8 baseline, compact comparison run | 49/50 | 45/50 | 44/50 |

At this initial checkpoint, **LFM3 was not accepted for live race use**. It requested unrelated
capabilities, mismatches temporal scopes, or produces invalid/oversized plans.
Manual inspection also found an unsupported starting-grid reply among completed
turns. Completion alone is not factual correctness.

The eight completed compact-LFM turns had median typed latency 410 ms, versus
594 ms over 49 completed Qwen turns. These are different surviving subsets, not
a demonstrated like-for-like speedup. A fresh radio check completed in 421 ms.
Three raw CPU/CUDA probes reproduced the same main over-selection behavior; that
small check does not establish backend equivalence, but does not implicate CUDA
alone in these failures.

LFM3 startup logs allocated about 1041 MiB for weights, 103.5 MiB for KV cache,
0.61 MiB for recurrent state and 69.8 MiB for compute: **about 1.19 GiB** of
buffers versus Qwen's **3.53 GiB**, saving approximately **2.34 GiB**. These
figures exclude driver/context overhead and Chatterbox. Whole-device memory
samples corroborated a reduction but include unrelated applications and are not
isolated process peaks. This does not prove the complete application fits a
2–3 GiB peak budget.

Reports and authored reply review:

- `data/benchmarks/lfm3-initial-20261108/`
- `data/benchmarks/lfm3-compact-20261108/`
- `data/diagnostics/lfm3-trial-20261108/native-parity.json`

Re-run with a new output directory:

```powershell
.\.venv\Scripts\python.exe scripts\try_lfm3.py --benchmark --compact-planner --full-suite --with-tts --baseline data\trials\chatterbox-v3-live.toml
```

The command runs English only, preserves all goldens, and saves fixture replies
locally for manual review. A completed benchmark means measurements finished,
not that its semantic checks passed. `--with-tts` keeps speech resident for memory
comparison; it does not synthesize or play the generated replies.

## Native adapter results — 8 November 2026

The same 50 English development questions and unchanged goldens were used.
Chatterbox V3 stayed resident, with no synthesis/playback or game/VR load. These
are development measurements from one run per revision, not independent held-out
acceptance or a confidence interval for race accuracy.

| Path | Completed | Exact plans | Evidence matches | Completed typed median / p95 |
| --- | ---: | ---: | ---: | ---: |
| LFM native, full original Core prompt (r1) | 31/50 | 24/50 | 28/50 | 340 / 516 ms |
| LFM native, compact Core prompt (r2) | 44/50 | 26/50 | 30/50 | 296 / 504 ms |
| LFM native, neutral position example, function reply (r3) | 39/50 | 26/50 | 30/50 | 290 / 563 ms |
| LFM native selection, first constrained JSON reply | 49/50 | 26/50 | 30/50 | 261 / 404 ms |
| LFM native selection, conversational presentation | 48/50 | 26/50 | 30/50 | 247 / 429 ms |
| Qwen4B/v8 baseline, r1 comparison | 49/50 | 45/50 | 44/50 | 597 / 875 ms |

The r3 revision removed an example that encouraged answering neutral position
questions with "No". It reduced completions, which is recorded rather than
hidden by choosing the earlier run. All 11 final pipeline failures were Core
draft rejections; fast completion is not evidence of correct understanding.
The final fresh `Track position` smoke test also failed: LFM copied literal
telemetry numbers into speech instead of evidence placeholders, including on
the repair attempt. The application rejected it rather than speaking an
unrefreshable answer. This prompted the constrained JSON reply correction below.
All-turn final median/p95 were 339/554 ms, including rejected questions. Different
surviving subsets and runtime versions prevent claiming a like-for-like speedup.

Manual review still found position answers to some radio checks, wrong selections
for original-grid requests and unrelated data in mixed questions. A separate
repeated conversation also contaminated later social turns with earlier
position/fuel selections, with one truncated native request. At that checkpoint,
live promotion was deferred. The positive result was a working native
integration and lower memory, not English semantic acceptance.

### Reply-format correction after terminal feedback

A controlled comparison ran both reply interfaces against the same selected
evidence for the reported prompts. `position` and `where are we` selected the
correct classification capability. Their native-function replies copied literal
numbers and failed validation; JSON decoding produced valid placeholders from
the same evidence. The trial now uses native functions only for selecting tools,
and the existing constrained JSON client for reply drafting. Qwen defaults and
its 192-token Core budget remain unchanged. Native Core retains its previous
effective 320-token budget; no limit was raised to hide a failure.

The full unchanged English suite completed **49/50**, with zero Core repairs and
the same **26/50 exact plans, 30/50 evidence matches**. All-turn median/p95 were
268/439 ms; completed-turn median/p95 were 261/404 ms. The one rejected turn was
a truncated last-lap-duration draft. Completion is format success, not semantic
accuracy: this change does not fix incorrect tool selection.

Replaying the reported conversation now answered the first two position prompts,
but subsequent radio, overview and pit questions still selected position. A
separate `cache_prompt=false` probe did not resolve those history-dependent
errors, although the radio check after clearing history succeeded in that probe.
This does not establish a cache bug or a general solution. Production cache
settings were left unchanged. Broad race-overview support and robust current-turn
interpretation remained open; live promotion was still deferred at that checkpoint.

The user-owned server on 8092 was left running. Verification used a separate
owned profile on 8093 and closed its model/speech workers. To load changed Python
code, type `/quit` in an existing terminal trial, then rerun the same command.
There is no new model download or setup requirement for this correction.

### Gap and catch-time terminal feedback

A further reported sequence was reproduced after the constrained reply fix:
position and gap-ahead answers completed in hundreds of milliseconds, but two
`gap behind` turns generated 768 tokens of ordinary prose with no tool call and
were rejected before telemetry execution. The fixture's actual behind gap is
1.6 seconds; the model's discarded prose guessed 2 seconds. Increasing the token
budget or speaking a partial generation would not fix that provenance failure.

Explicit request-level reasoning-off controls reproduced the failure. This is
consistent with the [model card's caveat](https://huggingface.co/LiquidAI/LFM3-1.7B)
that disabling thinking is not always reliable, but the probe does not establish
its full root cause. Native client settings and the 768-token limit are unchanged.
`are we catching` also chose a future catch-time tool instead of current relative
pace: the fixture supports `closing` at 0.05 s/s, with a conditional 20-second
catch estimate. Broad overviews still select classification alone, and spoken
wording exposes internal labels. These remain open as [INT-015](known-issues.md#int-015-experimental-lfm-native-selector-can-ramble-instead-of-requesting-a-tool).

### Conversational presentation and limitation follow-ups

The LFM trial now receives a compact speech view: verified values, human-readable
meanings, units, confidence, time scope and the placeholders required for each
comparison. Internal evidence IDs and source paths remain in the application's
retained context. The original materializer, live refresh and grounder still bind
the generated placeholders to those original sources. No measurements are dropped
to make an answer sound smoother, and no command-phrase routing was added.

The speaking prompt favors short race-radio sentences, clear ahead/behind
direction and only the facts the question needs. Its gap example is bound to the
actual metric and unit. The model still generates supported speech, rather than
selecting a fixed answer for each driver phrase.

Unsupported-only turns previously skipped Core and repeated one generic sentence.
Native Core now makes one bounded 64-token choice between direct, explanatory and
acknowledgment framing using the current message and delivered dialogue. This is
the same resident LFM model, not another model or a separate interpretation pass.
`LimitationReference.wording` defaults to the original `standard` rendering; the
native trial uses `radio` or `explain`. The application owns the actual limitation
text and its unknown references. Invalid choices, extra claims or transport errors
produce a logged, safe direct fallback; cancellation still propagates. Ambiguous
requests still ask for clarification, and partial answers retain supported facts
while removing unbound health/condition claims. Sentence joining also cleans up
an orphaned leading "and" after an unsupported clause is removed.

A local synthetic conversation produced:

- Gap behind: "Car behind is 1.6 seconds back." (280 ms)
- Damage: "I don't have that data available here." (269 ms)
- Availability follow-up: "That information isn't available in the data I have here." (240 ms)
- Position: "You're P5; last place is P6." (551 ms)

These are observed examples, not promised fixed wording or speech/audio latency.
Qwen's prompts, call budgets and standard limitation wording remain unchanged.
No voice, PTT, driver memory, model weights or live-default selection changed.

At this wording checkpoint, the 50-case English development run completed 48 turns, with 26 exact plans
and 30 evidence matches. Three Core repair attempts were recorded; the two failed
turns were truncated Core replies for `temporal-grid-after-current-en` and
`social-after-classification-en`. Completed-turn median/p95 were 247/429 ms;
all-turn median/p95 were 248/551 ms. This measurement preceded the final small
sentence-joining cleanup, which is covered by the tests. The earlier constrained
reply revision completed 49/50, so this is a wording improvement rather than an
accuracy improvement. The unchanged 26/50 interpretation score and the known
gap/history/scope failures remained open at that checkpoint. The later MVP model
decision does not close those correctness issues.

Validation: 638 tests passed, two existing NumPy-dependent tests were skipped;
Ruff and source mypy passed. Coverage includes same-source live refresh after
the compact view, unknown-to-known refresh rejecting stale limitations, zero
values, unsupported health claims, malformed framing, safe fallback and
cancellation. All inference used synthetic typed input, with no real race,
microphone recording or speech playback. Restart an existing terminal trial to
load the new Python code, then run the same `--native-tools` command.

The measured model/runtime buffers remained about 1.19 GiB. Both benchmark-owned
model servers and the shared speech worker closed successfully. No microphone,
speech playback, race, CPU latency or Turkish inference was validated here.

### Current first/last regression correction

The next reported conversation failed after a position answer because Core copied
redacted measurements from delivered history. Redaction also removed the word
`first` from the current question. Native selection now receives past dialogue as
quoted data instead of prior assistant messages; factual Core gets the current
question verbatim and no earlier spoken measurements. Qwen's existing prompts and
redaction behavior are unchanged.

Three native operations distinguish numerical position, leadership and last place.
They execute the same classification capability, with native-only `is_first` and
`is_last` claims calculated by Python. For a single first/last check, the
application prefixes that claim to LFM's supporting position sentence. Both the
yes/no claim and position retain their source references and refresh before
speech; the model does not guess the Boolean answer. This uses semantic tool
selection, not a lookup of driver phrases. Descriptive placeholders reduce alias
confusion, and spelled-out cardinal measurements are also rejected if unbound.
The root reply schema described above replaced the earlier nested schema after
wire probes found extra fields in supposedly constrained replies.

Scalar-only native drafts now receive a label and unit from the selected evidence
metadata. Thus a bare gap value cannot become an unexplained `1.6`; ahead/behind
direction and the seconds unit are retained without changing the source binding.
Complete model-generated sentences keep their wording.

Actual local GPU verification on the final code:

- The exact six-turn reported sequence completed without a repair or rejected
  question, at 353–390 ms per typed turn. At synthetic P5, both repeated
  `are we last` checks and `are we first` answered "No" with the current position.
- Eight first/last checks across leader, middle, last and solo classifications
  passed, including contradictory old P28 dialogue (317–421 ms). Regression tests
  also cover changes into and out of first/last between preparation and speech.
- The unchanged 50-turn English development suite completed **50/50**, with
  **29 exact plans, 35 evidence expectation matches and zero Core repairs**.
  Typed median/p95 were **322/473 ms**, maximum 595 ms, with shared speech resident.
  This improves on the previous 48 completed/26 exact/30 evidence checkpoint;
  it still falls short of the earlier Qwen baseline's 45 exact/44 evidence matches.
- The earlier gap/damage sequence completed with a labelled 1.6-second behind gap.
  The short follow-up sequence selected explanatory limitation wording, while the
  longer final conversation repeated direct wording. Follow-up style and social
  phrasing still need broader acceptance; completion does not establish correctness.

Validation: **665 tests passed**, two existing NumPy-dependent tests skipped;
Ruff, source mypy and whitespace checks passed. No goldens or acceptance thresholds
were relaxed. The owned benchmark model and speech workers closed. These are
synthetic, typed English checks without microphone, audio playback, race or VR
load. No training, model download, cache-setting change or live-default switch
was made. Restart the terminal trial with `/quit`, then rerun the same command.

Artifacts stay outside Git:

- `data/benchmarks/lfm3-native-20261108-r1/` (includes Qwen baseline)
- `data/benchmarks/lfm3-native-20261108-r2/`
- `data/benchmarks/lfm3-native-20261108-r3/` (previous function-reply format)
- `data/benchmarks/lfm3-native-json-20261108-r1/` (first constrained replies)
- `data/benchmarks/lfm3-radio-wording-20261108-r3/` (conversational presentation)
- `data/benchmarks/lfm3-ranking-20261108-r3/` (current ranking/history fix)
- `data/diagnostics/lfm3-ranking-20261108/user-sequence-final.log`
- `data/diagnostics/lfm3-ranking-20261108/live-matrix.json`
- `data/diagnostics/lfm3-ranking-20261108/regression-final.log`
- `data/diagnostics/lfm3-radio-wording-20261108/conversation-r3.log`
- `data/diagnostics/lfm3-native-adapter-20261108/repeated-chat.log`
- `data/diagnostics/lfm3-native-adapter-20261108/draft-probe.log`
- `data/diagnostics/lfm3-native-adapter-20261108/reply-format-comparison.json`
- `data/diagnostics/lfm3-native-adapter-20261108/json-reply-chat.log`
- `data/diagnostics/lfm3-native-adapter-20261108/json-reply-no-cache.log`
- `data/diagnostics/lfm3-native-adapter-20261108/gap-turns-wire.jsonl`
- `data/diagnostics/lfm3-native-adapter-20261108/gap-turns-no-reasoning-wire.jsonl`

Reproduce with a fresh output directory (or omit `--output` for a timestamp):

```powershell
.\.venv\Scripts\python.exe scripts\try_lfm3.py --native-tools --benchmark --full-suite --with-tts --baseline data\trials\chatterbox-v3-live.toml
```

The baseline profile must use v8, the same TTS configuration and an available
owned server port. `--with-tts` retains its worker for memory comparison only.

## Follow-up gates

- [x] Isolated verified runtime/weights, explicit thinking switch, terminal chat and English benchmark.
- [x] Measure memory with shared speech resident; inspect English failures and CPU/CUDA probes.
- [x] Native tool-format diagnostic: 20/21 expected selections in a small, no-history probe.
- [x] Implement an experimental native LFM adapter with descriptive function names, retaining
  existing deterministic telemetry/capabilities and grounded reply boundaries. Re-run the full
  English pipeline, paraphrases and history; native selection alone is not an acceptance gate.
- [ ] Accept English understanding and factual replies against the existing baseline, then test
  repeated conversations and combined VR/radio latency before promoting a live model.
- [ ] After English acceptance, evaluate Turkish zero-shot separately. If inadequate, investigate
  Turkish language adaptation plus bilingual racing/tool/reply supervised examples. Task SFT/LoRA
  alone is not a guarantee of learning a new language; train from full-precision weights, then
  validate English retention, Turkish hold-outs and quantization before deployment.
