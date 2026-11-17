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

The [15 November model comparison](dialogue-model-comparison.md) reopens this
selection: LFM is not leading the tested configurations on request selection or
typed latency. No replacement has been accepted or selected for the live app.

## Installed-session social regression, 15 November 2026

The installed session at 15:34-15:37 Istanbul time used LFM on CPU. It repeatedly
added a radio check to unrelated social replies, echoed an identity question and
treated a complaint about dirty driving as literal dirt. The original social
prompt included an unconditional instruction to confirm a radio check. Core also
received no dialogue for social follow-ups. This was reproduced with typed input.

Native social generation now receives the original question, requested language
and number-redacted delivered dialogue, without the empty telemetry inventory.
The existing model-selected social kind supplies one short wording hint instead
of presenting instructions for every social purpose together. The current question
remains authoritative. Hints guide free generation; they are not canned answers
or a driver-phrase router. Native tool descriptions also distinguish identity,
neutral comments, acknowledgment and venting. One resident LFM model still performs
selection and wording. Factual evidence bindings and delivery behavior are unchanged.

Social wording requests eight words as guidance, not a validity requirement. The
initial twelve-word rejection introduced in this slice was removed after it rejected
valid requested jokes. The existing sixty-word grounded radio limit, schema/evidence
validation and single repair attempt for invalid drafts remain. Replies are never
chopped mid-sentence. Factual evidence constraints remain unchanged.

The initial seed1 development set `fixtures/intelligence-evaluation/lfm-social-regression.json`
contained 16 questions: the reported inputs, paraphrases, two delivered-history
follow-ups and four telemetry/mixed controls. A CPU run against the installed
Q4_K_M weights and b12088 runtime produced:

| Check | Original instructions, one repeat | Revised instructions, two repeats |
| --- | ---: | ---: |
| Pipeline completed | 15/16 | 32/32 |
| Exact expected plans, including social kind | 6/16 | 22/32 |
| Evidence expectation matches | 15/16 | 32/32 |
| Core repairs | 1 | 0 |
| Typed median / p95 | 2.65 / 8.99 s | 2.58 / 5.43 s |

These are development measurements, not an accuracy claim or a controlled speedup
comparison. The original run used one repeat; the revised run used two. No STT,
TTS, playback, race/VR or Turkish was exercised. Source/input fingerprints remained
unchanged during each run. All 24 revised social replies were within twelve words.
These measurements preceded the removal of the fatal twelve-word style check.

Manual review found improved presence checks (`Yes, I'm here.`), identity
(`I'm Pitward, your race engineer.`), greetings and venting (`We've got this,
take it one lap at a time.`). Remaining failures are deliberately retained:
neutral comments get generic wording, the name follow-up can return only the role,
`Where are we running?` receives an unasked `No`, and a question about fuel laps
receives liters. Mixed turns can omit their social acknowledgment. Correct tool
selection and grounded values do not establish that the reply answers the question.
See INT-016 in [known issues](known-issues.md).

To repeat with provisioned source assets, close other engineers first:

```powershell
.\.venv\Scripts\python.exe scripts\benchmark_intelligence.py --config data\trials\lfm3-native-cpu.toml --candidate lfm3-native-v1 --dataset fixtures\intelligence-evaluation\lfm-social-regression.json --repeats 2 --require-owned-runtime --expect-backend cpu --show-replies --output data\benchmarks\lfm-social-review.json
```

Choose a new output filename for each run. `--show-replies` prints synthetic text
for human relevance review; the JSON report contains metrics rather than transcripts.
For free-form typed testing, use `scripts\try_lfm3.py --native-tools --config` with
the same profile. In this workspace an ignored CPU profile reusing installed assets
is `data/diagnostics/lfm-social-20261115/cpu.toml`; before/final reports and console
reviews are alongside it. Changes apply to source runs; the installed executable
requires a new build/update. This slice does not alter TTS deadlines or fix missing
normalized player-position data from the recorded race.

### Sequential driver replay and style-check correction

The driver's next eight-message terminal conversation was reproduced in order through
the real planner, Core, evidence refresh and grounder. Four probes were then repeated
with empty delivered history. The same bad overview, tyre-clarification and unrelated
position replies occurred with empty history too. Recorded raw tool calls, Core inputs
and drafts distinguish these causes:

- `Can you tell me a joke?` selected social conversation correctly. LFM generated a
  complete joke, then a complete repair, both over twelve words. Application validation
  discarded both solely for length. This was an integration defect introduced by the
  style ceiling. It is removed; brevity is guidance. Replaying the same sequence and
  fresh probes now completes **12/12**, versus **10/12** before removal. Each joke uses
  one Core call instead of a failed repair. These are completion counts, not accuracy.
- `What's that guy doing there!!` selected `get_current_race_position`. The unrelated
  position was already in LFM's raw reply template; grounding only inserted the value.
  No automatic application position fallback was responsible for this example.
- `Is this a good race so far?` retrieved only current classification. LFM's raw draft
  gave position and echoed the question. Core received the original question verbatim.
- `What tire should we use next?` selected `request_driver_clarification`. The application
  returned its generic clarification without a wording call. The requested unsupported
  recommendation was confused with an ambiguous request.
- `What compound should we use at the pitstop?` selected
  `project_stationary_pit_service_duration`. Its unavailable result became a generic
  prediction limitation. That sentence sounded plausible while concealing wrong selection.

The unresolved boundary is semantic relevance: a valid selected tool and a grounded
number do not prove that the answer addresses the driver. Future work must allow the
reply stage to reject irrelevant selected evidence and distinguish an unsupported but
clear request from one needing clarification, while retaining numerical validation.
Do not treat all of these failures as TTS, stale-history or an unseen second-model issue.
This experiment used one local CPU LFM server, with no microphone or speech playback.
Artifacts: `data/diagnostics/lfm-social-20261115/sequence-before.json` and
`sequence-after.json`, including raw selection/drafts and delivered-history snapshots.

### Relevance-review experiment: rejected, 15 November 2026

The proposed boundary change was tested in isolation against the current source.
Both runs used the same installed CPU Q4_K_M model, b12088 runtime, frozen sequence
203 fixture and ordered questions. Each run used one resident server for selection
and Core. The set contained the driver's eight-message conversation, four fresh-history
probes and four factual controls (first, last, gap behind and fuel laps).

The candidate added `answer`, `partial`, `social`, `unavailable` and `clarify` to
the existing reply pass. It could discard irrelevant selected facts; clear unavailable
and clarification decisions used application-owned wording. Factual replies still went
through the production placeholder materializer, refresh and grounder. It did not fetch
additional data or change the model. This was an experimental contract, not a live fix.

| Check | Current source | Combined review and wording |
| --- | ---: | ---: |
| Pipeline completed | 16/16 | 13/16 |
| Core requests, including repairs | 14 | 21 |
| Median all-turn typed latency | 4.28 s | 4.58 s |

Completion does not mean a correct answer. This single exploratory run is not a
controlled performance claim. The candidate failed the sequential overview and both
tyre-choice probes. Its initial position reply rambled about placeholders, social
replies pulled unrelated position/compound requests out of history, and first/last
controls were labeled social while speaking plain-text ordinal rank claims. Fuel
**laps** still received **liters**. Declaring selected facts irrelevant in the same
generation pass did not reliably prevent them from influencing the generated reply.

A smaller decision-only probe returned one disposition without writing speech. A
metadata-only pass was followed by a fairer pass with the same human-readable meanings,
values and claims available to Core. All 16 full-fact decisions completed, with median
latency **699 ms** per decision (not per complete answer). Clear tyre-choice and identity
questions still became `clarify`; straightforward rank/gap controls became `partial`.
The opponent remark changed between `partial` with history and `social` without it.
This probe is not accurate enough to gate live speech, and adding it would add an
inference call on top of the existing path. No end-to-end benefit was established.

**Decision:** do not promote either candidate. The temporary CLI flag and candidate
module were removed from application scripts. Current behavior, including the joke
style-check correction, remains. A future experiment can examine whether retaining the
original user turn and native tool-result exchange improves reply generation, rather
than adding more responsibilities to the constrained reply object. That is untested.

Ignored, reviewable artifacts under `data/diagnostics/lfm-social-20261115/`:

- `relevance-baseline.json` and `relevance-candidate.json`: selections, Core inputs,
  raw drafts, final text, failures and per-turn times;
- `relevance_candidate.py`: the rejected candidate implementation snapshot;
- `relevance-gate.json` and `relevance-gate-full.json`: decision-only probes; and
- `conversation_trace.py` and `relevance_gate_probe.py`: reproduction scripts.

To reproduce on this workspace, choose a new report name and run one at a time:

```powershell
.\.venv\Scripts\python.exe data\diagnostics\lfm-social-20261115\conversation_trace.py review-baseline-repeat
.\.venv\Scripts\python.exe data\diagnostics\lfm-social-20261115\conversation_trace.py review-candidate-repeat candidate
```

These use a separate owned CPU server on port 18193 and close it afterwards. They
exercise typed input and synthetic data, without microphone, STT, TTS, playback or
game/VR load. They do not alter the installed application or its settings.

The driver can also try the exact saved candidate interactively in this workspace:

```powershell
.\.venv\Scripts\python.exe data\diagnostics\lfm-social-20261115\interactive.py --variant relevance
```

Use `--variant current` to compare the existing application path. Run one at a time;
`/clear` resets dialogue and `/quit` closes the owned model server. Both modes use
the saved CPU profile on port 18192 and the same fixed synthetic fixture. Both were
smoke-tested with the real local model; reset and exit were checked. This is access
to the rejected relevance experiment, not the unimplemented native tool-result
handoff experiment, and it does not change the live application.

## Native tool-conversation trial, 15 November 2026

This is a new source-only trial, separate from the rejected relevance reviewer.
The ordinary control panel and `scripts/try_lfm3.py` keep their existing behavior.
Start the native conversation trial from the repository root:

```powershell
.\.venv\Scripts\python.exe scripts\try_lfm3_dialogue.py --config data\diagnostics\lfm-social-20261115\cpu.toml --trace
```

That ignored workspace profile reuses the installed CPU model/runtime. On another
checkout, pass a provisioned native LFM source profile instead, such as
`data/trials/lfm3-native-cpu.toml`. The server starts once and is reused. `/clear`
resets dialogue; `/quit` exits and closes the owned server. Close other trials on
the configured port first. `--question "Position?"` runs one prompt and exits;
`--trace` prints the selected calls, tool results, model drafts and validation reasons.
Input is typed English against synthetic sequence 203, with no STT, TTS or microphone.

The flow preserves actual roles in one conversation:

1. Original `user` message goes to LFM with optional native functions.
2. A requested function is returned as an `assistant.tool_calls` message.
3. Python validates the call before executing it. Current position uses the existing
   deterministic classification capability. Gap tools reuse the existing nearest-car
   capabilities. Unsupported questions use a named-quantity unavailable result. There
   are no social-purpose tools or forced social labels.
4. The actual `tool` result, paired by `tool_call_id`, is appended after that assistant
   message. LFM continues the SAME conversation with a normal text reply, without a
   `goal`/`speech` or relevance-decision JSON schema.
5. Python checks numerical claims against the observed position, binds supported rank
   forms to evidence, refreshes the reading and uses the existing grounder. Old history
   cannot validate a new numerical claim. Unsupported-only output is rendered as an
   application-owned limitation, so speculative tyre recommendations cannot be delivered.

Only one tool may execute per turn in this initial slice. There are normally one
model request for direct speech or two for a tool exchange, with one bounded repair
and at most three total requests. Unknown calls/arguments, unsupported numbers, control
markup, truncated output and failed validation are visible errors. Only completed turns
enter bounded history; it retains actual message roles and delivered text rather than
replacing previous numbers with synthetic redaction strings.

The transport explicitly requests no thinking and separate reasoning output. Initial
real probes nevertheless exposed residual `</think>` delimiters. It separates the final
answer after a single complete boundary; unfinished or ambiguous blocks fail. Current
tool-turn reasoning metadata is preserved for the next request and never spoken. This
follows the [pinned llama.cpp API](https://github.com/ggml-org/llama.cpp/blob/b12088/tools/server/README.md)
and [Liquid's tool conversation guidance](https://huggingface.co/LiquidAI/LFM3-1.7B#tool-calling).
Liquid also documents that disabling thinking can be inconsistent. This observation
does not establish that reasoning leakage caused the earlier JSON-path failures.

The initial two-tool CPU development check covered 13 inputs: five consecutive messages, then
fresh-context radio, position, tyre and role probes. **11/13 completed**. The initial
presence, explicit position and tyre-advice sequence completed; examples included
`Yes, I'm here`, a grounded P5 and an honest unavailable-data reply. Tyre advice is
still unsupported: its safe final wording is owned by Python, not generated strategy.

Remaining failures are retained: `Where am I?` in the sequence exhausted the bounded
native selection, and `Where are we running?` generated lap numbers for a position
result, which validation rejected. `Radio check` fetched and spoke an unrequested
position. A role question correctly identified the engineer as on the radio but added
position and unsolicited offers. Completion is not semantic acceptance. The tool menu
is much smaller than the live catalog, so this is not a controlled comparison proving
that conversation format alone improves LFM. No real-race, speech or VR acceptance is
implied, and the live application has not been switched.

Artifacts: `data/diagnostics/lfm-social-20261115/native-dialogue-check.json` and
`native-dialogue-check-final.log`. Tests cover native role/call-ID continuity, reasoning
separation, free radio replies, numeric and stale-history rejection, delivery refresh,
unknown tools/arguments, unsupported recommendations and shutdown after failure.

### Driver gap feedback: retry bug fixed, semantic acceptance still failing

The driver's follow-up sequence was reproduced exactly: presence, `Where am i`,
`How's the race going`, `Gap behind`, `Gap ahead`, and `What's the gap behind?`.
It exposed a concrete integration defect. After rejecting a number copied from history,
the trial changed the next call to `tool_choice="required"` and instructed LFM to fetch
position. That forced a function call even when no offered function answered the
original question. The race-overview and gap questions could therefore become tyre
requests, whose generic application-owned limitation concealed the wrong selection.

The repair now preserves `auto` when no tool was selected, keeps the original user
message and asks for a relevant answer or an honest limitation without stale numbers.
It never forces an unrelated function merely to validate a prior number. Traces now
include the effective `tool_choice`. Regression tests preserve the optional-tool repair.

The initial menu also omitted the existing gap capabilities. The current trial offers
`get_current_race_position`, `get_gap_ahead`, `get_gap_behind` and
`report_unavailable_information(quantity)`. The tyre-specific dummy tool is removed;
the generic unavailable tool preserves the requested quantity. No telemetry calculation
or live-app routing was replaced. Seed2 of the development corpus adds the driver's
three gap phrasings, with expected direction-specific capabilities.

Gap results carry seconds, a readable verified statement and the driver's relationship
to the other car. Number binding validates gap units separately from rank, refreshes
the selected gap before delivery and rejects stale rank substitutions. Explicit
driver-relative gap statements are checked for the demonstrated direction inversion.
This is bounded factual validation, not a complete natural-language truth checker.

The paired CPU replay used the six sequential inputs plus six independent probes.
The original two-tool path completed 12/12 but returned numerous irrelevant replies.
An intermediate four-tool run completed 9/12. Inspection of the failed drafts exposed
additional integration errors: valid `1.6-second cushion`, `a second behind`, and a
reference to `the next one` were rejected by the number parser. Those failures were
not evidence of a bad model interpretation. Number binding now accepts those forms
while preserving exact-value/unit checks. Focused checks also reject the observed
reversed edge claims, unsupported overall leadership and `holding your ground`
assertions based on a single rank snapshot. Other qualitative inferences remain
outside these bounded checks.

The final current-read replay completed **11/12**, median **2.49 seconds** per typed
CPU turn; the rejected turn took **6.62 seconds** including repair. All three fresh
gap probes retrieved the correct 1.6-second behind gap or 1-second ahead gap. However,
semantic acceptance still fails:

- In the six-turn conversation, `Gap ahead` selected **get_gap_behind**, returning
  the wrong quantity with a correct number for that wrong selection.
- `How's the race going` became a gap reading or position instead of acknowledging
  that this trial cannot assess overall race progress.
- The rejected `Gap behind` turn invented leadership and combined unrelated facts
  during repair. Completed replies also included unverified pace wording and repeated
  unasked follow-up questions.

These are real-model observations, not unit-test accuracy. A same-frame fact-reuse
experiment did not resolve relevance and was removed; each new numerical answer
still requires a current tool read. The trial remains unaccepted for live use.

A separate request-level thinking experiment requested a 64-token reasoning budget
and the existing 256-token total limit. It completed **3/12**, median 9.71 seconds per
turn, with most gap/overview turns returning incomplete output under that limit.
The default remains off.
This rejects that bounded configuration, not every possible thinking setup or token
budget. No new speech or game-load acceptance was performed.

Artifacts under `data/diagnostics/lfm-social-20261115/`:
`gap-driver-before.json`, `gap-driver-release-check.json`,
`gap-driver-parser-fixed.json` (includes the discarded fact-reuse experiment),
`gap-driver-current-read.json` (final code), `gap-driver-thinking64.json`,
`driver-gap-summary.json` and `driver_gap_repro.py`. The earlier `after`, `final`
and `verified` files retain intermediate gap-wording probes. All inference was typed
English with synthetic telemetry; no microphone, audio or game-load validation was
performed. The live application has not adopted any of these trial changes.

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
