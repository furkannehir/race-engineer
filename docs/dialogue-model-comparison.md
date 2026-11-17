# Local dialogue model comparison, 15 November 2026

This comparison investigates whether LFM3 is a good fit for Pitward's driver
conversation. It does not select a new live model or change the installed app.

## Method

The fixed development corpus contains 36 turns: 18 English and 18 Turkish.
It includes the driver's two reported conversations, independent paraphrases,
venting plus an information request, unsupported advice and an opponent-identity
question. Each sequence begins with empty history; its subsequent turns retain
the actual native user/assistant/tool messages. No translation stage is used.

All candidates use the same four functions, system instructions, synthetic
telemetry (P5, ahead gap 1 second, behind gap 1.6 seconds), 8192-token context,
single slot, 256-token request limit and b12088 llama.cpp runtime. A turn uses
one direct response or one tool selection followed by one response. There are
no model-driven repairs in this diagnostic. The original application remains
responsible for its separate delivery and freshness rules.

The benchmark preserves raw replies even when the application validator rejects
them. Otherwise an integration bug would be counted as a model failure and would
silently change subsequent history. English drafts are independently passed
through the existing trial grounder; Turkish is marked not applicable because
that grounder is English-only. App-owned limitation wording is recorded separately
and never overwrites the raw answer under review.

Exact tool matches are **interface conformance**, not answer accuracy. An honest
unsupported answer without calling `report_unavailable_information` misses the
prescribed tool but may still be semantically correct. Conversely, a correct
function call can be followed by an incorrect or irrelevant sentence. Reports
leave `automatic_speech_accuracy` null and preserve transcripts for human review.

Existing user processes were not stopped. A pre-existing installed CPU model
server remained after testing; its PID was outside all twelve benchmark runs.
Every benchmark-owned server closed. Background load was not controlled, so
these timings are not an exclusive-machine performance comparison.

The older Qwen weights run through this same native harness. This is not a rerun
of the original Context/Core v8 pipeline, whose historical results are documented
in [the LFM trial record](lfm3-trial.md#initial-results--8-november-2026).

LFM is tested with both the existing greedy settings and Liquid's recommended
general preset (temperature 0.1, top-k 50, repetition penalty 1.1). Qwen3.5 uses
its documented non-thinking text preset, plus an explicit greedy control after
the first preset run showed rambling and factual errors. Ministral uses temperature 0.1. Seeds
are 7 and 8 for the two GPU repetitions. Different sampling presets mean the
comparison measures usable configurations, not model weights alone.

GPU runs use full offload and FP16 K/V cache for all models. CPU and GPU runs are
sequential, with an owned server per candidate and explicit shutdown. The short
CPU comparison repeats the six-turn English gap conversation and its Turkish
counterpart once; it is not a full CPU quality benchmark. Cold startup is outside
per-turn latency. Each repetition's first turn is retained in the reported timings.

No speech worker, STT, TTS, microphone, audio playback, UI or game/VR load is
started. GPU allocations are read from server logs; periodic whole-device samples
also include other applications and may miss brief peaks. Neither is a guarantee
of application-wide peak VRAM during racing. Windows process peak working set
is recorded separately when available.

## Models and provenance

| Candidate | Weights | Revision |
| --- | --- | --- |
| LFM3-1.7B | Existing official Q4_K_M | `cd0c74c575b5c416f3011b5b4c298077f26ca8c2` |
| Qwen3-4B-Instruct-2507 | Existing Q4_K_M | `a06e946bb6b655725eafa393f4a9745d460374c9` |
| Qwen3.5-2B | Unsloth Q4_K_M, text only | `f6d5376be1edb4d416d56da11e5397a961aca8ae` |
| Ministral 3 3B Instruct 2512 | Official Q4_K_M, text only | `eb599d408350ea2bb60452cb86be7c7b2fc28227` |

Exact file sizes and SHA256 digests are pinned in
`scripts/setup_dialogue_comparison.py` and `scripts/compare_dialogue_models.py`.
Weights, runtime logs and raw transcripts stay under ignored `data/`.

Publisher references:

- [LFM model card and tool/history requirements](https://huggingface.co/LiquidAI/LFM3-1.7B)
- [Qwen3.5-2B non-thinking text settings](https://huggingface.co/Qwen/Qwen3.5-2B)
- [Ministral 3 3B model card](https://huggingface.co/mistralai/Ministral-3-3B-Instruct-2512)
- [Qwen3.5 quantization source](https://huggingface.co/unsloth/Qwen3.5-2B-GGUF)
- [Ministral official GGUF source](https://huggingface.co/mistralai/Ministral-3-3B-Instruct-2512-GGUF)

## Reproduce

Cached LFM/Qwen assets and the b12088 CPU/CUDA runtimes must be provisioned first.
Download the two additional verified weights:

```powershell
.\.venv\Scripts\python.exe scripts\setup_dialogue_comparison.py
```

Run models sequentially; choose a new output directory each time:

```powershell
.\.venv\Scripts\python.exe scripts\compare_dialogue_models.py --backend gpu --repeats 2 --output data\benchmarks\dialogue-comparison-gpu
.\.venv\Scripts\python.exe scripts\compare_dialogue_models.py --backend cpu --cpu-short --repeats 1 --output data\benchmarks\dialogue-comparison-cpu
```

Use repeated `--candidate` options to select a subset: `lfm`, `lfm-preset`,
`qwen4`, `qwen35`, `qwen35-greedy`, `ministral`. A busy port is rejected; the benchmark never
reuses an external server or silently switches backend. Reports include token
usage, server timing metadata, raw tool calls, raw speech, validation outcomes,
model hashes and a source/input fingerprint. A timeout stops that candidate.

## Interactive Ministral demo

Use the already provisioned comparison model in a terminal:

```powershell
.\.venv\Scripts\python.exe scripts\try_ministral.py --trace
```

It starts its own GPU server on port 18197 and reuses the verified cached weights.
No separately launched model server or iRacing session is needed. For an explicit
CPU trial, add `--backend cpu`. The default reply language is English; use
`--language tr` to start in Turkish.

Commands: `/clear` resets history; `/en` and `/tr` switch language and reset
history; `/help` lists commands; `/quit` exits. EOF and Ctrl+C also close the
owned server. `--question "Gap behind"` performs a single typed request.

The default fixture is now `fixtures/synthetic/ministral-lap-progress`: P5, ahead
1 second, behind 1.6 seconds, lap number 7, 6 completed laps, a 16-lap race with
10 laps remaining, and fuel 24 litres. Last lap is 92.5 seconds and session best
91.25 seconds. At 2 litres/lap there are 12 laps of fuel, a separate quantity.
The fixture retains
raw speed internally; conversational tools do not expose it.
It includes a 30-second history, observed burn of 2 litres/lap, explicit green
flags, known off-pit-lane state and nearby distances. These are synthetic inputs,
not newly implemented simulator channels or recorded race observations.
Use `--fixture recordings\your-recording` to seed the same tools from another
compatible normalized recording. Chat uses its final snapshot and retained
history; it does not start live iRacing or automatically advance the recording.
As of **16 November**, the default is the checked prototype described below.
Use `--raw` to reproduce the original unfiltered comparison behavior. Raw mode
shows rejected drafts with diagnostic validation; checked mode delivers accepted
text and keeps rejected drafts in `--trace`. Traces show readable Unicode,
operations, evidence scope and validation. There is no microphone or speech
playback. Restart an already running terminal demo to load the new code.

The interactive demo was smoke-tested with English GPU chat, Turkish CPU chat,
language switches, history reset and `/quit`; port 18197 was closed afterward.
Focused tests also cover EOF, Ctrl+C, startup failure, timeout cleanup and
readable Unicode traces. After the demo addition, **800 tests passed**, two
existing optional audio tests skipped, and Ruff/whitespace checks passed.

## Checked prototype follow-up, 16 November 2026

The shared response contract now permits repeating one evidence placeholder with
one unique binding. Thus `P5, fifth place` can use the same verified position
twice, while missing references, raw numerical claims and duplicate reference
definitions remain invalid. The native output parser also distinguishes a
driver-relative assertion from a car-relative sentence such as `You're followed
by a car that's 1.6 seconds behind you.` Turkish seconds suffixes are handled.
These changes fix demonstrated false positives; they are not a complete semantic
truth checker.

The new Ministral controller has a complete native operation menu. The model
chooses a current measurement, verified radio recall, free social speech, or an
unsupported-information report. Social wording is generated in the first tool
call's message argument, so it does not need an extra wording request. A factual
read or recall normally uses selection plus wording, on the same resident model.
The old forced-tool bug is not reinstated: greetings and venting have an explicit
social operation instead of being forced into one of the telemetry functions.
There is no Python driver-phrase router, second model, or separate context judge.

Exact numerical values live in a bounded verified-memory record: the last
successfully delivered position, ahead gap and behind gap. Ordinary working
history retains the driver's original transcript and the measurement topic,
rather than old values the model can copy into a new answer. Rejected drafts,
unspoken tool results and driver-supplied numbers cannot enter verified recall.
`/clear`, language switches, new sessions and generation changes clear this state.
This is transient conversation state, not persistent driver-profile learning.

`recall_radio_reading` retrieves the last delivered measurement or a named
quantity. It preserves the original observation sequence and marks it historical;
it is not refreshed from current telemetry or presented as a current value.
Current numerical replies require a current read and refresh before delivery.
Historical wording cannot claim that the old measurement is current. A bounded
repair may select a proper operation after a rejected social or historical draft;
at most two attempts/four model requests are allowed. Failed drafts stay out of
delivered history and are visible in traces.

### Actual model verification

The development replay changes live synthetic values between messages, including
changing the ahead gap to 1.9 seconds before asking whether the earlier call was
1 second. The next new reading is 2.4 seconds. Position changes P5/P4/P3/P6.
The final corpus also contains new paraphrases, venting with an information
request, and mechanical symptom reports.

The fixed final run used 42 turns twice: 31 English and 11 Turkish per repetition.
It delivered **82/84**, which is completion, not accuracy:

- English: **62/62 delivered**, **22/22 current-reading cases** with current
  observation provenance and **4/4 historical clarification cases** with the
  earlier delivered value. Typed median was **152 ms**, maximum **388 ms**.
- English venting: all four reported/development phrasings chose the social
  operation. Fresh venting paraphrases did so too. Mixed messages fetched the
  requested measurement. Mechanical symptoms reported missing diagnostic support.
- Turkish: **20/22 delivered**, **6/8 current-reading cases**, **0/2 historical
  clarification cases**. Presence checks could still become position readings,
  and wording remained awkward. The rejected clarification drafts confused
  temporal scope/direction. Turkish is experimental and fails acceptance.
- No live application or installed-model selection was changed. This remains a
  terminal prototype. Unasked follow-up questions, grammar, personality and some
  semantic mistakes remain; successful operation selection is not a complete
  natural-language correctness result.

This experiment also retained unsuccessful intermediate revisions. Stronger
instructions alone did not solve current-value reuse. The final raw control
used the same v3 development corpus; its prompt/operation set remained the
original comparison path. Do not compare its completion count to checked
completion as an accuracy score. The raw timing excludes its diagnostic
validator, while checked timing includes validation and bounded repairs.

Artifacts are under ignored `data/diagnostics/ministral-followups-20261116/`:
`raw-final.json`, `checked-final.json` (before the final false-positive correction),
and **`checked-reviewed.json`** (final two-repeat run). The final run recorded
unchanged source/input hashes and closed its owned server. No STT, TTS, playback
or real race was exercised. Earlier CPU/GPU comparison tables below describe
the original raw pipeline, not this checked controller's full audio latency.

The actual terminal command was also exercised with the driver's greeting,
clarification, eight repeated position questions, unsupported requests and venting
sequence: all 20 typed questions returned without a validation failure. A separate
CPU position smoke test completed in 6.91 seconds; this is not a CPU real-time
claim. Logs are `interactive-checked.log` and `cpu-smoke.log` alongside the reports.

Reproduce with a new output filename:

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_ministral_dialogue.py --dataset fixtures\intelligence-evaluation\ministral-followups.json --repeats 2 --output data\diagnostics\ministral-followup-check.json
```

Add `--language en` to focus on English, or `--raw` for the unfiltered control.
Validation after this slice: **829 tests passed**, two existing optional audio
tests skipped; Ruff, source mypy and whitespace checks passed. Coverage includes
changed live values, historical recall, rejected-draft isolation, state resets,
one-call social replies, repeated reference refresh and parser false positives.

### Driver review accepted as an English chat baseline, 16 November

The driver accepted the current English terminal prototype for continued work,
with creativity and occasional context misses deferred. The eight-turn trace
completed without a rejected reply, at 108–214 ms per typed turn. Both position
phrasings selected a current position read; catching and fuel-sufficiency requests
correctly reported the prototype's missing capabilities. This acceptance applies
to the chat prototype; live model selection remains unchanged.

Two reported failures remain open for future evaluation:

- `What's that guy doing!!` selected unsupported-information reporting. In this
  frustrated exchange it should acknowledge the rhetorical complaint without
  claiming to have witnessed the incident.
- `what the hell, give me simple answers please` selected current position and
  answered P5. This is communication feedback and should receive a brief
  acknowledgment, with shorter wording in subsequent replies.

The unavailable-information operation currently uses application-owned limitation
wording. Its repetition is partly a presentation constraint; changing model
sampling alone will not make those replies varied. Natural limitation wording,
communication feedback and question-shaped venting remain follow-up work.

A subsequent nine-turn driver trace reproduced the same context family through
historical recall: `what's this guy doing!!` selected `recall_radio_reading` for
position and repeated the prior P5. The past fact was valid but irrelevant, so
numeric/temporal validation accepted it. `ohh what the hell give me simple
answers please` selected the social operation this time, but its reassurance
did not explicitly acknowledge the style request. That is context-dependent
routing/wording, not a resolved communication-feedback feature. Pace and lap
questions correctly selected unsupported-information reporting for this limited
demo. All nine turns returned, at 113–329 ms; completion is not semantic accuracy.

The review also exposed a diagnostic aliasing bug: compacting a completed tool
result for working history mutated the same dictionary held in the trace. Tool
traces now retain an independent snapshot of the actual available reading, while
working history retains its compact historical metadata. This changes diagnostic
logging, not model input or delivered wording. A focused regression verifies both
representations stay independent.

### Planned English live trial, 18 November 2026

The driver agreed on 17 November to take this accepted English chat baseline
into a trial the following day. No race, microphone capture or audio trial was
started today. The checked Ministral controller is still terminal-only; an
opt-in live integration through the existing telemetry, PTT, STT, coordinated
speech playback and shutdown paths is required before the race trial. This
agreement does not promote Ministral to the default or update the installed app.

The trial should cover changing position and gaps, repeated fresh requests,
historical clarification, social/venting messages and unavailable questions.
Record STT, model, TTS and end-to-end latency separately, plus GPU memory with
iRacing/VR running, rejected replies and cancellation/shutdown outcomes. Typed
chat timings are not estimates of the complete spoken response time. Turkish
acceptance and expanded telemetry remain separate work.

## Existing telemetry and core exposure, 17 November 2026

The checked prototype now exposes the existing core and normalized data through
17 read/conversation operations. There is no new model, simulator write route,
lap-time adapter, pit-strategy calculation or change to the live/installed model.
The complete [access matrix](telemetry-expansion.md#implementation-and-ministral-access-snapshot)
lists the nine implemented calculations and 18 exposed current signal paths.

The new read-only bridge uses `DeterministicRaceCapabilities` and
`BoundedTelemetryMemory`, including raw current reads, current-field aggregates
and supported numeric windows up to 120 seconds. The model selects one operation;
a raw read can select up to eight signals. First/last checks have focused tools
returning the core's boolean verdict. Fuel range also returns its known inputs,
so missing burn can produce known fuel plus an explicit unavailable range.
Gap movement retains the core's qualitative direction; numerical rates are hidden.
Catch results retain estimate scope. Instantaneous speed is not exposed, and
generic history cannot retrieve it or numerical gap-window changes. Per-lap gap
comparisons and lap pace remain unavailable until a completed-lap calculation exists.
Lap number is not lap time, and reported-car count is not starting-grid size.

Wording receives only the current user message and actual selected result;
selection retains bounded dialogue context. The owned local model/server is
reused across questions. No second interpreter/judge model or driver-phrase
keyword router is added. Supported tokens or verified number/unit wording bind
to the existing grounder, which refreshes values before delivery. Unit changes,
invented values and ambiguous equal-valued metrics are rejected. A changing
boolean triggers bounded rewording against the refreshed result, rather than
changing `No` to `Yes` while retaining a contradictory old sentence.

Verified recall now covers delivered new readings/calculations too. Only facts
actually spoken are retained. A recalled measurement fragment receives explicit
past framing; it cannot acquire current scope. Clear, generation/session changes
and language-switch resets retain the existing isolation behavior. Each turn
still has at most one selected operation per attempt and one bounded repair,
with at most four model calls. Routine new factual reads normally use two calls.

The expanded workload is a **development** corpus, not a held-out acceptance
test. It covers all new feature families, changing fuel/lap/rank/pit/flag values,
historical fuel recall, missing timing/strategy/diagnosis, unsupported car-control
requests and the earlier repeated-position/gap sequence. The harness changes
frames only for explicit updates and rebuilds context with the existing builder;
it does not fabricate a flat gap history while reading an unchanged snapshot.
Reports fingerprint the dataset, fixture, bridge, controller, core calculations
and context builder, record rejected drafts, and close their owned server.

Run the current expanded English replay with a fresh output name:

```powershell
.\.venv\Scripts\python.exe scripts\try_ministral.py --trace
.\.venv\Scripts\python.exe scripts\evaluate_ministral_dialogue.py --language en --repeats 2 --output data\diagnostics\ministral-telemetry-check.json
```

Try fuel level/range, current lap number, first/last, recent position
changes, catching/closing, flags, pit-lane state and metre distances. Last/best
lap time, lap pace, numerical per-lap gains/losses, speed, weather, damage,
pit projections and simulator writes must stay
unavailable. `--raw` retains the original small comparison menu, so it does not
expose these expanded checked tools. `--backend cpu` remains an explicit runtime
option; GPU chat timing does not establish CPU or full-audio performance.

Earlier attempts are retained under ignored
`data/diagnostics/ministral-telemetry-20261117/`. R1 exposed valid numeral replies
being rejected; R2 exposed ordinal/number formats, history contamination and
unbound window duration. R3 still had incorrect qualitative sign/closing wording.
R4 rejected a wrong P16 ordinal for P6 rather than speaking it; R5 rejected an
unframed historical fuel fragment. These results were used to fix the bridge and
development validation; completion must not be described as semantic accuracy.

Earlier expanded English replay (`expanded-final-en.json`, development-v3, two
repetitions): **94/94 replies delivered**, median **272.686 ms**, maximum
**513.973 ms**. Eighteen turns used one model call and 76 used two. The report
confirms unchanged inputs and that its owned server closed. These GPU timings
include typed dialogue/tool validation but exclude model startup, STT, TTS,
playback and iRacing/VR load. The full test suite passes **888 tests**, with two
existing Chatterbox tests skipped because NumPy is unavailable; Ruff and mypy
also pass. No audio, microphone or live-race acceptance was run for this slice.

That review caught loss of the core's field descriptions: the last occupied field
rank had been described as the driver's previous rank. Tool results now retain
the existing output descriptions and source fields; the final replay identifies
P5 as the player and P6 as the last reported field position. This is separate
from the remaining wording defect: one behind-trend reply correctly said
`falling back, not catching` but then contradicted it with `The gap is closing`.
Numeric/state references alone do not prove the surrounding prose correct.
Keep that defect open before live promotion; 94 delivered replies is not 94
semantically correct answers.

Subsequent driver feedback rejected instantaneous speed and per-second gap rates
as useful radio answers. The checked tools now omit speed entirely and expose
only qualitative gap movement, with numerical rates retained inside core math.
Derivative queries and non-latest gap-history queries cannot reveal those rates
or manufacture per-lap results. Rendered-speech validation also rejects speed
units and gap rates per second/lap, including a model appending a denominator to
a valid current-gap reference. Genuine per-lap gains/losses remain unavailable;
they require verified lap-boundary history, not a unit conversion. Development-v4
adds pace, per-lap comparison and excluded-speed requests to the workload.

The final measurement-policy replay (`driver-units-final-en.json`, development-v4,
two English repetitions) delivered **96/100 replies**, median **237.864 ms**,
maximum **623.009 ms**, with unchanged inputs and its owned server closed. All
speed, speed-history, pace and numerical per-lap requests reported unavailable;
closing answers were qualitative. No delivered response used excluded speed or
gap-rate units. The initial policy replay had substituted position movement for
speed history; the final bounded repair rejects that generated claim and selects
the unsupported-information route. This is output validation, not driver-text routing.

Four broader failures remain, twice each: a valid bare `You are last` lacked the
required evidence reference and was rejected, while an opponent-identity request
selected position/gap data and proposed an unsupported per-lap claim, also
rejected. These are open validation/tool-selection issues; the 96 deliveries do
not establish general accuracy. After the policy changes, **907 tests pass**,
with the same two NumPy-dependent Chatterbox skips; Ruff and mypy pass. These
tests/replays include no audio or live racing, and do not validate Turkish.

Turkish remains experimental: an R3 presence check selected unavailable data and
a lap-time request failed with an incomplete response. Those language issues are
not resolved by exposing more telemetry.

## Lap timing, race progress and fuel, 18 November 2026

The checked demo adds last/best lap durations, last-versus-best comparison,
current/completed/total/remaining race counts and session clock. Five new typed
core capabilities execute the math in Python. The menu now has 26 operations,
14 implemented core capabilities and 26 normalized signal paths; the adapter
selects 25 SDK variables. Full details, null/sentinel handling and exclusions
are in the [access matrix](telemetry-expansion.md#implementation-and-ministral-access-snapshot)
and [core notes](race-capabilities.md#lap-timing-and-race-progress-18-november-2026).

Fuel endurance is application-formatted as "laps of fuel". Fuel-to-finish is a
separate conservative scheduled-distance estimate with one reserve lap. Positive
shortage/surplus are supplied by core; the model need not change a negative margin
into a different number. A range-only reply cannot claim sufficiency; bounded
repair can select the finish calculation. Timed-only races cannot produce exact
remaining laps or a supported fuel-to-finish estimate. Repeated last/best samples
cannot be averaged as distinct completed laps. Per-second gap rates remain hidden;
seconds per lap are allowed only for actual supplied lap durations, not gaps.

The first expanded replay delivered 127/140 English replies. Manual review found
the overview's current-lap field being used as remaining laps and the signed
fuel margin needing an absolute-value shortfall. Focused remaining/completed/total/
time operations now return only their requested field, and core supplies positive
shortage/surplus. Verified decimal-word binding accepts "one point six" only when
it matches the actual known gap; "one and six" remains rejected.

The next frozen replay (`data/diagnostics/ministral-laps-20261118/focused-en.json`)
delivered **135/140 replies**, median **348.302 ms**, maximum **1926.331 ms**.
It uses the cached GPU model, two repetitions of 70 English development turns,
and reports unchanged inputs and owned-server shutdown. This is a development
workload, not held-out accuracy; timing excludes startup, audio and race load.
Remaining rejections: two pit-rejoin questions, two opponent-identity questions
and one fuel-situation turn. Source/core validation covers missing timing,
sentinels, timed races, reserve boundaries and fresh fuel verdict repair. The final
suite passes **942 tests**, with two existing Chatterbox tests skipped because
NumPy is unavailable. Ruff and mypy pass; no actual speech or live-race test was run.

Human-review limits remain visible: one sufficient-fuel reply says "including"
where the reserve is additional; a low-fuel reply correctly gives the shortfall
but awkwardly describes the reserve as fuel available. A timed-lap follow-up
repeats the clock instead of clearly explaining the missing count. The incomplete
"are catching" still defaults to the car ahead rather than clarifying. Those
surrounding-wording/selection issues are not proven correct by numeric grounding.
The new data tools are ready for driver chat testing; no live Ministral promotion,
speech test or Turkish acceptance is implied.

```powershell
.\.venv\Scripts\python.exe scripts\try_ministral.py
.\.venv\Scripts\python.exe scripts\evaluate_ministral_dialogue.py --language en --repeats 2 --output data\diagnostics\ministral-laps-check.json
```

Try "What's my last lap?", "What's my best lap?", "How far off my best was that?",
"How many laps left in the race?", "How many laps of fuel?", and "Is it enough?".
The original compressed fixture is retained; the new fixture is still synthetic,
not a physically faithful lap trajectory or recorded iRacing session. Existing
`--fixture` recordings need a new capture to contain the newly selected fields.

Radio lap-time presentation now renders `92.5` as "one thirty two point five".
This is application formatting of verified last/best durations, including recall,
not model arithmetic. Raw seconds with units, a matching spoken radio time, or a
verified placeholder all bind to the same evidence; unverified time words remain
rejected. Minute rollover and millisecond precision are retained. Gaps, session
clock and last-versus-best differences are unaffected. See the
[formatting contract](race-capabilities.md#lap-timing-and-race-progress-18-november-2026).

The focused local GPU formatting smoke test delivered eight replies over two
repetitions: last lap was "one thirty two point five", best was "one thirty one
point two five", and the difference remained "1.25 seconds". Completion is not
semantic accuracy: after best lap had been discussed, the model twice selected
that reading when asked to recall last lap. This topic-selection issue remains
deferred; explicit-tool tests verify correct historical value retention and
fresh-value formatting independently. Results are under ignored
`data/diagnostics/ministral-lap-speech-20261118/`. The full suite passes 965 tests,
with the same two NumPy-dependent audio skips; Ruff and mypy pass. This smoke
test checks text through the local model, without audio playback or live racing.

## Results

Four distinct Q4_K_M models were tested in six configurations: **432 GPU turns**
(two repetitions of all 36 turns/configuration), plus **72 CPU turns** (one
12-turn subset/configuration). The two greedy controls were retained alongside
the recommended presets; unsuccessful configurations were not discarded.

### Tool selection and GPU memory

Each language has ten supported numerical requests per repetition. The following
counts require selecting the correct position/ahead/behind function and completing
the response. They exclude unsupported and social requests, so a correct direct
limitation is not unfairly treated as a failed numerical lookup. They do **not**
establish correct wording, correct direction in speech or general answer accuracy.

| Configuration | English reads /20 | Turkish reads /20 | Allocated GPU buffers |
| --- | ---: | ---: | ---: |
| LFM, existing greedy | 15 | 8 | 1.19 GiB |
| LFM, recommended preset | 14 | 3 | 1.19 GiB |
| Previous Qwen4B, greedy | 18 | 12 | 3.53 GiB |
| Qwen3.5-2B, recommended preset | 19 | 13 | 1.35 GiB |
| Qwen3.5-2B, greedy control | 18 | 8 | 1.35 GiB |
| Ministral 3 3B, temperature 0.1 | 18 | 18 | 2.89 GiB |

Buffers include weights, FP16 K/V, recurrent state where applicable, and compute.
The older allocation helper omits recurrent state from its numeric subtotal;
the consolidated analysis includes the logged 0.61 MiB for LFM and 19.27 MiB for
Qwen3.5. These are not entire-process/device peaks. Context size and FP16 cache
precision are fixed controls, not optimized deployment settings.

### Typed response latency

GPU numbers use all 36 attempts/language/configuration. CPU numbers use only six
attempts/language/configuration. Application validation and all speech stages are
outside these measurements; failures remain in the latency distributions.

| Configuration | GPU EN median / p95 | GPU TR median / p95 | CPU EN median | CPU TR median |
| --- | ---: | ---: | ---: | ---: |
| LFM, greedy | 247 / 315 ms | 247 / 528 ms | 3.31 s | 4.97 s |
| LFM, recommended | 242 / 321 ms | 252 / 558 ms | 3.58 s | 4.56 s |
| Previous Qwen4B | 161 / 224 ms | 174 / 252 ms | 2.49 s | 2.27 s |
| Qwen3.5-2B, recommended | 255 / 319 ms | 284 / 376 ms | 2.62 s | 1.91 s |
| Qwen3.5-2B, greedy | 168 / 218 ms | 162 / 225 ms | 1.53 s | 1.18 s |
| Ministral 3 3B | 106 / 148 ms | 99 / 160 ms | 1.78 s | 1.42 s |

LFM's English replies averaged about 27 words with greedy decoding and 30 with
its recommended preset, compared with about 10 for the previous Qwen. Extra
generation is one plausible contributor to latency; this does not isolate model
architecture or guarantee the same ordering under game load. The short CPU run
is a feasibility sample, not evidence of reliable real-time speech performance.

### Human review: none is accepted for live use

First-repetition transcripts were inspected separately from the counters above.
Representative findings are retained rather than replaced with canned replies:

- **LFM:** recommended settings did not solve relevance. A presence check invented
  P4 without a tool; an overview became a gap reading; a mixed vent/gap reply
  claimed the trailing car was ahead. Turkish frequently printed JSON-looking
  pseudo-calls instead of actually calling a tool and produced malformed text.
- **Previous Qwen4B:** English was generally concise and it correctly explained
  missing identity/trend/advice support. However, the repeated behind-gap request
  reused historical speech without a new lookup. A fresh Turkish position question
  invented P12 without a tool. The native interface is not equivalent to its old
  v8 bilingual pipeline, so this does not establish that every old Qwen path fails.
- **Qwen3.5 recommended preset:** its 19/20 English selection count concealed poor
  speech: invented competitor numbers, unsupported leadership/pace conclusions,
  and incorrect gap wording. It was not selected merely for having the highest
  lookup score. Turkish reversed ahead/behind selections and invented numbers.
- **Qwen3.5 greedy:** much shorter English, no English grounder rejections in this
  run, and a memory footprint close to LFM. It still selected ahead for the driver
  following, answered identity with distance, and inferred race quality/leadership.
  Turkish still reversed directions and invented P12/15-second measurements.
  Zero English rejections therefore does not establish semantic correctness.
- **Ministral:** fastest measured configuration and strongest Turkish supported
  tool selection in this screen. It produced concise useful gap readings, but
  guessed P6 for the opponent's identity without evidence, reused an old gap,
  called the race competitive without support, and substituted a current interval
  for a closing-trend answer. Some Turkish wording reversed the relationship
  (`Arkadaşın 1,6 saniye önde`) despite a correct behind-gap tool result. Tyre and
  social wording also need work. Its 2.89 GiB allocation is materially above LFM.

The validator is independently imperfect. It rejects valid repeated P5/fifth
wording and the correct sentence `You're followed by a car that's 1.6 seconds
behind you.` Conversely, it accepts qualitative claims such as a competitive
race without evidence. These defects were left unchanged during the experiment;
changing models is not a substitute for fixing them.

**Decision:** this screen supports reopening the LFM choice, not automatically
promoting a replacement. Ministral is the strongest next bilingual trial candidate
if its memory cost is acceptable. Greedy Qwen3.5 is the low-memory English
alternative worth further investigation. Neither passes the complete request,
uncertainty, freshness and speech-direction requirements yet. The live default
and the installed application remain unchanged.

### Artifacts and verification

The consolidated measurements are in ignored
`data/benchmarks/dialogue-model-comparison-20261115-summary.json`.
Individual reports, native replies, token counts and runtime logs are in:

- `data/benchmarks/dialogue-comparison-20261115-cached-{gpu,cpu}-r1/`
- `data/benchmarks/dialogue-comparison-20261115-qwen35-{gpu,cpu}-r1/`
- `data/benchmarks/dialogue-comparison-20261115-qwen35-greedy-{gpu,cpu}-r1/`
- `data/benchmarks/dialogue-comparison-20261115-ministral-{gpu,cpu}-r1/`

All owned servers closed, all model hashes were verified and each batch's tracked
source/input fingerprint stayed unchanged during that batch. The harness was
extended between batches to add the explicit greedy control and broader source
fingerprinting; the raw conversation/tool execution logic and dataset were
unchanged. This is development screening, not a blinded held-out comparison.

Validation: **795 tests passed**, two existing NumPy-dependent audio tests skipped;
Ruff, source mypy and whitespace checks passed. Focused coverage verifies that
application rejections preserve raw history, unsupported canonical replies cannot
hide bad raw advice, Turkish avoids English validation, timeout stops further
requests and closes the owned server, and trial settings never rewrite defaults.
