# Driver-meaning dataset: expanded review draft

Status: **1,400 assistant-authored examples, 700 English and 700 Turkish**. There are
157 scenarios and 100 provisional leakage groups. All full-row reviews remain `draft` and
all splits are `unassigned`; partial owner feedback is recorded below. The dataset
tools themselves perform no training, model selection, downloads or live changes.
This is an expanded annotation draft, not an accuracy result or a validated training set.

On 2026-10-11 the owner authorized an exploratory training attempt using draft
labels. The [isolated action-classifier pilot](driver-interpreter-training-pilot.md)
is now completed; its split manifest and checkpoints are separate ignored artifacts.
Reviews and corpus splits were not changed, and the live interpreter was not replaced.

Active corpus: [review-expanded.v2.json](../fixtures/driver-meaning/review-expanded.v2.json).
Preserved first-pilot corpus: [review-expanded.v1.json](../fixtures/driver-meaning/review-expanded.v1.json).
Original 42-example seed: [review-draft.v1.json](../fixtures/driver-meaning/review-draft.v1.json).
Authored expansion source: [expansion-catalog.v1.json](../fixtures/driver-meaning/expansion-catalog.v1.json).
Coverage expansion source: [coverage-catalog.v2.json](../fixtures/driver-meaning/coverage-catalog.v2.json).
The seed and 800-row corpus are preserved unchanged; v2 contains the full 800-row
parent followed by 600 new drafts. A separately authorized v2 training run is now
completed; the original checkpoint is preserved. See the [second-pilot results](driver-interpreter-training-pilot.md#second-pilot-v2-2026-10-11).
Validator: `src/race_engineer/evaluation/driver_dataset.py`.

## Targeted v2 expansion, 2026-10-11

The owner authorized expanding the weak categories after the first training pilot
and two failed playground phrases. Sixty authored scenarios add five phrasings in
each language, not template-generated prefix/suffix combinations. Related variants
and contextual contrasts are kept together; close additions reuse existing group
IDs rather than inventing independence. The 60 scenarios add **41 provisional groups**.

| Action | v1 examples | v2 examples |
| --- | ---: | ---: |
| Vent | 34 | 114 |
| Car-problem report | 62 | 132 |
| Correction | 22 | 82 |
| Preference change | 42 | 102 |
| Communication feedback | 32 | 92 |
| Unclear | 20 | 80 |
| Radio check | 22 | 62 |
| Acknowledgment | 12 | 52 |
| Greeting | 10 | 50 |
| Out of scope | 12 | 52 |
| Information request | 588 | 708 |

Actions overlap. Information requests now occur in 50.6% of rows rather than 73.5%;
this is better coverage balance, not a claim that every action is equally sampled.
The added contexts distinguish pure and mixed vents, concrete symptoms, past-reply
feedback versus lasting style preferences, corrections to targets/units/windows,
weather questions versus weather greetings/complaints/holiday requests, connection
checks despite prior weather discussion, and ambiguous fragments versus understood
but unsupported questions. Full requests and meanings remain annotated, not learned
by the action-only classifier. Wind/rain/pit-service annotations do not implement
new telemetry capabilities.

The v2 read-only seed-17 split check finds **948 train / 216 validation / 236 test**
rows, with equal English/Turkish counts in every partition and positive support
for all 11 actions plus clarification. Training contains 50 vent-positive rows,
40 of them pure vents in four groups; the v1 pilot had 12 vent-positive training
rows and none pure. This check writes no assignments to the corpus and fits no model.
Positive support does not guarantee naturalness, label correctness or good model
generalization. Family grouping is provisional and still needs human semantic review.
The class distributions differ across partitions and from the first pilot; results
from a later v2 run would not be an apples-to-apples score comparison with v1.

The combined v2 corpus has 228 dialogue examples, 80 clarification-needed examples
and 28 illustrative synthetic-ASR examples. Its provenance records 284 exposed
development rows and 1,116 newly authored rows; neither count describes fresh real
speech or a locked holdout. Every new annotation remains draft. Exact user-supplied
failure phrases stay in the pilot notes rather than being falsely marked as newly
assistant-authored gold. Their related added families are marked previously probed.

## Review in PowerShell

Use the ordinary project `.venv`; no model runtime, Qwen, or iRacing is needed.

```powershell
.\.venv\Scripts\python.exe scripts\inspect_driver_dataset.py
.\.venv\Scripts\python.exe scripts\inspect_driver_dataset.py --language en --list
.\.venv\Scripts\python.exe scripts\inspect_driver_dataset.py --language tr --list
.\.venv\Scripts\python.exe scripts\inspect_driver_dataset.py --example dm11-tr
.\.venv\Scripts\python.exe scripts\inspect_driver_dataset.py --coverage
.\.venv\Scripts\python.exe scripts\inspect_driver_dataset.py --pilot-split-check
.\.venv\Scripts\python.exe scripts\inspect_driver_dataset.py --language tr --action communication_feedback --list --limit 10
.\.venv\Scripts\python.exe scripts\inspect_driver_dataset.py --language en --focus fuel_quantity --list --limit 20
.\.venv\Scripts\python.exe scripts\inspect_driver_dataset.py --language tr --list --offset 100 --limit 100
.\.venv\Scripts\python.exe scripts\inspect_driver_dataset.py --tag synthetic-asr --list --limit 20
```

The inspector defaults to **v2**. Use `--dataset fixtures\driver-meaning\review-expanded.v1.json`
to inspect the preserved first-pilot corpus. The default output is a compact
content-free inventory with a normalized corpus hash.
`--coverage` includes detailed action, focus, time-scope, and tag counts. `--list`
and `--example` deliberately display authored text and annotations in your console.
The inspector writes no reports, modifies no reviews, and calls no models. It validates
the entire corpus before filtering. Structural validation is **not** human approval.
Its contract is review-only: it rejects train/test assignments. Even reviewed rows
would not make this corpus automatically training-ready. Filters (`--language`,
`--action`, `--focus`, `--tag`, `--group`) select rows for display; the summary always
describes the whole corpus. Listed rows default to at most 100; `--offset`/`--limit`
paginate after filtering. `--example` selects an ID before pagination. Use
`--dataset fixtures\driver-meaning\review-draft.v1.json` to inspect the preserved seed.
`--pilot-split-check` adds a content-free, draft-label group-coverage feasibility
check with `--split-seed 17` by default. It does not import a model runtime, train,
approve annotations, save a split or make the corpus automatically training-ready.

## Preserved v1 expansion inventory and limitations

| Coverage | Examples |
| --- | ---: |
| English / Turkish | 400 / 400 |
| More than one racing-information request | 72 |
| Supplied dialogue history | 78 |
| Clarification needed | 30 |
| Mixed current/historical request scopes | 10 |
| Mixed English/Turkish wording | 10 |
| Illustrative synthetic ASR noise | 18 |

The categories overlap and are not empirical frequencies from real racing. There
are 41 open-vocabulary request focuses, including class/overall rankings, fuel
quantity/range/sufficiency, gaps and catching, pit strategy, tyre/brake/damage
assessment, historical speed statistics, race clocks, flags, weather, and penalties.
Greeting, acknowledgment, pure reports, frustration, temporary back-off, lasting
communication requests, ambiguity, unrelated tasks, and instruction-like text are
included without inventing telemetry requests for each message.

The expansion source has **76 authored families and 379 bilingual variant pairs**.
Most families have five manually written phrasings per language; one unrecoverable
noise family has four. A builder copies the family-level annotations to each variant
and materializes the JSON. It does not call an LLM, translate at runtime, enumerate
prefix/suffix combinations, train a model, or mark reviews complete. Individual
source spans for compound requests are explicitly authored. Shared canonical
annotations are an authoring convenience, **not independent human labels**: review
each variant for changed predicates, implied assumptions, natural Turkish, or a
meaning that needs a different annotation. Eight hundred rows do not mean eight
hundred independent situations. Punctuation/case/spacing-only full-input duplicates
are rejected; semantic duplicate detection still needs human judgment.

Action counts are not balanced: 588 rows include a racing-information request,
while smaller action categories need targeted review. Groups are provisional and
can be large; compound turns can connect related families across current group
boundaries. No split is assigned. Before training, review and merge connected
families as appropriate and ensure each proposed target has independently authored
training, validation, and test coverage. This expansion does **not** manufacture a
fresh locked test from variants of the same canonical question.

```powershell
.\.venv\Scripts\python.exe scripts\build_driver_dataset.py --check
.\.venv\Scripts\python.exe scripts\build_driver_dataset.py --edition v2 --check
```

These verify that each materialized corpus matches its parent and authored catalog.
The builder defaults to v1 for reproducibility; `--edition v2` selects the 800-row
parent, the new coverage catalog and the separate 1,400-row output.
Without flags, the builder validates and reports coverage only. `--write` creates
the absent expanded fixture or verifies an identical existing one; it refuses to
overwrite a different file. It never changes the seed. Source revisions and any
intentional refresh must be managed explicitly, so reviewed annotations are not
silently discarded by regeneration.

## What we annotate

Each row has an ID, scenario, leakage group, language, current driver utterance,
role-separated recent dialogue, origin, prior exposure, review status, and meaning.
The current speaker must be the driver; engineer turns appear only as context.
All text is newly authored for this file or adapted from our own already-discussed
failure phrases. No external radio transcripts, recordings, or private session logs
have been imported. This is not an independently collected speech corpus.

The annotation records:

- **Actions**, allowing more than one: radio check, information request, problem
  report, vent, acknowledgment, correction, communication preference, feedback about
  the engineer's current communication, greeting, out-of-scope, or unclear. These are
  provisional annotation categories, not allowed phrases.
- **Meaning**, a short English explanation of what the driver communicates.
- **Requests**, each with a source span in the current message, an open-vocabulary
  focus, the intended question, subject/reference basis, time scope, and resolution.
  Their questions are labels for meaning, not generated replies or executable queries.
- **Uncertainty**, separating unresolved meaning from missing telemetry. An
  understood pit question can require unavailable evidence without being linguistically
  ambiguous. An ambiguous social remark need not cause an intrusive clarification.
- **Rationale** and **review state**. Every row starts `draft`; approval/rejection
  needs an actual reviewer and timestamp. Never attribute our authoring to the user.

Canonical meaning explanations use English for both languages; Turkish input is
not translated at runtime. The Turkish variants use natural phrasing rather than
mandatory word-for-word translations. They share scenario/group IDs with related
English variants, and are not independent holdout examples.

`reference_basis` documents where the resolved subject/meaning comes from:
`explicit` includes an antecedent in this same utterance; `radio_convention` uses
the declared driver-to-own-engineer racing context; `dialogue` uses supplied turns;
`unresolved` must not name an opponent. Symbolic `self`, `car_ahead`, and `car_behind`
identify semantic roles, not actual car IDs. Multiclass standings, on-track versus
classification targets, and changing physical opponents need additional examples
before deployment. A historical `window_seconds` is an explicit requested duration,
not a measured telemetry value.

Racing `requests` exclude non-racing questions and preference commands. Those are
still preserved in the actions/meaning fields; an empty requests list does not
mean the driver said nothing. The response/action policy remains a separate task.

## First-batch review index

Each ID prefix has `-en` and `-tr` entries. These meanings are proposals to review.

| IDs | Scenario | Annotation distinction to check |
| --- | --- | --- |
| dm01 | Radio presence | No position or fuel request |
| dm02 | Am I last? | Preserve the last-place predicate |
| dm03 | Am I first? | Preserve leadership; not interchangeable with last |
| dm04 | One-word fuel | Broad fuel update, not an invented finish projection |
| dm05 | Gap ahead | Preserve opponent direction and time-gap meaning |
| dm06 | Incident vent | Opinion does not verify an incident or fault |
| dm07 | I know what I am doing | Social intent uncertain; no invented telemetry request |
| dm08 | Vent plus gap | Keep both actions, without conflating opponent references |
| dm09 | Fuel plus last | Two distinct questions, no required social action |
| dm10 | Can we make it, no context | Goal unresolved; clarification needed |
| dm11 | Same words, fuel dialogue | Fuel-to-finish goal resolved from history |
| dm12 | Same words, catching dialogue | Catching goal resolved instead of fuel |
| dm13 | Hypothetical pit rejoin | Clear question; calculation availability is separate |
| dm14 | Historical average speed | Preserve average and twenty-second window |
| dm15 | How is the race looking? | Broad assessment, not a canned rank reply |
| dm16 | Tyre report plus staying out | Driver-reported condition, future question, new focus |
| dm17 | Copy/thanks | Acknowledgment, no fresh retrieval |
| dm18 | Stop position announcements | Communication preference, no automatic profile write |
| dm19 | No, behind us | Correct the earlier target using dialogue |
| dm20 | Cake recipe | Understand it without inventing a racing topic |
| dm21 | I know what I am doing, after repeated reminders | Back off current unsolicited advice; not another gap request |

## Owner feedback, 2026-10-10

- The owner accepted the proposed core meanings of English `Fuel.` (broad current
  fuel update) and `That was dirty.` (venting without verified incident claims).
  This does not claim approval of every schema field or the Turkish variants.
- For `I know what I am doing.`, the owner explained a common intent: after repeated
  warnings about a closing car, it means "back off and let me race." Added `dm21-en`
  and an authored Turkish adaptation `dm21-tr`, with explicit repetitive engineer
  dialogue. Both remain drafts pending review of the complete new examples.
- `communication_feedback` captures this context-dependent feedback rather than
  forcing it into `vent`, a new telemetry request, or a durable preference command.
  Standalone `dm07` variants remain uncertain without the repeated-reminder context;
  they share a leakage group with `dm21` to preserve the contrast in future splits.
- Intended later response behavior: acknowledge briefly and reduce repeated routine
  advice. Do not argue or repeat the same gap report. Interpretation alone is not
  permission to disable critical calls, silence the radio indefinitely, or save a
  permanent profile setting. Scope and duration of any eventual back-off policy
  need separate design; no such runtime policy is implemented in this dataset task.

## Owner playground feedback, 2026-10-11

Two supplied English development failures are recorded in the
[pilot notes](driver-interpreter-training-pilot.md#owner-playground-feedback-2026-10-11):
`such a dirty move` was classified as a car-problem report rather than venting;
`how's the weather today` received no action rather than an information-request
action. These are proposed corrections to user-identified mistakes, not completed
full-row reviews. Relevant existing families were validation-only in the first
pilot; vent training coverage was limited to mixed vent-plus-gap turns.
Keep these exact supplied phrases marked as exposed development material in any
future expansion. No rows, review approvals or corpus splits were changed here.

## Annotation rules

1. Label the current driver turn. History resolves references; it does not cause us
   to answer an old question again. Retain supplied roles and temporal order.
2. Preserve questions, negation, predicates, direction, conditions, and windows.
   Topic-only labels would hide the original first/last bug.
3. Keep simultaneous actions and requests separate. Do not use one `mixed` bucket
   as a substitute for the components that made the turn mixed.
4. Do not hallucinate a missing referent. Do not treat all vague requests as fuel.
   Distinguish ambiguity essential to fulfilling a request from social uncertainty.
5. Driver reports and complaints are not verified race facts. No drafted response
   should claim "we saw it," actual rank, actual fuel, or opponent fault.
6. A capability need not exist today to annotate its intended meaning. The tyre
   example deliberately tests vocabulary beyond the previous six topics. This
   does not authorize implementing the capability or expanding production routing.
7. Mark a disputed label, explain alternatives, and revise it before approval.
   Human review should judge naturalness, interpretation, reference resolution,
   grouping, and whether a clarification is actually warranted.

For your review, comments such as `dm21-tr: make the reminders sound more natural`
or `dm04-en: fuel should include range by default` are enough to start the discussion.
Approval should be recorded only after that discussion, not inferred from silence.

## Splits and exposure

In the preserved v1 draft, 244 rows mark `previously_probed`: wording or closely related
failure scenarios have already been discussed or used in our earlier experiments.
556 rows mark `newly_authored`, not "unseen test." These exposure tags are authored
provenance notes, not proof of statistical independence. All rows are development/review
material once we inspect them. The original Laya/intelligence fixtures are unchanged.

Related language variants, identical phrases with different history, minimal
contrasts, and request components appearing in compound turns share conservative
leakage groups. Scenario pairs are structurally checked; cross-scenario grouping
still requires human semantic review. IDs alone cannot discover every paraphrase.

Do not make a random row-level train/test split. After review and substantial
expansion, assign whole groups, and keep any paraphrase or synthetic augmentation
with its parent. Keep full dialogue/session families together for future real data.
Validation and a final locked test must be distinct, with language-specific results.
This seed is deliberately **not** a locked evaluation set or a ready training set.

## After review

Review the added pure reports, greetings, unsupported racing requests, unresolved
opponents, corrections, negation, mixed scopes, distractor dialogue, code-switching,
and multiclass targets. These have drafts now, not validated coverage. Further
situations should follow actual observed failures rather than just a row-count goal.

ASR examples are **synthetic**, not collected from Qwen3-ASR or a microphone.
`input_kind: synthetic_asr` requires a separate `intended_text`. Mild recoverable
noise is distinct from eight intentionally unintelligible examples, whose correct
annotation is uncertainty/repetition, **not the hidden intended topic**. Never feed
`intended_text`, annotation, rationale, tags, IDs, or grouping metadata to an
interpreter as evidence. Train/evaluate on the observed text plus permitted dialogue
and declared scenario assumptions only. Real ASR captures and realistic noise
distributions remain future collection work.
Real logs need a separate explicit collection decision, redaction, and provenance.

Choose a first learnable target only after labels are stable. A multilabel classifier
may learn conversational actions/topic support; it will not automatically reproduce
the full natural-language meaning, predicates, or dialogue resolution in this schema.
Benchmark those capabilities separately, including abstention and CPU latency.
An explicitly opt-in draft-label pilot now tests conversational actions and
clarification only; it does not learn the full requests above. The dataset builder
and inspector still include no automatic training, approval or export. Production
model selection, final schema mapping, independent acceptance evaluation and live
integration remain pending; see the [pilot notes](driver-interpreter-training-pilot.md).
