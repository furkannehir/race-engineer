# Known issues and deferred improvements

## STT-001: Short and single-word questions are sometimes misrecognized

**Status:** CE-05R.3 diagnostics and release-tail mitigation implemented; sample evaluation pending

**Reported:** 2026-09-21, driver feedback after the combined live test

Recognition sometimes fails, especially for one-word input. The driver accepts the current
prototype for now. This report does not yet identify whether capture boundaries, the energy
gate, language detection, or the recognizer is responsible; do not assume an engine defect.

The live path now plays an original opening radio-noise cue after the microphone stream is
ready and starts buffering only after that cue finishes. A closing noise cue confirms
release. This gives the driver an explicit speaking boundary and prevents the cue itself
entering the ASR clip.
It is a usability mitigation, not evidence that short-utterance recognition is fixed; the
diagnostic and engine comparison below remain required.

CE-05R.3 now retains a configurable 120 ms of microphone input after PTT release so the
last phoneme is less likely to be clipped. Every Qwen adapter submission emits content-free
`stt_audio_diagnostic` measurements, and `diagnose-wav` exposes the same gate/boundary data
for a deliberately supplied WAV without loading the model. Neither path stores audio or
transcripts. These changes locate likely capture/gate failures; they do not prove recognition
accuracy or choose a replacement engine.

Future work:

- run the new boundary/gate diagnostic and transcription on explicitly supplied or consented
  English/Turkish test samples; no background microphone recording;
- evaluate one-word questions such as "Position", "Fuel", "Sıra", and "Yakıt", along with
  longer paraphrases and realistic racing noise;
- assess capture timing, silence thresholds, language selection, accuracy, and latency
  before choosing tuning or an alternative local recognizer; and
- request a repeat when input cannot be reliably understood, rather than inventing a query.

Keep Qwen3-ASR as the current adapter. faster-whisper remains an evaluation alternative,
not a newly selected engine or automatic fallback. Natural phrasing must remain supported;
the test examples are not a restricted command vocabulary.

## TEL-001: Position changes can lag until the timing line

**Status:** CE-05R.1 implemented; live race validation pending

**Reported:** 2026-09-26, driver feedback after the CE-05/05.1 live test

The engineer could retain the previous position after a start or mid-lap overtake because
normalization used `PlayerCarPosition` for the player and `CarIdxPosition` for opponents.
Those scored position values can remain unchanged while live per-car track progress has
already changed.

CE-05R.1 now samples `CarIdxLap` and `CarIdxLapDistPct` in the same frozen SDK buffer and
derives a complete current overall running order during the racing state. The order updates
the player and opponent positions together, so conversation refresh, field-status answers,
position-change events, and strict-policy candidates see one consistent frame. The
official fields remain the fallback whenever the live progress set is incomplete or
invalid. Formation and non-race sessions retain their established behavior.

Regression tests prove that a pass changes P1 to P2 while both official inputs remain P1,
and that one missing opponent progress value disables the entire live derivation. Validate
this in a live race start, an ordinary overtake, a pit cycle, a lapped-car interaction, and
after a retirement before closing the issue.

## TTS-001: Spoken replies need a more natural voice

**Status:** Deferred improvement - accepted prototype limitation

**Reported:** 2026-09-21; follows earlier feedback about the Turkish voice

The voice feels unnatural/off to the driver, particularly in Turkish. Current Piper voices
are sufficient for the prototype; no voice or engine change is requested in this increment.

Future work: compare local English/Turkish voices or engines using racing vocabulary,
numbers, gaps, pronunciation, prosody, and short conversational acknowledgments. Include
driver listening feedback, synthesis latency, resource impact during iRacing, and engine/
voice distribution terms. Preserve replaceable adapters, language routing, cancellation,
and the live radio scheduler. Do not select on naturalness alone.

**Candidate review, 2026-09-24:**
[FreyaTTS-small](https://github.com/freyavoiceai/FreyaTTS) is one Turkish-specific option
for a later comparison, not a preferred or selected replacement. Its published model is a
183M-parameter, character-level Turkish synthesizer with a deterministic default voice,
48 kHz mono output, and Apache-2.0 code and weights. The authors report 8.0% WER / 3.0%
CER on their 495-sentence Turkish evaluation set, RTX 4090 RTF 0.10-0.11 with roughly
1.5 GB VRAM, and Apple M3 CPU RTF 0.70. Their seven-rater study favors Freya's naturalness,
although the top MOS confidence intervals overlap and Piper achieves lower WER. These are
upstream measurements, not Race Engineer or Windows/iRacing results.

Freya is Turkish-only and depends on PyTorch plus the VoxCPM2 AudioVAE. Its current local
model path still fetches that VAE through Hugging Face, and its dependency versions are
not pinned. Integration must therefore use separately pinned, checksum-verified local
model/VAE assets in an optional isolated worker with network access disabled. Keep Piper
for English and as the lightweight fallback; route by validated reply language behind the
existing `ConversationSpeaker` protocol. Do not add Freya/PyTorch to the default install.

Before promotion:

- audit and retain Freya, AudioVAE, package, dataset, and voice/model distribution terms;
- compare Piper and Freya with blind driver listening on short race calls, acknowledgments,
  Turkish characters, driver/track terms, positions, lap numbers, fuel, and decimal gaps;
- verify comma-decimal and unit normalization explicitly: Freya's current digit expansion
  handles integer/dot runs but does not define Race Engineer's Turkish `2,4 saniye` form;
- pin the canonical voice seed and treat failed/collapsed synthesis as an error or Piper
  fallback instead of silently switching to a different speaker seed;
- measure cold start, synthesis and release-to-first-audio p50/p95, real-time factor, peak
  RAM/VRAM, cancellation, and combined load with iRacing, ASR, and conversation inference;
- test Windows CPU and NVIDIA CUDA locally. Upstream publishes no Windows AMD GPU path, so
  retain CPU fallback and make no AMD acceleration claim without a separately validated
  runtime; and
- require fully offline startup after explicit setup, bounded worker failure, no hidden
  downloads, and no regression to radio priority, expiry, or freshness checks.

Upstream references: [model card](https://huggingface.co/freyavoice/Freya-TTS),
[technical report](https://arxiv.org/abs/2607.09530), and
[evaluation set](https://huggingface.co/datasets/freyavoice/freya-tr-eval).

## CONV-004: Add contextual race-engineer acknowledgments and reassurance

**Status:** R.5 natural bounded composition implemented behind the CE-05 preview switch;
listening/live validation pending

**Requested:** 2026-09-21

Conversation should respond naturally to remarks as well as factual questions. For
example, "That's dirty" could receive a concise acknowledgment or "Focus on your race
now", rather than being forced into a position/fuel query or a generic unsupported reply.
The requested feel is a supportive race engineer, not simply a spoken telemetry lookup.

Validation and later extensions:

- broaden independently authored English/Turkish paraphrase evidence without turning the
  examples into a fixed command vocabulary;
- listening-test whether the bounded acknowledgment/refocus templates feel natural and
  concise during a race, including repeated remarks and intentional silence;
- validate radio priority, expiry and interruption in live racing; critical calls must
  still take precedence and busy situations must not gain extra chatter;
- distinguish the driver's report from verified race evidence: "Copy. Focus on your race"
  is a possible neutral reply, while "Yes, we saw it" requires evidence the system can
  actually observe. Do not invent witnessed contact, assign blame, or imply steward review;
- keep factual race answers grounded and critical calls deterministic; personality must
  not become an unrestricted source of race facts or strategy; and
- add evaluation cases separating factual questions, emotional remarks, mixed inputs,
  and ambiguous comments, including when clarification or silence is preferable.

The bounded acknowledgment/close acts, mixed factual composition and delivery-aware state
are implemented in CE-05. CE-05R.5.1–R.5.4 add typed speech clauses, deterministic natural
English/Turkish variants, exact grounded-part coverage, a radio word limit and automatic
fallback to the earlier bounded wording. They remain behind the preview switch, which the
development configuration currently enables deliberately; fresh bilingual model evidence
and listening/live validation still gate default promotion. See
[CE-05 live dialogue](ce05-live-dialogue.md).

## CONV-005: Relational overall-field questions

**Status:** Implemented in CE-05.1; real-race classification validation pending

**Requested:** 2026-09-26

The engineer previously exposed only the player's numeric overall position, so a natural
question such as “Am I on the last place?” could at best produce “You're P23” and could not
ground the yes/no comparison. CE-05.1 adds one `field_status` meaning for English/Turkish
questions about being last, place out of the field, and how many classified cars are behind.

The answer remains deterministic. It requires `position` and `opponents` telemetry
capabilities plus a unique, gap-free set of positive overall positions after known pace-car
exclusion. It carries both current position and classified total, is refreshed immediately
before delivery, and reports unavailable rather than guessing when the order is incomplete
or contradictory. Class position and original-starting-field comparisons remain unsupported.

Scripted tests cover last/not-last, bilingual wording, missing/duplicate classifications,
contract validation, replay transitions and live refresh. A 2026-09-26 CPU smoke test with
the real local Qwen model passed three English/Turkish paraphrases and a compound field-status
plus gap question. A real iRacing race is still required to validate classification behavior
across gridding, disconnects, retirements and finish-state transitions.

## CONV-006: Field comparison can contradict the reported position

**Status:** Fixed in CE-05R.2; targeted local-model check passed, live validation pending

**Reported:** 2026-09-26, driver feedback after asking whether the car was first

The CE-05.1 semantic plan represented first/last/count questions with only `field_status`.
The deterministic renderer therefore assumed the original last-place meaning. If Qwen
mapped “Am I first?” to that query while the driver was P1, the result could be the logically
contradictory “No, you're P1.” The numeric telemetry was grounded; the missing requested
relationship caused the incorrect yes/no word.

CE-05R.2 adds a bounded field relationship to the semantic request and grounded answer:
first, last, cars ahead, cars behind, or position out of total. The model chooses only this
meaning. Ordinary code validates the complete field, computes the comparison/count and
renders matching English/Turkish wording. Relation and facts remain separate through
dialogue memory and pre-delivery refresh. Unknown combinations fail schema validation.

Regression tests cover P1/P2 leading answers, last-place compatibility, ahead/behind counts,
place out of field, Turkish wording, Qwen-plan mapping, and a P1-to-P2 refresh between the
initial decision and delivery. Run new independent model paraphrases and a live race test
before closing the issue.

The 2026-09-27 retained-Qwen CPU smoke check passed the final English/Turkish first, last
and rear-count probes on the synthetic four-car field. An initial “Lider miyiz?” run chose
the correct `first` relation but English reply language; a symmetric Turkish example fixed
that phrase and a distinct “Şu an birinci miyiz?” paraphrase also passed. This small targeted
check does not replace an independently authored bilingual set or live validation.

## CONV-007: Telemetry facts are difficult to extend consistently

**Status:** CE-05R.4 provider catalog and R.5 composition implemented; broader facts and
R.5.5 acceptance pending

**Reported:** 2026-09-26, driver feedback after the CE-05/05.1 live test

Fact retrieval was spread across separate semantic maps, capability checks, controller
branches, legacy lookup code and live refresh logic. Adding information risked advertising
one capability while resolving or refreshing it differently elsewhere. It also made the
bounded deterministic grounding layer feel like a growing collection of special cases.

CE-05R.4 introduces one complete `FactCatalog`. Every `RaceQuery` is registered exactly
once with its semantic meaning/reference and deterministic resolver. Context assembly,
Qwen compatibility conversion, initial answers and delivery-time refresh use that catalog;
the legacy route delegates to it while preserving its nearest-gap behavior. Duplicate or
missing registrations fail construction, and replacement-provider tests prove that one
registration drives both advertised availability and the grounded answer.

This does not expose arbitrary telemetry to Qwen or remove the need to define trustworthy
derived facts. New information still needs a typed query, capability-aware provider,
bilingual wording and evaluation. CE-05R.5 now composes those facts more naturally without
adding another model call; broader fact coverage remains separate.

## CONV-001: Vague fuel follow-up switches to position

**Status:** Open - first conversational prototype

**Observed:** 2026-09-20, local Qwen3-4B-Instruct-2507 Q4_K_M, CPU

After "How is fuel looking?", "Where are we now?" returns current position rather than
staying with fuel or clarifying the topic. The position value is grounded, but the intended
topic is lost. Reproduced by `fuel-context` in `fixtures/conversation/cases.json`.

## CONV-002: Ambiguous opponent reference can be guessed

**Status:** Open - first conversational prototype

**Observed:** 2026-09-20, same local-model setup

With no preceding opponent context, "Is he pulling away?" can select the car ahead instead
of asking which car. Gap trends are unsupported, so it does not invent a trend, but the
reference resolution is wrong. Reproduced by `ambiguous-reference` in the model evaluation.

## CONV-003: Unsupported part of a compound question can be omitted

**Status:** Open - first conversational prototype

**Observed:** 2026-09-20, additional paraphrase evaluation

"Tell me the overall position and recommended tyre pressures" returns position but omits
the explicit acknowledgement that tire advice is unavailable. It does not invent tire
pressures. Reproduced by `partially-supported-compound` in
`fixtures/conversation/holdout.json`.

CONV-001 through CONV-003 were reproduced with typed input, independently of transcription.
Keep the failing evaluation cases rather than weakening their expected behaviour.

## IR-001: False blue-flag call immediately after race start

**Status:** Fixed in normalization - live validation pending

**Area:** iRacing telemetry normalization and flag-event derivation

**Observed:** 2026-09-19, session `iracing:1:0`, source sequence `3732`

During the M3 live speech validation, the system announced "Blue flag" roughly 2.7 seconds
after the green-flag call. The driver confirmed that no blue-flag condition existed. The
policy, language, and TTS stages behaved correctly for the event they received. Inspection
of the recorded state showed the player starting lap 1, becoming P1, and no active opponent
with a greater completed-lap count. The defect was the normalizer accepting the raw blue
bit without establishing that it could apply to the player.

Race-session normalization now requires a valid player classification and an active,
non-pace-car opponent with a greater completed-lap count before exposing blue. The raw bit
continues to determine whether iRacing considers traffic close enough; the additional check
only corroborates that lapping traffic exists. Practice and qualifying preserve iRacing's
blue signal once player position is valid because completed-lap comparisons do not carry
the same meaning there.

The reduced regression fixture at
`fixtures/iracing/ir001_false_blue_start/raw_samples.jsonl` proves both that the captured
start transition produces no event and that a lapping-opponent control does produce one.
A live race with a genuine blue flag is still required before closing the issue completely.
No policy cooldown or timing workaround was added.
