# Known issues and deferred improvements

## STT-001: Short and single-word questions are sometimes misrecognized

**Status:** Deferred improvement - accepted prototype limitation

**Reported:** 2026-09-21, driver feedback after the combined live test

Recognition sometimes fails, especially for one-word input. The driver accepts the current
prototype for now. This report does not yet identify whether capture boundaries, the energy
gate, language detection, or the recognizer is responsible; do not assume an engine defect.

Future work:

- distinguish clipped/discarded capture from incorrect transcription using explicitly
  supplied or consented English/Turkish test samples; no background microphone recording;
- evaluate one-word questions such as "Position", "Fuel", "Sıra", and "Yakıt", along with
  longer paraphrases and realistic racing noise;
- assess capture timing, silence thresholds, language selection, accuracy, and latency
  before choosing tuning or an alternative local recognizer; and
- request a repeat when input cannot be reliably understood, rather than inventing a query.

Keep Qwen3-ASR as the current adapter. faster-whisper remains an evaluation alternative,
not a newly selected engine or automatic fallback. Natural phrasing must remain supported;
the test examples are not a restricted command vocabulary.

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

**Status:** Deferred feature request - not implemented

**Requested:** 2026-09-21

Conversation should respond naturally to remarks as well as factual questions. For
example, "That's dirty" could receive a concise acknowledgment or "Focus on your race
now", rather than being forced into a position/fuel query or a generic unsupported reply.
The requested feel is a supportive race engineer, not simply a spoken telemetry lookup.

Future work:

- add bounded conversational acts for acknowledgments, reassurance, and refocusing,
  alongside the existing read-only factual-query route;
- support English/Turkish paraphrases and conversational context without requiring fixed
  phrases or generating a reply to every remark;
- keep responses short and subject to radio priority, expiry, and interruption; critical
  calls still take precedence and busy racing situations should not gain extra chatter;
- distinguish the driver's report from verified race evidence: "Copy. Focus on your race"
  is a possible neutral reply, while "Yes, we saw it" requires evidence the system can
  actually observe. Do not invent witnessed contact, assign blame, or imply steward review;
- keep factual race answers grounded and critical calls deterministic; personality must
  not become an unrestricted source of race facts or strategy; and
- add evaluation cases separating factual questions, emotional remarks, mixed inputs,
  and ambiguous comments, including when clarification or silence is preferable.

These three items are follow-up quality/features, not changes to current runtime behavior.
See [roadmap.md](roadmap.md) for milestone status and the proposed next implementation slice.

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
