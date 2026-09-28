# CE-05 opt-in live dialogue

CE-05 implements the delivery-aware conversational path as a preview. Its schema default is
disabled because the initial screening does not satisfy promotion gates; the current
development `default.toml` deliberately enables it for testing. The established Qwen-v1
route remains available by setting:

```toml
[conversation.dialogue]
enabled = false
adapter = "qwen-v1-hybrid"
```

The preview retains the same local Qwen3-4B model and llama.cpp server. Its compatibility
plan preserves the factual queries, adds CE-05.1's `field_status`, and adds only `mode`
(request, correction, repeat) plus bounded `acknowledge`/`close` acts. The model never
supplies answer wording, race numbers, tool calls, database writes, or preferences. Schema
validation rejects unknown fields and incompatible combinations.

## What changes when enabled

- A mixed remark such as “That was dirty. Gap behind?” may become a neutral “Copy” plus
  a freshly retrieved gap. It cannot claim contact was seen or assign blame.
- A standalone frustration remark gets the initial calm-teammate response “Copy. Focus
  forward” (or the Turkish equivalent). A repeated nearby acknowledgment is shortened to
  avoid nagging. Bounded templates, not the model, produce those words.
- Corrections, repeats, partial answers, targeted clarification and standalone thanks use
  session-owned dialogue state. “Thanks” can deliberately produce no reply.
- Explicit profile language still wins. Otherwise ASR language is a hint rather than an
  implicit preference write. Raw utterances remain out of retained dialogue state by default.
- Before playback, all factual parts are read again from current telemetry. Opponent facts
  are tied to the same driver ID and side; replacement/movement discards the reply instead
  of silently rebinding “him.” Approved social/clarification parts survive refresh.
- The shared radio records accepted/queued/started/completed/interrupted/cancelled/expired/
  failed lifecycle transitions by response ID. Text-only completion cannot claim speech.
  Critical automatic calls retain priority and can interrupt conversation audio/capture.
- Session changes, reconnects, stale facts, expired turns and late model output discard
  state/replies. One driver turn is admitted at a time with no pending backlog.
- CE-05.1 adds a grounded `field_status` capability. Natural English/Turkish variants of
  “Am I last?”, “Is anyone behind us?” and “Sonuncu muyuz?” map to one internal meaning.
  Deterministic code requires a complete, unique overall classification, excludes the known
  pace car during normalization, and returns position, field total and cars behind. It says
  the classification is unavailable instead of inferring across gaps or duplicate places.

## CE-05R.1: current running position

The first recovery slice fixes the stale-position source beneath both strict calls and
conversation. iRacing's scored `PlayerCarPosition`/`CarIdxPosition` values remain the safe
fallback, but an active race now derives the current overall order from each classified
car's live `CarIdxLap + CarIdxLapDistPct` progress. This lets a mid-lap pass update the
latest `RaceContext` and any fact refresh before the next timing-line crossing.

The derived order is fail-closed: it is used only in a Race while the session is racing,
the player appears in the classification, pace cars are excluded, and every classified car
has a finite lap and lap-distance value. A missing or invalid competitor makes the whole
calculation fall back to the official positions rather than mixing sources or guessing.
Frames using the derived order advertise `live_position` for recordings and diagnostics.
Automated tests cover a mid-lap P1-to-P2 change with unchanged official values, the emitted
position event, and incomplete-progress fallback. A live start/overtake test remains the
acceptance gate.

## CE-05R.2: truthful field relationships

The second recovery slice removes the ambiguity that caused a leading driver to hear
“No, you're P1.” `field_status` now carries the relationship actually requested:
`first`, `last`, `cars_ahead`, `cars_behind`, or `position_of_total`. Qwen interprets that
bounded meaning from free English or Turkish phrasing; it still does not calculate the
answer or write the response.

Deterministic code validates the complete current classification, performs the requested
comparison/count, and chooses matching bilingual wording. The relationship is retained in
dialogue memory and refreshed with the latest position immediately before delivery, so a
P1-to-P2 change can turn “Yes, you're leading” into “No, you're P2” without another model
decision. Missing relationships and relationships attached to unrelated queries fail schema
validation. The affected semantic, state, input, grounded-answer, response and private Qwen
plan contracts advance to v2 rather than silently changing their serialized v1 meaning.

A local CPU smoke check on 2026-09-27 used the retained Qwen model and the synthetic
four-car classification. Six English/Turkish prompts covered first, last and cars-behind
relationships, including “Am I first?”, “Lider miyiz?” and “Şu an birinci miyiz?”. The
model selected the requested relation and deterministic code produced a logically matching
grounded reply in all final probes. One initial short Turkish probe selected the correct
relation but English reply language; adding a symmetric Turkish leading example corrected
that phrase and a distinct paraphrase. This is targeted integration evidence, not an
independent bilingual promotion set or live-race validation.

## CE-05R.3: short-input capture diagnostics

The third recovery slice adds a bounded post-release microphone tail and content-free audio
diagnostics so capture/gating failures can be separated from language detection or ASR
recognition failures. Real English/Turkish one-word sample validation remains pending; see
[speech input](speech-to-text.md) and [STT-001](known-issues.md#stt-001-short-and-single-word-questions-are-sometimes-misrecognized).

## CE-05R.4: capability-aware fact providers

The fourth recovery slice replaces the scattered telemetry-query switches with one
declarative `FactCatalog`. Each bounded semantic query has exactly one deterministic
provider. The same registration now drives the capability list shown to the semantic judge,
initial fact resolution, retained-Qwen query conversion, legacy compatibility and fresh
pre-playback resolution. Position, complete-field relationships, lap, opponent gaps, fuel,
and explicitly unsupported meanings preserve their established behavior.

Providers receive only a typed snapshot and return a validated `RaceAnswer`; Qwen still
cannot provide values, call arbitrary tools or inspect the simulator SDK. Opponent providers
remain bound to driver ID and side and fail closed when that identity changes. Adding a new
fact still requires a reviewed typed query, provider, bilingual wording and tests, but no
longer requires editing independent lookup switches throughout the dialogue pipeline.

R.4 is an architectural foundation, not a claim that every iRacing telemetry field is now
available and not a move to unrestricted model-generated answers. R.5 builds natural but
grounded response composition on that foundation; its live acceptance evidence remains open.

## CE-05R.5.1–R.5.4: natural grounded radio composition

The fifth recovery slice now compiles each refreshed response decision into a versioned
utterance plan. Its ordered clauses are limited to acknowledgment, grounded fact,
clarification and failure. Every fact clause names exactly one grounded answer part; plan
validation requires every answer to appear exactly once and in decision order. Duplicate,
missing or foreign fact references fail closed.

A deterministic bilingual renderer selects concise English/Turkish radio variants from
the response and clause IDs. It can say “P7 right now”, “Affirm, P1. We're leading”, or a
brief calm-teammate acknowledgment, but it cannot add race values or claims. In particular,
an unverified complaint may receive “Copy” or a refocus response; the renderer cannot claim
that contact was seen. Repeated acknowledgments collapse to a brief “Copy”/“Anlaşıldı”.

The natural output is checked for response/language scope, exact fact coverage,
acknowledgment and clarification parity, nonempty single-line text and a 48-word radio
limit. Any planning, rendering or validation failure uses the previous bounded composer.
No second Qwen inference was added. The existing live radio already refreshes facts and
opponent identity immediately before invoking the composer, so changed position or gap
values determine the final wording sent to Piper.

R.5.1–R.5.4 have deterministic and live-path integration coverage. R.5.5 remains the
acceptance pass: independent replay scenarios, English/Turkish listening review, critical
radio interruption, and a real iRacing run with latency/resource observations.

## Testing

Normal tests use scripted semantic meanings, fake telemetry/clocks and fake speech; they
do not load Qwen or optional encoder dependencies. Coverage includes mixed social/factual
responses, no-reply, partial/unsupported parts, current-fact refresh, stable opponent
identity, bilingual bounded wording, delivery state and critical radio regressions.

An exploratory local-model smoke check on 2026-09-24 used the configured CPU-only
Qwen3-4B GGUF and the synthetic paused replay. It correctly handled English `Position`,
Turkish `Neredeyiz?`, a rear-gap question, a mixed complaint plus position question, a
standalone Turkish complaint, and standalone thanks/silence. The mixed example initially
showed prompt sensitivity before the compatibility prompt was refined, so these hand-picked
checks are integration evidence, not semantic promotion evidence. The server reported about
7.5 seconds for the first uncached inference and roughly 2â€“3 seconds for subsequent probes;
these are exploratory server timings, not PTT-to-first-audio measurements. The normal
launcher also reached `compute: cpu:cpu`. Windows CLI output is forced to UTF-8 so a valid
Turkish response cannot fail merely because the inherited console uses CP-1252.

An additional 2026-09-26 CPU smoke check used
`fixtures/synthetic/field_status`. The real local Qwen mapped “Am I on the last place?”,
“Sonuncu muyuz?” and “Is anyone behind us?” to `field_status`, and preserved both parts of
“Are we last, and what's the gap ahead?”. The deterministic resolver produced `P4 of 4`,
`P3 of 4`, the rear-car count and the 1.2-second gap. These are targeted integration probes,
not a new independent semantic qualification set.

For a deliberate real run, set `enabled = true`, start the panel normally, and test simple
questions before mixed remarks/corrections. Expected preview examples:

- “Position?” / “Kaçıncıyız?” → grounded current overall position.
- “Am I first?” / “Lider miyiz?” → a grounded yes/no leading comparison plus current place.
- “Am I last?” / “Sonuncu muyuz?” → grounded position out of the complete current overall
  classification, or an explicit unavailable answer when classification is incomplete.
- “How many cars are ahead/behind?” → the requested count plus current place out of field.
- “That was dirty. Gap behind?” → neutral acknowledgment plus grounded rear gap.
- “Position, and is he catching?” → position plus targeted opponent clarification when
  there is no established referent; trend remains explicitly unsupported.
- “Say again” after interrupted audio → recomposed reply using current facts.
- “Thanks” / “Sağ ol” → no extra radio reply.

Record qualitative failures, exact input language, whether audio was interrupted, and the
intended meaning. Do not tune against the frozen locked suite and reuse it as held-out proof.
Before enabling this by default we still need a new independently authored bilingual
holdout, real one-word ASR samples, listening review, radio-interruption checks, and
same-hardware PTT-to-first-audio/resource measurements. CE-05 implementation is complete;
promotion/extended live validation remains pending.
