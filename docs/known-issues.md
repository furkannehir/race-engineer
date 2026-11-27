# Known issues

Open items only. A fixed issue is removed; its history stays in git. A dialogue failure
seen in a live session also becomes a case in the eval question bank
(`fixtures/intelligence-evaluation/`).

## Speech output

**TTS-002: Chatterbox can add gibberish after the sentence.** A nine-word reply once
produced 12 s of audio with an unrelated tail, and the failure was reproduced offline in
both BF16 and FP32. An EOS token was present every time, so EOS doesn't prove clean
audio. Chatterbox 0.1.7 hard-codes `max_new_tokens=1000` (40 s of audio), and nothing
bounds the output by text length except a 60 s sanity limit.

**TTS-001: The voice could sound more natural.** The built-in Chatterbox voice is
accepted for now. Turkish pronunciation and naturalness haven't been reviewed by a
listener.

**TTS-003: Speech isn't real-time on CPU.** Short CPU checks took about 6.6 s (English)
and 16 s (Turkish). Machines without NVIDIA, and every installed build today (see
DIST-002), can miss the reply delivery budget.

## Speech input

**STT-001: Short and single-word questions are sometimes misrecognized.** For example
"Position", "Fuel", "Sıra" and "Yakıt". The cause (capture boundaries, the energy gate,
language detection or the recognizer) hasn't been isolated. Measure with consented
English and Turkish samples before tuning or switching engines.

## Dialogue

**INT-017: Valid replies are rejected, and the driver hears nothing.** The reply check
rejects correctly rounded numbers ("1.6 seconds" for a 1.6347 s gap), spoken numbers
("one point six"), number words in idioms ("one corner at a time", "first-class"), and
bare gaps with no unit. After two rejections the driver gets only a screen notice. "Fifth
place" is also rewritten to "P5 place".

**INT-018: Turkish understanding is unfinished.** A Turkish radio check can be
classified as unsupported information, and some Turkish replies reverse ahead and behind
despite a correct tool result. English is the primary validated language.

**INT-016: Replies can miss the driver's meaning.** Known cases:

- A frustrated remark about an opponent selects the current position.
- Asked to recall the last lap, the model picks the most recent best-lap reading.
- A request for fuel laps can get fuel in litres.
- Mixed social and factual input can lose the acknowledgment.
- "Is he pulling away?" with no earlier context picks the car ahead instead of asking
  which car (was CONV-002).
- The unsupported part of a compound question can be left out without saying it's
  unavailable (was CONV-003).
- A vague follow-up after a fuel question switches to position (was CONV-001).

## Telemetry

**IR-001: Blue-flag fix needs a live check.** The false blue flag right after the start is
fixed in normalization and covered by `fixtures/iracing/ir001_false_blue_start`. A race
with a genuine blue flag is still needed to confirm real blue flags are still announced.

## Distribution

**DIST-002: Installed builds don't use the GPU.** The installer ships only the CPU
llama.cpp runtime and a CPU speech worker. Only source mode uses CUDA.

**DIST-001: Clean-machine acceptance is pending for the frozen speech worker.** The fix
for the 0.1.0 startup failure (Transformers source files and dependency metadata in the
bundle) is verified on the build machine. A full clean-Windows run is still needed.
