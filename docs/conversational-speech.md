# Local spoken replies: Piper

`voice-replay` now speaks the same grounded answer it prints, in English or Turkish.
Qwen3-ASR transcribes your question; Qwen3-4B interprets it; application code retrieves
replay facts and renders the answer; Piper synthesizes that answer locally. There is no
cloud service, voice cloning, or second model rewriting the answer.

`voice-replay` is still the paused-replay prototype. The separate
[`voice-iracing` command](live-conversation.md) now reads live telemetry and coordinates
Piper replies with automatic SAPI calls in one bounded radio scheduler. Standalone
`read-iracing` keeps its existing automatic-call priority queue.

## Setup

From the repository root, after the existing conversation and STT setup:

```powershell
.\.venv\Scripts\python.exe scripts/setup_radio_tts.py
```

The explicit online setup installs Piper 1.8.0 and sounddevice into the separate
`data/tts-prototype/runtime` environment, then downloads two pinned voice models, their
configuration files, and model cards. Model downloads total about 177 MB. It does not
replace the ASR environment or change automatic SAPI calls. Assets are ignored by Git.
Existing model files are verified and not overwritten; incomplete downloads remain as
`.partial` files for inspection. Runtime synthesis never downloads voices.

Selected prototype voices:

| Reply language | Piper voice | Model card's dataset notice |
| --- | --- | --- |
| English | `en_US-ljspeech-high` | Public domain |
| Turkish | `tr_TR-dfki-medium` | CC BY-NC-SA 4.0 |

These are two separate voices, not one speaker with a consistent bilingual identity.
Voice naturalness and racing pronunciation still need listening tests.

Piper's current engine is [GPL-3.0](https://github.com/OHF-Voice/piper1-gpl).
The [Turkish model card](https://huggingface.co/rhasspy/piper-voices/blob/c10ece1aade47bb51c153c893d14e5bf8e5b7117/tr/tr_TR/dfki/medium/MODEL_CARD)
lists a non-commercial, share-alike dataset license and fine-tuning from Lessac.
The [English model card](https://huggingface.co/rhasspy/piper-voices/blob/c10ece1aade47bb51c153c893d14e5bf8e5b7117/en/en_US/ljspeech/high/MODEL_CARD)
lists the LJ Speech dataset as public domain. Keep these notices with the models. Engine,
voice, and upstream dataset terms require review before product distribution; this
prototype selection is not a distribution-license decision or clearance.

## Talk to the engineer

Terminal 1 (unless the conversation server is already running):

```powershell
.\.venv\Scripts\python.exe scripts/start_conversation_model.py
```

Terminal 2:

```powershell
.\.venv\Scripts\python.exe -m race_engineer voice-replay fixtures/synthetic/conversation `
  --config config/default.toml
```

Wait for `Ready`, hold F8, speak, then release. Your transcript appears before conversation
inference finishes. The grounded answer appears as text before audio playback starts.
English answers use the English voice; Turkish answers use the Turkish voice. `--language`
forces the answer language and therefore also its voice.

The microphone stream is stopped before transcription and stays stopped through synthesis
and playback. A second question is not captured or queued during a reply. Release a key
held during playback before pressing it for the next question. This is turn-taking, not
barge-in or echo cancellation. Headphones are recommended for listening tests.

ESC exits while waiting/listening. **Ctrl+C stops synthesis/playback and exits** by killing
the worker and closing capture. There is no separate stop-speaking/resume command yet.

Add `--text-only` to skip voice loading and playback entirely. `chat-replay` remains
text-only. `[radio_tts].enabled = false` also disables conversational speech.

## Independent radio checks and output selection

Neither the microphone, ASR, nor conversation model is needed for these checks:

```powershell
.\.venv\Scripts\python.exe -m race_engineer test-radio --config config/default.toml --language en
.\.venv\Scripts\python.exe -m race_engineer test-radio --config config/default.toml --language tr
.\.venv\Scripts\python.exe -m race_engineer list-output-devices
```

`test-radio --no-playback` synthesizes in memory without opening an audio output. `--text`
accepts a custom test sentence. For an end-to-end WAV-input test, `voice-replay --audio`
now speaks the answer too; add `--text-only` to retain its old quiet behavior.

Output defaults to the Windows/default PortAudio device. Add `--output-device N` to
`voice-replay` or `test-radio`, choosing an index from `list-output-devices`. Indices can
change after reconnecting hardware. Prefer an MME/default output for this baseline: the
voices produce mono PCM16 at 22,050 Hz, and some WASAPI devices reject that format. There
is no automatic output-rate conversion or switch to another device.

## Configuration and failure behavior

`[radio_tts]` in `config/default.toml` controls conversational audio only:

- `english_model_path` and `turkish_model_path` choose local ONNX files with adjacent
  `.onnx.json` configurations; language and supported eSpeak phonemizer must match.
- `volume` is 0 through 1, default 0.8; `length_scale` is 0.5 through 2, default 1.
  Larger length scales speak more slowly.
- `threads` defaults to four CPU inference threads. No GPU runtime is installed.
- `startup_timeout_s` defaults to 60; `playback_timeout_s` defaults to 90 and covers
  synthesis plus playback. Timeout/cancellation kills the worker, not just its awaiter.

The worker loads both voices once per interactive session and uses private stdin/stdout
pipes, not a network endpoint. It buffers each reply's audio in memory, limited to 60
seconds and 1,500 input characters, before opening playback. It refuses oversized output
rather than deliberately speaking a truncated answer. No reply WAVs are written to disk.
Only validated application replies reach speech; silence/unsupported ASR input never does.
Clarifications, unavailable-data answers, and safe conversation-error replies are spoken too.

If startup fails, interactive conversation continues in text-only mode with setup guidance.
If synthesis or playback fails, the already printed answer remains usable; the next turn
can restart the worker. Partial audio from a device failure is not automatically replayed.
Single-WAV mode reports a nonzero exit code when requested speech fails.

Worker errors use bounded reason codes, never native exception text or reply text. Replies
remain visible in the terminal as before. The ASR, conversation, and TTS layers remain
independently replaceable behind their core interfaces.

## Verification and remaining work

Automated checks cover bilingual routing, persistent-worker reuse, timeout/restart,
cancellation cleanup, bounded audio, malformed messages, text-only mode, and interactive
text fallback. No unit test downloads voices, records the microphone, or plays real audio.

Initial local checks on 2026-09-20 synthesized and played radio checks in both languages
through the default Razer output. Playback completion is programmatically verified;
perceived quality and pronunciation still require the driver's listening assessment.
A synthetic English WAV also passed the full ASR-to-conversation-to-playback path:
"Where are we?" produced and spoke "You're P6 overall." No microphone was recorded for
this check. It validates integration, not human recognition or perceived voice quality.

Reported `synthesis_ms` excludes voice loading. `playback_ms` includes stream setup and
drain. `ASR + reply processing` excludes synthesis/playback and starts after capture stops.
These are diagnostics, not a measured push-to-talk release-to-first-audible-response SLA.
Live integration and critical-call interruption are implemented in `voice-iracing`.
The actual iRacing shakedown, in-race CPU/frame-time impact, and detailed end-to-end latency
evaluation remain pending.

The pinned voice repository revision is `c10ece1aade47bb51c153c893d14e5bf8e5b7117`.
Setup verifies the published LFS SHA256 values:

- English: `5d4f08ba6a2a48c44592eed3ce56bf85e9de3dd4e20df90541ae68a8310c029a`
- Turkish: `2844717f524ab965d3fe86e60562cbb601d3e456836efcc2196cc3a14112a8fb`

Implementation follows the [Piper Python API](https://github.com/OHF-Voice/piper1-gpl/blob/v1.8.0/docs/API_PYTHON.md)
with local ONNX CPU sessions and sounddevice raw playback.
