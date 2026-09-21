# Live iRacing conversational shakedown

`voice-iracing` runs telemetry, strict-policy automatic announcements, push-to-talk,
Qwen3-ASR, the local Qwen conversational model, and Piper replies in one application.
Conversation now reads the latest accepted iRacing context rather than a replay fixture.
It is read-only: there are no pit commands, setup changes, or simulator controls.

## Run

For the native Windows controls, use the [control panel](control-panel.md). It manages
the local model server and this same live pipeline. The terminal workflow below remains
available; do not run it at the same time as a panel session.

Use the existing [conversation](conversation.md), [STT](speech-to-text.md), and
[Piper](conversational-speech.md) installations. No new model download is required.
Start with a practice/test session, not a competitive race.

Terminal 1, unless the local conversation server is already running:

```powershell
.\.venv\Scripts\python.exe scripts/start_conversation_model.py
```

Terminal 2, from the repository root:

```powershell
.\.venv\Scripts\python.exe -m race_engineer voice-iracing --config config/default.toml
```

Start/join iRacing. Wait for both `telemetry ready` and `Radio ready`, then hold F8, speak,
and release. Ask questions freely in English or Turkish. Telemetry keeps updating during
recognition, model inference, and playback; the model interprets the question but never
supplies numeric race facts.

**Do not run `read-iracing` or `voice-replay` alongside this command.** The live command
already runs the automatic policy/speech pipeline. Separate processes cannot coordinate
their audio and would produce duplicate or overlapping calls.

F8 is a global held-key check, not an intercepted key. If it conflicts with iRacing or
another application, choose a free terminal binding, for example `--ptt-key RCTRL`.
The control panel additionally offers press-to-bind for arbitrary keyboard keys, standard
mouse buttons, and digital wheel/controller buttons or hats. Those bindings remain active
while iRacing has focus; analog axes such as steering and pedals are ignored.

ESC exits while waiting/listening; Ctrl+C exits during processing or playback too.
Shutdown cancels capture, radio playback, and workers and closes the telemetry source
and recorder. The separate conversation server remains under your control.

Optional settings:

- `--device N`: microphone index from `list-input-devices`.
- `--output-device N`: **Piper only**, from `list-output-devices`. SAPI automatic calls
  still use the Windows default output. Set that default to the same headset if desired.
- `--language en` or `--language tr`: force conversational answer language and voice.
  Automatic calls retain their existing English strict-policy wording/SAPI voice.
- `--text-only`: mute **both** automatic and conversational speech for this live command.
- `--limit 1200`: stop after 1,200 accepted telemetry frames; zero means run until exit.
- `--output recordings/live-radio-01`: record frames, events, policy decisions, intents,
  and automatic utterances into a **new** fixture directory. Existing directories are
  rejected. Microphone audio, questions, and conversational replies are not recorded.

Example recorded run (choose a new folder on each run):

```powershell
.\.venv\Scripts\python.exe -m race_engineer voice-iracing --config config/default.toml `
  --output recordings/live-radio-01
```

## Radio behavior

One bounded scheduler owns both speech engines and grants the microphone a capture slot.
It keeps the existing automatic intent priorities, deadlines, and interruption policies.
Conversational replies have priority 50: important/critical pending automatic calls go
first, routine calls go afterwards. Queued replies expire after the configured snapshot
age limit (three seconds by default), rather than accumulating.

- A push-to-talk press while speech is playing or queued is refused. Release and press
  again after the radio clears; there is no conversational barge-in yet.
- Noncritical automatic calls wait while the microphone captures your question.
- Critical calls cancel capture, discard that clip, wait for the microphone stream to
  stop, and then speak. Ask again afterwards. They can also interrupt conversational
  playback; interrupted replies are not automatically resumed/repeated.
- Once capture stops, automatic calls can play during recognition/model inference.
- An interrupted child process is stopped before the next voice starts. Cancellation
  cleanup is awaited by the radio worker, **not** by the telemetry submission path.
- Failed audio leaves printed answers available and does not stop telemetry. Radio logs
  contain request IDs/outcomes, not questions or answer text.

This shares scheduling in `voice-iracing`; the existing standalone `read-iracing` queue
and `voice-replay` behavior remain available and independently testable.

## Freshness and session isolation

Live snapshots use current UTC, not a replay clock. There is no placeholder/fabricated
frame before the first valid telemetry update. Replay frames are rejected in live mode
even if the general telemetry configuration allows them.

The latest accepted context replaces the previous one. Conversation retrieves facts
after model inference and refreshes them again when its audio slot opens. Piper then
prepares audio in memory and waits for an explicit freshness acknowledgement before
opening playback. An expired prepared answer is discarded, not played as current data.

Disconnects, session changes, and stale telemetry invalidate pending/playing speech.
An independent watchdog checks freshness every 100 ms; the default accepted frame age is
three seconds. This also covers paused/frozen telemetry with no new accepted ticks.
Reconnections get a new internal generation even when iRacing reuses its session ID and
restarts tick numbering. A question spanning those generations is discarded and session
memory is reset. The telemetry context/policy state is rebuilt on resumption.

Already-spoken words cannot be recalled. Freshness here means validated snapshots at
answer construction/playback start and cancellation on detected loss of live data—not
a guarantee that a moving gap or position cannot change while a sentence is spoken.

## First combined test

1. Start stationary in a practice session. Ask about position and fuel; compare with
   iRacing's displayed data. Missing/unreliable fields must produce an explicit unavailable
   answer, not an invented number.
2. Drive a few laps. Ask about gaps, lap, and fuel, then use contextual follow-ups. Check
   that the answers follow changing telemetry rather than the old P6 replay fixture.
3. Try English and Turkish. Evaluate actual microphone accuracy and voice intelligibility.
   Replacing the Turkish voice remains deferred; this integration does not change it.
4. Observe naturally occurring automatic calls. Listen for overlap, interrupted questions,
   or repeated answers. Do not create incidents solely to trigger a critical call.
5. Leave the session, wait for the unavailable message, and rejoin. Old-session speech and
   follow-up context must not leak into the new session.
6. Check game frame-time/FPS and telemetry warnings (`telemetry_sample_gap`,
   `sampling_deadline_missed`, slow reads). Report questions, expected/actual answers,
   approximate delay, and any radio outcome/error lines.

Fuel-to-finish strategy and gap trends are still unsupported. Existing conversation
interpretation issues and the blue-flag defect are unchanged; see [known-issues.md](known-issues.md).

## Validation status

Automated tests simulate moving telemetry during model inference, disconnection, stale
frames, same-ID reconnection, microphone cancellation, bounded priority playback,
critical-call preemption, late TTS freshness rejection, recorder continuity, and shutdown.
The real Piper worker's prepared-audio gate was checked with playback denied; no microphone
was recorded for that check. On 2026-09-21 the driver reported that the combined live run
works sufficiently for now, with short-input STT issues, unnatural TTS, and a request for
more contextual engineer-style reactions. These are deferred in [known-issues.md](known-issues.md).
This is qualitative feedback, not completion of every test above. Measured audio quality,
latency, in-game resource impact, and extended live scenarios remain to be evaluated.
Processing timers are diagnostics, not a measured first-audio SLA.
