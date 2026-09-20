# M3 local text-to-speech

The initial speech adapter plays validated `Utterance` values through the Windows Speech
API (SAPI). It uses voices already installed on the machine, works offline, and adds no
Python package or hosted-service dependency.

Conversational `voice-replay` replies now use a separate bilingual Piper adapter. See
[conversational-speech.md](conversational-speech.md) for its setup and `[radio_tts]` settings.
The automatic race-call SAPI path described here is unchanged.

## Runtime behavior

Speech runs outside the telemetry loop in a bounded async queue. The queue:

- rejects utterances whose intent deadline has passed;
- plays higher-priority pending calls first;
- lets critical intents interrupt playback according to their `interruption_policy`;
- drops lower-priority pending work when a full queue receives more important speech; and
- isolates synthesis failures so telemetry, policy, language, and recording continue.

The SAPI COM boundary runs in a hidden PowerShell child process. Text and voice settings
are passed as encoded child-process environment values rather than interpolated into a
shell command. Normal logs contain intent IDs and playback outcomes, not spoken text.
Each result also reports queue-wait and playback duration in milliseconds for live latency
validation.

## Configuration

```toml
[tts]
enabled = true
adapter = "windows-sapi"
rate = 0
volume = 100
playback_timeout_s = 10.0
queue_capacity = 8
```

`rate` accepts SAPI values from -10 through 10, and `volume` accepts 0 through 100. When
`voice` is omitted, Windows uses its default SAPI voice. To select a specific installed
voice, copy its full name into the configuration:

```toml
voice = "Microsoft David Desktop - English (United States)"
```

List the available names and perform an audible check with:

```powershell
race-engineer list-tts-voices --config config/default.toml
race-engineer test-tts --config config/default.toml
```

The test command says “Radio check. Race engineer online.” by default. It also accepts a
custom `--text` value.

## Deliberately deferred

This automatic-call adapter uses the Windows default audio output and installed desktop
voices. Piper voice downloads and output-device selection are available for the separate
conversation prototype, but are not used to render automatic calls. The new
[live conversation command](live-conversation.md) coordinates both engines in one radio
scheduler. Pre-generated critical audio and in-race latency benchmarking remain later work.
