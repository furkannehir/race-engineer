# M3 local text-to-speech

The initial speech adapter plays validated `Utterance` values through the Windows Speech
API (SAPI). It uses voices already installed on the machine, works offline, and adds no
Python package or hosted-service dependency.

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

The first adapter uses the Windows default audio output and installed desktop voices. Voice
downloads, neural voice engines, explicit output-device routing, pre-generated critical
audio, and audio latency benchmarking remain later adapter improvements. None require a
change to the policy or language contracts.
