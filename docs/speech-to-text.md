# Local push-to-talk: speech input

The replay conversation accepts microphone input through Qwen3-ASR. It prints the
transcript and a grounded answer, now also spoken by the separate local
[Piper output adapter](conversational-speech.md). Use `--text-only` for input-only tests.
The separate [live command](live-conversation.md) connects this input to iRacing context
and coordinated radio playback. Existing automatic SAPI wording is unchanged.

## Setup and run

Run commands from the repository root. The one-time setup installs a CPU ASR runtime in
`data/stt-prototype/runtime`, adds the small microphone dependency to the main `.venv`, and
downloads the official Qwen3-ASR model. It does not replace the text conversation runtime.

```powershell
.\.venv\Scripts\python.exe scripts/setup_stt.py
```

This explicit setup requires internet access. The ASR weights are about 1.6 GB, plus runtime
packages. Everything under `data` is ignored by Git. Runtime recognition uses local files
only, with Hugging Face offline mode enabled and remote model code disabled. There is no
cloud transcription or automatic switch to another recognizer.

Start the existing conversation model in terminal 1:

```powershell
.\.venv\Scripts\python.exe scripts/start_conversation_model.py
```

List microphones without recording anything:

```powershell
.\.venv\Scripts\python.exe -m race_engineer list-input-devices
```

In terminal 2, start the spoken replay:

```powershell
.\.venv\Scripts\python.exe -m race_engineer voice-replay fixtures/synthetic/conversation `
  --config config/default.toml --text-only
```

Wait for `Ready`, hold **F8**, speak, and release it. The terminal shows transcription and
the engineer's answer. This example keeps replies text-only. To hear replies, complete the
[voice setup](conversational-speech.md) and omit `--text-only`.
Try English or Turkish phrasing freely, including follow-ups.
The microphone stream runs only during capture; no microphone audio is written to disk.
The ASR worker is loaded once and reused between turns.

Use `--device 1` to select a listed microphone or `--ptt-key F9` to change the binding.
Indices are machine-specific and may change after reconnecting devices. The default uses
the system input device. Supported keyboard bindings are F1-F24, SPACE, RCTRL, and RALT.
Keys are not intercepted, so choose one that does not conflict with other applications.
Native steering-wheel/controller bindings remain later work.

ESC exits while waiting or recording; Ctrl+C also exits, including during processing.
A press held during processing must be released before starting another question. The
prototype does not queue microphone questions while the previous answer is processing.

Replay remains paused at the selected frame. Use `--frame-index 1` to start at another
snapshot; this voice command does not yet expose the text client's `/next` controls.
`--language tr` or `--language en` forces the answer language. Otherwise replies use the
recognized English/Turkish language. ASR language detection and answer language are separate.

## Test transcription independently

For an existing recording, use a PCM16 mono/stereo WAV at 16 kHz, 44.1 kHz, or 48 kHz:

```powershell
.\.venv\Scripts\python.exe -m race_engineer transcribe-wav path/to/question.wav `
  --config config/default.toml
```

This command needs only the ASR worker, not the conversation server. Output includes text,
detected language, clip duration, and model processing time. It does not write a recording.

To pass a WAV through both recognition and conversation without opening the microphone:

```powershell
.\.venv\Scripts\python.exe -m race_engineer voice-replay fixtures/synthetic/conversation `
  --config config/default.toml --audio path/to/question.wav --text-only
```

Use your own English/Turkish recordings to assess accents, short questions, and racing noise.
Synthetic speech smoke tests prove integration, not microphone or bilingual recognition quality.

## Limits, privacy, and errors

- Capture defaults to 15 seconds maximum. An overlong hold is discarded, not truncated and
  submitted as an incomplete question. Release the key and try again.
- Clips shorter than 0.2 seconds are ignored. A configurable energy gate rejects silence,
  DC offset, and near-silent input before model inference. This is not a speech/noise
  classifier: engine noise or music can still pass the gate and be misrecognized.
- Only successful English/Turkish transcripts reach conversation. Silence, unsupported
  language, malformed output, capture overflow, and model failure leave its history alone.
- The ASR process uses a private stdin/stdout pipe, not a network endpoint. Startup and
  inference timeouts kill the worker; the next accepted question can start a fresh one.
- Audio is transient memory. Transcripts are intentionally displayed in the terminal and
  become bounded in-memory conversation history, but are not added to application logs
  or recording fixtures. Files supplied through `--audio` are read, not modified.
- `ASR + reply processing` measures work after capture stops, not full push-to-talk latency
  and not time to first audible reply. Piper synthesis/playback is timed separately.

Configuration lives in `[stt]` in `config/default.toml`. Paths are relative to the working
directory, so run from the repository root. `language = "auto"` enables detection; `"en"`
or `"tr"` can help with very short questions when testing a particular language.

If capture fails, check the device index, Windows microphone permissions, and sample rate.
Some WASAPI devices require `sample_rate_hz = 48000`; the worker resamples to 16 kHz.
If quiet speech is discarded, first adjust input gain, then the `silence_threshold_dbfs`
setting if necessary. A lower threshold admits quieter input and more background noise.

The setup script installs CPU PyTorch. Selecting `device = "cuda"` additionally requires a
compatible CUDA PyTorch installation in the isolated environment; it is not an automatic
GPU setup. CPU performance and resource contention still need validation during a race.

## Implementation and reproducibility

`SpeechRecognizer` accepts a bounded PCM clip and returns a versioned `Transcription`.
Qwen3-ASR is behind that interface; faster-whisper remains a possible future alternative.
The worker uses the [official native Transformers Qwen3-ASR integration](https://huggingface.co/Qwen/Qwen3-ASR-0.6B-hf),
which documents both English and Turkish. The first model is `Qwen/Qwen3-ASR-0.6B-hf`, pinned
to revision `7f1569a48a89f3e3f4dc3a5c9d28bddd903bc76c`. The `-hf` checkpoint is the native
Transformers packaging of the selected Qwen3-ASR family, not the conversation model.

Setup pins Transformers 5.17.0 and CPU PyTorch 2.14.0. SHA256 checks:

- `model.safetensors`: `d3f212dd20abecd315d830bc54ae3865e56ebfc3276484e57b771288ba27fd35`
- `tokenizer.json`: `fe1fad59be22a41ee293363fcf95fdedbc7c93f3b49270b1d2e18bd1399a7a05`

Capture uses [sounddevice raw streams](https://python-sounddevice.readthedocs.io/en/0.5.3/api/raw-streams.html).
Push-to-talk polls the high held-state bit of
[GetAsyncKeyState](https://learn.microsoft.com/en-us/windows/win32/api/winuser/nf-winuser-getasynckeystate);
it does not log keystrokes or install a global keyboard hook.

Automated tests use synthetic PCM, fake key states, fake microphone streams, and fake worker
pipes. They verify capture bounds, release/cancel behaviour, silence rejection, bilingual
result parsing, timeout cleanup, and the conversation handoff. Human microphone tests in
English and Turkish are still required before claiming recognition accuracy.
