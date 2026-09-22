# Local conversational prototype: text over replay

This first slice accepts freely phrased English and Turkish questions about a paused
telemetry fixture. Qwen3-4B-Instruct-2507 resolves the question and recent conversational
context into validated, read-only queries. Application code retrieves current facts and
renders short bilingual replies. There is no keyword/synonym command router.

The reply wording is deliberately deterministic in this slice. The model does not yet
write unrestricted answers: JSON-schema validity alone would not prove those answers
factually correct. The supported information is limited, but the input phrasing is not.
Semantic interpretation can still be wrong and needs real-model evaluation.

## Run it

The runtime and model are separate from the Python package. On the initial development
machine they are placed under the git-ignored `data/conversation-prototype` directory.
For another checkout, follow the setup section below first.

From the repository root, start the local model in one terminal:

```powershell
.\.venv\Scripts\python.exe scripts/start_conversation_model.py
```

Wait for the server to report that it is listening. In a second terminal:

```powershell
.\.venv\Scripts\python.exe -m race_engineer chat-replay fixtures/synthetic/conversation `
  --config config/default.toml
```

Example questions (not a required vocabulary):

- "Where are we?"
- "What about the guy behind?"
- "Peki öndeki?"
- "Yaklaşıyor mu?"
- "Kaçıncıyız ve depoda kaç litre var?"
- "Can we finish without stopping?"

The demo data is synthetic, not a recording of an actual race. It starts at P6, with an
ahead gap of 1.8 seconds, a behind gap of 2.4 seconds, and 32 liters of fuel. These are
fixture values, not promises about what a real session will expose.

Replay is paused until explicitly advanced:

- `/next` or `/next 10`: move forward by that many frames, retaining conversational context.
- `/frame 0`: seek to a zero-based frame index and clear conversational history.
- `/state`: show replay position, session, and source sequence without asking the model.
- `/reset`: clear conversation history without moving the replay.
- `/quit` or Ctrl+C: exit the conversation client. Stop the model terminal separately.

The demo has three frames. Frame 1 changes position to P5 and the gap behind to 1.1 seconds;
frame 2 omits fuel and gap data. Ask a follow-up after advancing to check fresh retrieval
and honest handling of missing information.

For a single question and a traceable JSON result:

```powershell
.\.venv\Scripts\python.exe -m race_engineer chat-replay fixtures/synthetic/conversation `
  --config config/default.toml --frame-index 1 --question "Where are we?" --json
```

Use `--language tr` or `--language en` to force reply language. Otherwise Qwen selects it
from the current question and conversation. On model failure before any language has been
detected, the configured `default_language` is used.

An existing recording directory can replace `fixtures/synthetic/conversation`. It still
runs as a paused replay, never as live telemetry. Model errors produce a readable response;
single-question mode also returns exit code 1. Configuration/fixture errors return 2.

## Supported information and boundaries

Current queries cover overall position, lap, same-lap gaps ahead/behind, fuel remaining,
and observed average fuel consumption. Each answer carries a session ID, source sequence,
replay/live marker, and its retrieved facts in the JSON result.

Fuel-to-finish and gap trends are recognized but explicitly unavailable. The prototype
does not infer a trend from a single gap or recommend pit strategy from fuel level alone.
Class position, driver names, flags, tires, historical comparisons, and settings changes
are not supported. The known false blue-flag issue is unchanged and outside this slice.

Only meanings/questions are retained in bounded in-memory conversation history; previous
answer values are not supplied to the model. Facts are fetched again after inference.
Session changes clear memory and discard in-flight answers. Freshness is checked before
and after inference. Replay freshness uses the selected frame's clock, so an old recording
is valid replay data; a future live provider must supply current UTC time.

The strict policy, live iRacing command, existing deterministic announcements, and speech
queue are unchanged. The `chat-replay` command remains text-only; the new
[speech-input slice](speech-to-text.md) adds Qwen3-ASR push-to-talk through `voice-replay`.
`voice-replay` also speaks replies using the local bilingual
[Piper output adapter](conversational-speech.md); `--text-only` disables playback. Neither
replay command connects to live iRacing. The separate
[`voice-iracing` command](live-conversation.md) now connects the same conversation layer
to live context and coordinates radio playback. faster-whisper remains a future
alternative. Voice quality and in-race performance still need evaluation.

## Local model setup

The application uses llama.cpp's local schema-constrained chat endpoint. It connects only
to literal `127.0.0.1`, ignores HTTP proxy environment settings, rejects redirects, and has
no cloud fallback, API key, automatic download, or hosted SDK dependency. See the
[llama.cpp server documentation](https://github.com/ggml-org/llama.cpp/tree/master/tools/server).

The initial smoke-test setup uses a CPU runtime with eight threads, an 8192-token context,
and a Q4_K_M GGUF. CPU remains the configured compatibility baseline while accelerated
runtimes are evaluated; this is not an in-race performance recommendation. The weights are
an Unsloth conversion of the selected Qwen model, not a different conversational model. See the
[original model card](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507) and the
[GGUF publisher](https://huggingface.co/unsloth/Qwen3-4B-Instruct-2507-GGUF).

Pinned setup artifacts:

| Artifact | Source | SHA256 |
| --- | --- | --- |
| Windows CPU runtime ZIP, about 18 MB | [llama.cpp b10964](https://github.com/ggml-org/llama.cpp/releases/download/b10964/llama-b10964-bin-win-cpu-x64.zip) | `917f39c076402c421224824607397af20f53625a60defc20e8dd22446bf4c5d7` |
| Q4_K_M model, about 2.50 GB | [Pinned GGUF revision](https://huggingface.co/unsloth/Qwen3-4B-Instruct-2507-GGUF/resolve/a06e946bb6b655725eafa393f4a9745d460374c9/Qwen3-4B-Instruct-2507-Q4_K_M.gguf) | `3605803b982cb64aead44f6c1b2ae36e3acdb41d8e46c8a94c6533bc4c67e597` |

Download these files using the links, verify them with `Get-FileHash -Algorithm SHA256`,
then put the GGUF in `data/conversation-prototype` and extract the ZIP into
`data/conversation-prototype/llama-b10964-cpu`. The start script also accepts `--server`
and `--model` if you keep model assets elsewhere. No global installation or PowerShell
execution-policy change is required.

After setup, both server and conversation client can run without internet access. Bind
the server to loopback as in the script. App logs contain failure codes, not prompts,
transcripts, or answers; terminal replies are intentionally visible. External runtime
logging settings are separate. Conversation history is not saved to disk.

The `[conversation]` TOML section controls the local server port, model alias, timeout,
bounded history, freshness limit, and default language. `[conversation.runtime]` controls
the machine-local executable/model paths, CPU thread count, context size, parallel slots,
startup/probe/shutdown limits, compute mode, device, and offload. The same launch service is
used by the panel and `scripts/start_conversation_model.py`; the script reads
`config/default.toml` unless `--config` selects another file. Its legacy `--server`,
`--model`, `--port`, and `--threads` arguments remain bounded one-run overrides.

`mode = "cpu"` is the current default and uses the pinned CPU-only binary with zero GPU
layers. `gpu` and `auto` require both `accelerated_executable_path` and
`accelerated_backend`. The runtime runs a bounded `--list-devices` probe before attempting
acceleration. With `fallback_to_cpu = true`, a missing/unusable accelerated runtime or one
failed startup is cleaned up and retried once on CPU. An already-running matching server is
reused and reported as unmanaged because the application cannot verify or change its
hardware settings. No accelerated binaries are installed by this slice; CUDA/Vulkan setup,
panel controls, and hardware performance claims belong to CE-03.

## Verification

Unit tests use scripted planners and mocked HTTP responses. They test grounding,
freshness, bilingual rendering, session reset, error handling, and CLI behaviour; they do
not establish Qwen's ability to interpret natural language.

With the real local server running, execute the separate, synthetic model evaluation:

```powershell
.\.venv\Scripts\python.exe scripts/evaluate_conversation.py
```

It checks English/Turkish phrasing, language switching, follow-ups, compound questions,
clarification, unavailable capabilities, and a prompt-injection attempt. Exit code 1 means
at least one case failed. Per-turn results include latency and the selected queries. This
small suite is an integration check, not proof of general language accuracy or a racing
latency benchmark. The evaluation intentionally prints synthetic replies for inspection.

Additional paraphrases, first evaluated after the prompt was tuned on the initial set:

```powershell
.\.venv\Scripts\python.exe scripts/evaluate_conversation.py `
  --cases fixtures/conversation/holdout.json
```

### Initial result, 2026-09-20

The CPU Q4_K_M smoke run passed 18/20 development turns and 10/11 additional turns.
The 100 automated application tests, lint, and type checks passed separately. Loaded-model
text turns were roughly 1-3 seconds in this run; this is not an in-race or speech-latency
measurement. Initial prompt processing can take longer.

This is ready for exploratory replay testing, not acceptance as a reliable live engineer.
The remaining interpretation failures are deliberately retained in the evaluation, which
currently exits with code 1. See CONV-001 through CONV-003 in [known issues](known-issues.md).
