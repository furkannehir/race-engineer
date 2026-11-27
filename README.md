# Pitward

Pitward is a local AI race engineer for iRacing on Windows. It makes automatic radio calls
(flags, position, pit, session phase) and answers push-to-talk questions about gaps,
position, lap times, race progress and fuel, in English and Turkish.

Everything runs on your PC: speech recognition (Qwen3-ASR), the dialogue model
(Ministral 3 3B through llama.cpp) and the voice (Chatterbox Multilingual). No account,
no cloud, and no telemetry leaves the machine. Pitward is read-only: it never changes car
settings or sends inputs to the simulator.

## Download

Get the Windows installer or portable ZIP at **[pitward.com](https://pitward.com)** or from
the [GitHub releases](https://github.com/furkannehir/race-engineer/releases). The packaged
app doesn't need Python. First setup downloads about 7.0 GB of local AI components under
their own third-party terms.

An NVIDIA GPU is recommended. Installed builds currently run the dialogue model and the
voice on CPU, so replies are slower (see [known issues](docs/known-issues.md)).

## Development

Requires Windows, Python 3.12 and, for the source default, an NVIDIA GPU.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev,desktop]"
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m mypy
```

Install the local models, then start the panel:

```powershell
.\.venv\Scripts\python.exe scripts\setup_ministral.py --backend gpu
.\.venv\Scripts\python.exe scripts\setup_stt.py
.\.venv\Scripts\python.exe scripts\setup_radio_tts.py
.\start-panel.cmd
```

Contribution rules for people and agents are in [AGENTS.md](AGENTS.md).

## Design constraints

- The domain never imports a simulator SDK, Qt, an LLM SDK or a TTS SDK.
- Python owns the race facts; the model only chooses what to read and how to say it.
- Critical calls are deterministic and never wait on a model.
- Serialized contracts and fixtures carry explicit schema versions.
- Internal measurements use SI units. Timestamps are timezone-aware UTC.
- Queues are bounded, and expired speech is dropped rather than played late.
- Driver data and logs stay local unless explicitly exported.

## Docs

- [Architecture](docs/architecture.md): processes, data flow, dialogue, radio and key
  decisions
- [Development](docs/development.md): source setup, running, terminal tools,
  configuration and diagnostics
- [Distribution](docs/distribution.md): Windows builds, updates and the release flow
- [Known issues](docs/known-issues.md)
- [Fixture format](docs/fixture-format.md)
- [Release records](docs/releases/)

## License

See [LICENSE](LICENSE). Third-party model and runtime terms are in
[distribution/](distribution/).
