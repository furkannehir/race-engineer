# Race Engineer

Race Engineer is a local-first, simulator-adaptable foundation for real-time racing
assistance. Its central policy engine decides what is worth communicating; telemetry,
language generation, and speech synthesis remain replaceable adapters.

This repository contains the M0 architecture baseline and the initial M1 iRacing
telemetry reader:

- versioned domain contracts and adapter protocols;
- validated TOML configuration with environment overrides;
- structured console or JSON logging;
- deterministic, versioned replay fixtures;
- a headless async pipeline and fake implementations;
- automated contract, configuration, fixture, and pipeline tests.
- a reconnecting iRacing shared-memory reader with atomic snapshots, normalization,
  event derivation, and fixture-compatible recording.

Production policies, model-backed language generation, speech synthesis, persistence
schemas, and the desktop UI remain later milestones.

## Requirements

- Python 3.12 or newer
- Git

## Development setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
pytest
ruff check .
mypy
```

Validate the default configuration or inspect a replay fixture:

```powershell
race-engineer validate-config --config config/default.toml
race-engineer inspect-fixture fixtures/synthetic/green_flag
```

With iRacing running, print normalized frames until interrupted:

```powershell
race-engineer read-iracing --config config/default.toml
```

Record 600 normalized frames and their deterministic events into a new fixture directory:

```powershell
race-engineer read-iracing --config config/default.toml --limit 600 `
  --output recordings/my-session
```

The live reader intentionally does not retain driver names or arbitrary raw SDK data.

## Design constraints

- The domain never imports a simulator SDK, Qt, an LLM SDK, or a TTS SDK.
- Serialized contracts and fixtures carry explicit schema versions.
- Internal measurements use SI units.
- Wall-clock timestamps are timezone-aware UTC values; session time is monotonic seconds.
- Queues are bounded and expired work is designed to be discarded rather than accumulated.
- Driver data and logs remain local unless explicitly exported.

See [docs/architecture.md](docs/architecture.md) and
[docs/fixture-format.md](docs/fixture-format.md) for the M0 decisions.
