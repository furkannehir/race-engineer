# Race Engineer

Race Engineer is a local-first, simulator-adaptable foundation for real-time racing
assistance. Its central policy engine decides what is worth communicating; telemetry,
language generation, and speech synthesis remain replaceable adapters.

This repository contains the M0 architecture baseline, the M1 iRacing telemetry reader,
the deterministic M2 policy slice, and the first M3 language slice:

- versioned domain contracts and adapter protocols;
- validated TOML configuration with environment overrides;
- structured console or JSON logging;
- deterministic, versioned replay fixtures;
- a headless async pipeline and fake implementations;
- automated contract, configuration, fixture, and pipeline tests;
- a reconnecting iRacing shared-memory reader with atomic snapshots, normalization,
  event derivation, and fixture-compatible recording; and
- deterministic race-context features, strict rules, scheduling constraints, and
  traceable policy decisions; and
- fact-bound English wording with deterministic replay and an explicit fallback boundary.

Model-backed language generation, speech synthesis, durable decision storage, adaptive
policies, and the desktop UI remain later work.

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
race-engineer replay-policy fixtures/synthetic/m2_green_flag `
  --config config/default.toml
race-engineer replay-language fixtures/synthetic/m3_green_flag `
  --config config/default.toml
```

With iRacing running, print normalized frames until interrupted:

```powershell
race-engineer read-iracing --config config/default.toml
```

Record 600 normalized frames plus their events, policy decisions, speech intents, and
generated utterances into a new replayable session directory:

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
[docs/fixture-format.md](docs/fixture-format.md) for the M0 decisions, and
[docs/strict-policy.md](docs/strict-policy.md) for the M2 policy behavior. The M3 wording
boundary is described in [docs/language-generation.md](docs/language-generation.md). The
architecture also records Jev-assisted, shadow-mode policy ranking as a post-M3 improvement.
The updated [implementation
plan](docs/personalized-ai-race-engineer-implementation-plan.docx) places that evaluation
in M4, after the end-to-end M3 speech milestone.
