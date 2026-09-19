# Replay fixture format v1

A fixture is a directory containing `manifest.json` and one or more JSON Lines files.
Paths in a manifest are relative to the fixture directory and may not escape it.

```text
fixture-name/
  manifest.json
  frames.jsonl
  expected_events.jsonl
  expected_intents.jsonl
```

The manifest declares `fixture_version`, a stable fixture identifier, a description, and
the paths of its streams. `frames_file` is required; expected streams are optional so a
fixture can target normalization, event derivation, policy behavior, or the complete
headless pipeline.

Each non-empty JSONL line is exactly one versioned contract. Ordering is significant.
Files are UTF-8 and deterministic snapshots use compact JSON with sorted keys. Real driver
names and unrelated telemetry must be removed before a recorded session is committed.

Fixture format versions and individual contract schema versions evolve independently.
Readers reject unsupported versions rather than guessing.

Simulator adapters may place an additional privacy-safe source stream beside these files.
For example, the iRacing M1 fixture contains `raw_samples.jsonl`; its manifest points to
the normalized frames and derived events that the source stream must reproduce. Source
streams have their own schema versions and must omit names, secrets, and unrelated SDK
fields.

The synthetic M2 fixture includes normalized frames, derived events, and expected speech
intents. Replaying it with a fixed policy configuration must reproduce the same intents;
policy-decision reasons are asserted alongside the fixture in automated tests.
