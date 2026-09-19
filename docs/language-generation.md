# M3 deterministic language generation

M3 turns an approved `SpeechIntent` into an `Utterance`. It is a wording boundary, not a
second policy engine: the adapter may phrase facts supplied by policy, but it may not add
new race facts, recommendations, or urgency.

## Initial adapter

The default `deterministic` adapter provides short English templates for:

- formation, green, caution, and checkered phases;
- green, yellow, red, blue, white, black, and checkered flags;
- position changes;
- pit entry and exit; and
- remaining fuel expressed as laps or liters.

Policy-provided `critical_template` wording takes precedence for safety calls. The legacy
`message` fact is supported only as already-approved wording. Every generated utterance
records the adapter name, adapter version, and template identifier without putting its text
in normal logs.

The adapter rejects unsupported languages, unknown fact shapes, empty templates, and text
that exceeds the intent's word limit. The live pipeline also verifies that the returned
utterance belongs to the input intent. A failed utterance is logged and skipped while
telemetry capture and later policy evaluation continue.

## Future model boundary

`FallbackLanguageGenerator` can wrap a future model-backed generator with a deadline and
contract validation. Failure, timeout, or invalid output activates a second
`LanguageGenerator`, intended to be this deterministic adapter. The default configuration
does not yet enable a model, so live wording is reproducible and requires no network call.

## Replay acceptance

Fixture version 3 adds `expected_utterances.jsonl`. Reproduce the initial acceptance
fixture with:

```powershell
race-engineer replay-language fixtures/synthetic/m3_green_flag `
  --config config/default.toml
```

`expected_utterances_match` must be `true`. A live recording made with `read-iracing
--output` can be inspected and replayed with the same command.
