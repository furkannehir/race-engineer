# Local decision and radio history

SQLite schema v2 adds a privacy-safe observational history for deterministic automatic
calls. It is enabled by default and does not change policy priorities, suppression,
wording, queue order, or playback behavior.

## What is stored

Each policy decision stores the selected driver-profile ID, decision/candidate/session
identifiers, source sequence, UTC decision time, approved or suppressed outcome, reason,
priority, and the approved intent ID when present.

Each terminal automatic-radio outcome stores its linked decision and intent IDs, one of
`completed`, `cancelled`, `expired`, or `failed`, an operational reason code when relevant,
and available start/finish timestamps. Cancellation reason codes distinguish conditions
such as mute, interruption, invalidation, queue pressure, and shutdown.

The tables cannot store telemetry facts, utterance text, microphone audio, transcripts,
prompts, model replies, or driver names learned from iRacing. Conversational replies are
not included because they are not strict-policy decisions. This slice records evidence;
it does not infer preferences, score the driver, or train/rank messages.

## Retention

`[history] retention_days = 30` in `config/default.toml` is the default rolling retention
period. Expired decisions and their linked outcomes are pruned when durable history opens
for a live run. You can apply the same cutoff manually:

```powershell
.\.venv\Scripts\python.exe -m race_engineer history --config config\default.toml prune
```

Set `[history] enabled = false` to stop new runtime history writes. Existing rows remain
until a later prune; disabling history is not an implicit deletion command.

## Inspect recent sessions

```powershell
.\.venv\Scripts\python.exe -m race_engineer history --config config\default.toml recent
.\.venv\Scripts\python.exe -m race_engineer history --config config\default.toml recent --limit 5
```

The output contains per-session counts for approved/suppressed decisions and terminal
radio statuses. It contains no message content. Limits must be between 1 and 100.

History initialization and individual writes are failure-isolated: runtime writers use a
short lock timeout, so a locked or unavailable database produces a bounded diagnostic log
and the live engineer continues. Idempotent retries are accepted; conflicting reuse of a
decision or intent ID fails closed in storage without affecting the radio pipeline.
