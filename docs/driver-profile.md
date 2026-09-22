# Local driver profile and explicit preferences

The first SQLite driver-memory slice is implemented behind
`SqliteDriverProfileRepository`. It creates a versioned local database at
`paths.database_path` (`data/race_engineer.sqlite3` by default) and provides one safe
default profile on first use. SQLite comes from Python's standard library; there is no
service, account, or cloud fallback.

## Current data boundary

Schema v1 stores:

- manually assigned profile ID and display name;
- creation/update timestamps and the single default-profile marker;
- the current value of three supported communication preferences; and
- the explicit `PreferenceCommand` that produced each change, including source and UTC
  timestamp.

The supported settings are `announce_position_changes`, `announce_pit_transitions`, and
`reply_language` (`auto`, `en`, or `tr`). They are driver-scoped. Session and global
commands are rejected until their precedence and lifetime semantics are implemented.
Safety/critical calls are intentionally not configurable through this store.

The database does **not** contain telemetry, microphone audio, transcripts, prompts,
model replies, inferred personality traits, race results, or driver names learned from
iRacing. A reset writes explicit safe-default commands so the audit trail remains intact;
it does not erase history or silently infer new preferences.

## Headless controls

From the repository root:

```powershell
race-engineer profile --config config/default.toml show
race-engineer profile --config config/default.toml rename "Furkan"
race-engineer profile --config config/default.toml set reply_language tr
race-engineer profile --config config/default.toml set announce_position_changes false
race-engineer profile --config config/default.toml reset
```

Boolean values accept only `true` or `false`. Every successful command prints the current
profile and complete effective preference set as JSON. The control panel uses this same
default profile: its language selector writes `reply_language`, and its preferences dialog
writes the two announcement switches with `ui` provenance. It refreshes the profile when
the dialog opens and immediately before Start, so CLI changes are visible predictably.

Machine-local PTT, audio device and volume values are deliberately separate in
`data/control-panel.json` version 3. When an older panel file is opened, its three legacy
preference values are imported once only if the SQLite profile has no explicit values.
Existing SQLite preferences always win, and the upgraded JSON no longer contains driver
preferences.

## Storage guarantees

Migration 1 creates `schema_migrations`, `driver_profiles`, `preference_commands`, and
`communication_preferences` as SQLite STRICT tables. Foreign keys are enabled on every
connection, writes use explicit transactions, a busy timeout protects short contention,
and WAL mode supports later runtime readers without relaxing validation.

Only one profile can be marked default. Command IDs are idempotent: an exact retry is safe,
while reuse with different content is rejected. Older commands cannot overwrite newer
values. An unknown future migration or an unrelated SQLite database is rejected without
being adopted or modified.

Migration 2 adds the separate, content-free `policy_decisions` and `radio_outcomes`
tables described in [decision-history.md](decision-history.md). Removing an expired decision
cascade-removes its outcome; profile deletion remains the ownership boundary for both
preferences and history.

Automated tests cover first-run migration, restart persistence, profile separation,
default selection, validation, audit provenance, idempotency, stale/conflicting commands,
reset behavior, future schemas, unrelated databases, headless commands, panel migration,
runtime handoff, decision/outcome linkage, retention, and session summaries.
