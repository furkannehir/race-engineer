# Working roadmap

Updated 2026-09-21. This is the current progress/backlog companion to the original
[implementation plan](personalized-ai-race-engineer-implementation-plan.docx), not a claim
that every original milestone exit criterion has been formally validated.

## Current checkpoint

The driver has tried the combined live iRacing prototype and considers it sufficient to
continue development. Short-input recognition, voice quality, and conversational reactions
are deferred under [STT-001, TTS-001, and CONV-004](known-issues.md).

This is qualitative functional acceptance. There are no new measured accuracy, latency,
frame-time, or endurance results accompanying the feedback. Existing defects remain open.

| Milestone | Current position | Remaining/deferred work |
| --- | --- | --- |
| M0: Foundation | Contracts, configuration, logging, fixtures, and tests implemented | Maintain compatibility as new domains are added |
| M1: iRacing telemetry | Live reading, normalization, recording, reconnect handling implemented | IR-001 blue-flag investigation; broader scenario validation |
| M2: Strict policy | Core policy implemented; panel now persists/applies the existing position and pit call switches | General driver-profile/scoped preference service remains missing |
| M3: Speech / conversational increment | Automatic calls and local English/Turkish live voice conversation implemented; driver accepts the current prototype | Deferred quality upgrades; measured latency/resource and extended live validation |
| M4: Adaptive personalization | Not implemented | Local memory, trustworthy outcome/feedback data, baseline ranker, shadow evaluation |
| M5: Driver controls / UI | Basic PySide6 radio desk implemented early: lifecycle, audio checks, press-to-bind keyboard/mouse/wheel PTT, mute, tray and local defaults | Real-race panel and wheel-button shakedown; voice preference commands, resource controls, session review |
| M6: Second simulator | Not started | Another adapter and cross-simulator contract validation |

## Current priority: control-panel shakedown

The driver selected the second radio-desk design and asked to prioritize testing the
existing engineer in a real race before learning or voice-based settings. The basic
[control panel](control-panel.md) is implemented and tested with fake runtime/hardware.
Actual mic/voice checks, in-game responsiveness, tray operation during a session and
endurance remain driver validation tasks. The press-to-bind layer detects the connected
Thrustmaster TX and is covered with simulated devices, but an actual held wheel-button
capture still needs the driver's shakedown. Existing defects are not resolved by the UI.

## After the shakedown: local memory and explicit preferences

Begin the prerequisites for M4, and close the durable-preference gap in M2. The code has
a configured SQLite path and a `PreferenceCommand` contract, but the `memory` package is
still a placeholder. The panel's small local JSON defaults, TOML policy settings and replay
recordings are not a full driver profile or a queryable decision/outcome history.

Recommended implementation order:

1. Add versioned SQLite migrations and repositories for a local driver profile and explicit
   communication preferences, separate from any future inferred preferences.
2. Define validated preference settings, defaults, session-versus-driver scope, and reset
   behavior; provide headless controls and prove persistence across restarts.
3. Apply explicit preferences predictably to communication behavior while preserving
   critical-call rules. Do not grant the conversational model unrestricted database writes.
4. Persist privacy-conscious policy decisions and radio outcomes with identifiers and
   reasons, so later feedback and ranker evaluations can be traced to actual behavior.
   Do not start storing microphone audio, prompts, or transcripts by default.
5. Use that foundation to define feedback signals and evaluate a local ranker in shadow
   mode before allowing it to influence audible output.

This is the proposed next implementation slice, not already implemented work. Preference
vocabulary and storage/retention details should be specified with its implementation.
Natural-language preference commands and the existing UI should share that validated
service later; contextual social responses remain the separate deferred CONV-004 feature.

## Constraints carried forward

- Keep speech recognition, conversation, synthesis, and personalization local by default.
- The original plan's Jev evaluation remains optional/deferred. The later entirely-local
  decision does not authorize hosted ranking, data uploads, or remote shadow requests.
  Any such evaluation needs a separate explicit opt-in.
- Preserve critical deterministic calls and factual grounding; quality/personality upgrades
  must not weaken them.
- Keep all three reported upgrades in the backlog rather than tuning or replacing engines
  as part of the next storage/preferences task.
- Do not treat the driver's prototype acceptance as resolution of the blue-flag defect or
  the existing conversation interpretation failures.
