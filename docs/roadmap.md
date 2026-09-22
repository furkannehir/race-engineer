# Working roadmap

Updated 2026-09-22. This is the current progress/backlog companion to the original
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
| M2: Strict policy | Core policy, SQLite-backed preferences, and durable content-free decision/outcome history implemented | Broader scenario and real-race validation remain |
| M3: Speech / conversational increment | Automatic calls and local English/Turkish live voice conversation implemented; driver accepts the current prototype | Deferred quality upgrades; measured latency/resource and extended live validation |
| M4: Adaptive personalization | SQLite profile/preferences plus observational policy-decision and radio-outcome history implemented; no inferred learning yet | Define feedback signals, baseline ranker, shadow evaluation |
| M5: Driver controls / UI | Basic PySide6 radio desk implemented early: lifecycle, audio checks, press-to-bind keyboard/mouse/wheel PTT, mute, tray and driver preferences | Real-race panel and wheel-button shakedown; voice preference commands, resource controls, session review |
| M6: Second simulator | Not started | Another adapter and cross-simulator contract validation |

## Current priority: control-panel shakedown

The driver selected the second radio-desk design and asked to prioritize testing the
existing engineer in a real race before learning or voice-based settings. The basic
[control panel](control-panel.md) is implemented and tested with fake runtime/hardware.
The driver reports that the Thrustmaster TX binding and combined radio work well enough to
continue, with noticeable but currently acceptable layered speech latency. Extended-race
responsiveness, tray operation, timing breakdowns, and endurance remain validation tasks.
Existing defects are not resolved by the UI.

## Current priority: feedback signals and shadow evaluation

The versioned SQLite profile repository, validated preference vocabulary, audit commands,
safe reset, headless controls, panel integration, runtime handoff, and content-free durable
decision/outcome history are now implemented. Panel JSON version 3 retains only machine
controls; legacy driver preferences are safely imported once. History has a 30-day default
retention boundary and does not influence live behavior.

Recommended implementation order:

1. **Done:** add versioned SQLite migrations and repositories for a local driver profile
   and explicit communication preferences, separate from inferred preferences.
2. **Done:** define the initial setting vocabulary, safe defaults, driver-only scope, reset
   behavior, headless controls, provenance and restart persistence.
3. **Done:** route the panel preference dialog through the repository and apply the
   selected default profile predictably to communication behavior while preserving
   critical-call rules. Do not grant the conversational model unrestricted database
   writes.
4. **Done:** persist privacy-conscious policy decisions and radio outcomes with identifiers and
   reasons, so later feedback and ranker evaluations can be traced to actual behavior.
   Do not start storing microphone audio, prompts, or transcripts by default.
5. Use that foundation to define feedback signals and evaluate a local ranker in shadow
   mode before allowing it to influence audible output.

Step 5 is the proposed next implementation slice, not already implemented work. Feedback
semantics and shadow-evaluation acceptance criteria must be specified before implementation.
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
