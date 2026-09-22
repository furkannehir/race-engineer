# Desktop radio desk

The selected dark, two-column design is implemented as an optional native PySide6
Windows panel. It wraps the same `voice-iracing` pipeline; it is not another speech queue
or a browser dashboard. This brings the basic M5 controls forward for real-world testing,
before adaptive driver memory or natural-language preference commands.

## Launch

From the repository root, install once:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[desktop]"
```

Then double-click `start-panel.cmd`, or run:

```powershell
.\.venv\Scripts\python.exe -m race_engineer.ui
```

The existing local Qwen conversation model, Qwen3-ASR runtime, and Piper voices must
already be installed. Their existing [setup guides](live-conversation.md#run) still apply.
Opening the window enumerates audio devices but does not start telemetry, inference,
capture, playback, or controller input. Controller detection starts only when you open
the push-to-talk binding dialog. There is no download or cloud fallback.

## First shakedown

1. Select the microphone and output. System default is the safest first choice. Output
   selection affects **Piper replies only**; automatic SAPI calls use Windows' default
   playback device. Set that default to the same headset yourself if needed.
2. Click **Test mic** and speak for three seconds. It reports a level, without saving
   audio. **Test voice** plays a short Piper radio check; Auto uses English for this test.
   Both tests can be cancelled, and cannot overlap a live session.
3. Click **Change** beside PTT, then press the control you want. The panel accepts arbitrary
   keyboard keys, the five standard mouse buttons, and digital buttons or hats/D-pads on
   steering wheels, button boxes, joysticks, and gamepads. Steering, pedals, triggers, and
   other analog axes are deliberately ignored so driving cannot open the microphone. Save
   the detected binding; there is no device-name or button-number hunting.
4. Click **Start engineer**. The panel reuses a matching local conversation server or
   starts the runtime selected in `[conversation.runtime]` on literal `127.0.0.1`. The
   default remains the downloaded CPU runtime; accelerated setup and controls are planned.
   Wait for both iRacing and
   speech readiness. A green iRacing indicator requires accepted, fresh telemetry, not
   just an SDK connection. A stopped panel does not monitor iRacing.
5. Start stationary in practice. Check English/Turkish questions and automatic calls
   before committing to a race. Then minimize to the tray and hold PTT to talk normally.
6. Restore from the tray to stop. Closing a running panel asks to stop and quit, then
   waits for cleanup. A reused external model server is left running; a model server
   started by the panel is stopped with its session.

Do not run `voice-iracing`, `read-iracing` or `voice-replay` alongside the panel. A local
lock prevents two panels from the same checkout; it does not detect independently
launched CLI processes or another checkout.

**Mute all audio includes critical calls.** It cancels current speech, discards queued
speech and drops new calls/replies while muted. Unmuting does not replay old messages.
Capture and telemetry remain enabled. Mute is deliberately not remembered across app
launches. Volume applies to both engines at the next Start, not Windows system volume.

Device, PTT, language, volume and policy controls are locked while running. Stop first
to change them. Escape remains available to iRacing; only the CLI's capture loop treats
Escape as exit. Device identities are checked before starting to avoid silently using a
different audio device after an index change. PTT controller bindings store the device
name, SDL GUID, device-index hint, control type, and control number. A uniquely reindexed
device is recovered; a missing or ambiguous device fails closed and asks for a rebind.
Reconnect changed controller hardware before opening the binding dialog again.

## Preferences and privacy

The current preference dialog exposes only supported policy switches: position-change
and pit-entry/exit announcements. Flags and critical policy rules are unchanged. There
is no invented Quiet/Talkative or answer-style setting that the backend cannot apply.

Driver communication preferences are validated and atomically written to the default
profile in ignored `data/race_engineer.sqlite3`. The language selector saves immediately;
the dialog saves its two switches only when **Save** is pressed. All three take effect on
the next Start, when the panel refreshes the profile before creating the runtime worker.
Headless `profile set` changes are therefore picked up without reopening the panel.

Machine-local PTT, audio-device and volume settings remain in
`data/control-panel.json` version 3. On the first launch after upgrading from version 1 or
2, old communication preferences are imported into SQLite only when the profile has no
explicit preferences; an existing profile always wins. The JSON is then rewritten without
driver preferences. The base TOML stays untouched.

Use `race-engineer profile --config config/default.toml reset` to restore the driver's
three safe defaults while retaining their audit history. Moving `control-panel.json` aside
resets only machine controls. Corrupt settings or incompatible profile databases are
reported and left intact rather than silently overwritten. Session-scoped preferences,
learning, voice-based edits and broader personality controls remain deferred.

Bounded diagnostic logs live in ignored `logs/control-panel.jsonl` (2 MB plus two rotated
files). They do not persist microphone audio, question/reply text or model prompts. The
terminal launch can still print conversational text, as the existing CLI does; the
double-click/pythonw launch discards console output. The panel does not record telemetry
fixtures; use the existing CLI recording option when specifically collecting one.
Live panel sessions do retain content-free policy decisions and terminal automatic-radio
statuses for 30 days by default. They contain no spoken text or conversational questions;
see [decision-history.md](decision-history.md).

## Validation and limits

Automated tests cover SQLite-backed preferences and legacy import, audio and PTT device
identity, keyboard/mouse/wheel capture, digital hat directions, GUI controls/dialogs,
preview isolation, cancellation,
model ownership, telemetry readiness and master mute. Native
Qt screenshots and the selected source are in `docs/design/control-panel/`, with the
visual comparison in the root `design-qa.md`. No real microphone recording, model speech,
or competitive-race session was performed as part of implementation verification.

For a hardware-free layout preview:

```powershell
.\.venv\Scripts\python.exe -m race_engineer.ui --preview
```

The preview is clearly labeled and never starts hardware. Screenshots are generated with
`scripts/check_control_panel.py`; this renders real widgets through Qt's offscreen platform.

The blue-flag bug, short-input recognition, voice quality and conversation issues remain
open in [known-issues.md](known-issues.md). This panel is a shakedown tool, not a claim of
race-certified accuracy, latency or endurance. There is no installer or auto-updater yet.

Implementation references: Qt's [thread/signals documentation](https://doc.qt.io/qtforpython-6.10/PySide6/QtCore/QThread.html)
and [system-tray API](https://doc.qt.io/qtforpython-6/PySide6/QtWidgets/QSystemTrayIcon.html);
[QtAwesome's Material Design icon library](https://github.com/spyder-ide/qtawesome);
[pygame-ce's raw joystick API](https://pyga.me/docs/ref/joystick.html) and SDL's
[background joystick-events hint](https://wiki.libsdl.org/SDL3/SDL_HINT_JOYSTICK_ALLOW_BACKGROUND_EVENTS).
Icons are loaded privately as application fonts; the panel does not install system fonts.
