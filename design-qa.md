# Control panel design QA

## Comparison target

- Source visual truth: `docs/design/control-panel/selected-radio-desk.png`, the second
  displayed ideation result selected by the driver.
- Rendered implementation: `docs/design/control-panel/implementation.png`.
- Full-view side-by-side evidence: `docs/design/control-panel/comparison.png`.
- Supporting state evidence: `stopped.png`, `muted.png`, `minimum-size.png`,
  `ptt-binding.png`, and `preferences.png` in the same directory.
- Runtime: native PySide6 Windows desktop panel. Browser screenshots and browser-console
  checks do not apply to this surface.

## Normalization and state

- Source pixels: 1499 x 1049, including the generated native title bar.
- Source content crop: 1495 x 995, scaled proportionally to 1040 x 692 for comparison.
- Implementation pixels and logical viewport: 1040 x 700 at Qt scale factor 1.
- Implementation capture: direct `QWidget.grab()` through Qt's offscreen platform.
- Compared state: engineer running, radio ready, iRacing connected, local speech ready,
  F8 PTT, system-default audio, 80% volume, automatic English/Turkish reply language,
  and master mute off.
- Native title bars were excluded from both sides because the implementation delegates
  that chrome to Windows. The preview-only notice in the implementation is not present in
  a normal launch and does not change the production layout.

## Full-view evidence

The final comparison preserves the source's 40/60 two-column hierarchy, vertical divider,
large operational status, F8 keycap, two readiness rows, master mute, settings form, and
persistent tray action. Controls remain fully visible at the 1000 x 690 minimum window,
with no clipping, overlap, or broken wrapping. The stopped and muted captures communicate
their state without relying on color alone.

## Focused evidence

- `preferences.png` verifies readable labels, unchecked/checked states, explanatory copy,
  and Save/Cancel actions at the dialog's natural size.
- `ptt-binding.png` verifies the press-to-bind surface, supported device classes,
  digital-only safety note, selected-control state, and explicit Save/Cancel actions.
- `minimum-size.png` verifies the densest dynamic heading (`Thinking...`) with the complete
  form and persistent actions still visible.
- `muted.png` verifies both the explicit `Audio muted` heading and the selected master
  toggle. No focused crop is needed because these controls are readable at 1:1 in their
  complete state captures.

## Fidelity surfaces

- Fonts and typography: Segoe UI regular, semibold, and bold are loaded for deterministic
  captures. Heading/body hierarchy, optical weight, line height, and wrapping match the
  native-utility character of the source; no text truncates.
- Spacing and layout rhythm: column proportions, margins, section breaks, form alignment,
  keycap scale, footer separation, and action placement match the source. The implementation
  uses a slightly tighter native control density without changing information hierarchy.
- Colors and tokens: charcoal surfaces, warm-white foreground, subdued secondary text,
  cyan interaction/status accents, green running status, and subtle borders map cleanly to
  the source with accessible redundant status labels.
- Image and icon fidelity: the interface contains no photographic or decorative raster
  assets. Status, headset, and toggle icons come from one Material Design icon library,
  loaded privately as application fonts; no placeholder, emoji, handcrafted SVG, or CSS-art
  substitutions are used.
- Copy and content: source labels are preserved. The additional output-device limitation
  note accurately exposes the existing SAPI/Piper behavior. Preview-only copy is explicitly
  labeled and absent in production.
- States and accessibility: automated tests exercise stopped, starting/running, telemetry
  readiness, muted, failure, microphone test, voice test, settings lock, keyboard/mouse/
  wheel PTT dialog,
  preferences, device reordering, cancellation, and preview isolation. Controls have
  accessible names, visible focus borders, keyboard-operable native behavior, and text
  equivalents for color states.

## Findings

No actionable P0, P1, or P2 visual or interaction mismatches remain.

- [P3] Real Windows display-scaling polish remains a live shakedown item.
  Location: complete panel on the driver's racing display.
  Evidence: deterministic captures cover scale factor 1 and the minimum window, while the
  driver's actual DPI, taskbar, and multi-monitor arrangement have not been captured.
  Impact: unusually high display scaling could reveal minor native-control spacing issues.
  Follow-up: verify the visible panel at the driver's normal Windows scale during the first
  stationary practice-session test and adjust only if a concrete issue appears.

## Comparison history

1. Initial capture was blocked because Qt's offscreen platform did not discover installed
   Windows fonts, leaving text absent. The capture harness now explicitly loads the installed
   Segoe UI family; `comparison-pass1.png` preserves the failed evidence.
2. The first valid comparison found moderate density drift and an oversized double-groove
   volume control. Typography/keycap sizing, form alignment, and slider styling were
   corrected; `comparison-pass2.png` preserves that intermediate evidence.
3. The final comparison shows the corrected hierarchy and single-groove slider with no
   remaining actionable P0/P1/P2 findings.

## Implementation checklist

- [x] Match selected two-column hierarchy and dark visual system.
- [x] Implement functional start/stop, audio tests, press-to-bind PTT, mute, preferences,
  and tray actions.
- [x] Verify stopped, ready, processing, muted, failure, dialog, and minimum-size states.
- [x] Verify settings persistence, device identity, model ownership, and graceful cleanup.
- [x] Pass automated tests, lint, static typing, and whitespace checks.
- [ ] Complete the driver's real-device stationary practice-session shakedown.

## Residual test gap

No real microphone recording, wheel-button hold, Piper/SAPI playback, live model inference,
iRacing telemetry, or long-running tray session was invoked during visual QA. Those require
the driver's local hardware and simulator session and are documented as the next shakedown,
not as a remaining code/visual defect.

final result: passed
