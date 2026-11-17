# Engineer control inventory for iRacing

Recorded 17 November 2026. Companion to the
[telemetry expansion inventory](telemetry-expansion.md): this page catalogs
actions Pitward could perform at the driver's request, rather than facts it
could read or calculate. This is a researched backlog, not an implemented voice
control feature. No simulator settings were changed during this investigation.

## Implementation and Ministral access snapshot

Source audit: **17 November 2026**. None of the catalogued simulator write
actions is implemented in Pitward's adapter or exposed as a checked Ministral
tool. `pyirsdk` providing a command is dependency support, not an integrated
Pitward action adapter. No voice-to-pit execution is available yet.

| Inventory entries | Pitward action-adapter status | Checked Ministral execution |
| --- | --- | --- |
| CTL-01 through CTL-11: pit services | Not implemented; SDK/macro routes are researched only | No tools |
| CTL-12 through CTL-15: car-specific next-stop adjustments | Not implemented; require per-car validation | No tools |
| CTL-16 through CTL-27: cockpit adjustments | Excluded: driver-only | Excluded; no execution tools |
| CTL-28 through CTL-35: vehicle operations/recovery | Excluded: driver-only | Excluded; no execution tools |
| CTL-36 through CTL-38: optional UI/radio/view utilities | Not implemented | No tools |
| CTL-39: force-feedback changes | Excluded: driver-only | Excluded; no execution tools |
| CTL-40 through CTL-45: optional display/recording/replay/practice utilities | Not implemented | No tools |
| CTL-46: garage setup | Not implemented; separate conditional extension | No tools |
| CTL-47: hosted administration | Not implemented; outside the proposed MVP | No tools |

Readback is a separate gap: the telemetry adapter reads current fuel and own
pit-lane state, but not requested fuel addition, selected tyre-change flags,
compound, cold pressures, tear-off/fast-repair selections or service status.
Those fields must be added before claiming verified pit-request state. Replay
detection in the telemetry adapter is likewise not replay-control execution.

For all reads, calculations and the expanded 26-operation Ministral menu, see the
[telemetry implementation matrix](telemetry-expansion.md#implementation-and-ministral-access-snapshot).
The checked prototype has no action tool. Its normalized reads and existing core
calculations operate on fixture/recorded data rather than the live iRacing stream.
The agreed driver-only exclusions below remain in force.

## Agreed control scope, 17 November 2026

The driver decided that cockpit adjustments belong to the driver. This is a
product boundary, not a deferred implementation task:

- **Pitward:** prepare permitted next-stop fuel, tyres, compound, cold pressures,
  cleaning/tear-off and fast-repair requests, plus verified car-specific changes
  performed by the pit crew at a stop.
- **Driver:** operate immediate cockpit settings and vehicle controls, including
  brake bias/migration, ABS, TC, engine/differential/hybrid settings, live ARBs,
  weight jacker, limiter, ignition, wipers, lights and boost/DRS. Pitward will not
  execute these through SDK commands, macros, mapped inputs or UI navigation.
- **Read and advise:** cockpit telemetry may still be read, explained and used
  for recommendations. Changing the setting remains the driver's action.

The excluded controls below stay as technical reference, not as future voice
control candidates. Utility features such as replay or display navigation are
separate optional ideas; this decision does not start their implementation.

## Integration routes and availability

- **SDK:** a dedicated broadcast command exposed by the installed `pyirsdk`
  1.3.6 binding. The command's existence is verified in its source, not tested
  against a running simulator in this investigation.
- **Macro:** an officially documented iRacing command or configured chat macro.
  The SDK can launch numbered macros, but does not expose an arbitrary chat-text
  setter. Dynamic text would need an additional input/UI transport.
- **Mapped input:** a configured driver-side key/button or a car-specific
  black-box control. This needs an input adapter, known bindings and simulator
  focus/context. It is not an SDK setter.
- **Conditional UI:** a possible car/session-specific adjustment whose exact
  control, limits and execution route must be validated before advertising it.

Reading `dc*` (in-car) or `dp*` (pit request) telemetry does not make that field
writable. There is no generic `set_telemetry(name, value)` command in this binding.

The dedicated SDK command inventory below is finite. The car-specific inventory
is a set of control families: the available options, steps, names and rules vary
by car, track configuration, session and simulator version. A per-car profile and
actual-session verification are needed before claiming complete control of a car.
Fixed-setup restrictions also need checking; do not assume every setting is
available or that every fixed race disables every cockpit adjustment.
Technical availability never overrides the agreed driver-only boundary.

iRacing's remote **Crew Chief/Spotter** role can change pit services and pit-stop
adjustments, but cannot change the driver's F8 in-car settings. Local software
could technically operate mapped driver controls, but Pitward excludes that
capability for cockpit adjustments under the agreed scope. See
[official spotting/crew-chief documentation](https://support.iracing.com/support/solutions/articles/31000162971-spotting).

## Pit service requests

These normally prepare the next stop; selecting a service does not mean it has
already been performed. Availability, tyre-set limits, refuelling and compound
rules must come from the active session/car. Sources: installed SDK command
definitions, [official pit macros](https://support.iracing.com/support/solutions/articles/31000170165-pit-macros-chat-commands)
and [AutoFuel behavior](https://support.iracing.com/support/solutions/articles/31000169381-how-to-use-autofuel).

| ID | Change | Route | Important distinction |
| --- | --- | --- | --- |
| CTL-01 | Enable refuelling and set fuel to add | SDK `fuel`; macro `#fuel` | This is the requested addition, not a write to current tank contents. |
| CTL-02 | Cancel refuelling | SDK `clear_fuel`; macro inverse | Use the explicit clear operation; SDK fuel argument zero means retain the existing requested amount. |
| CTL-03 | Enable/disable AutoFuel and adjust lap margin | Macro `#autofuel`; mapped controls | No dedicated AutoFuel mode in the installed `PitCommandMode`. AutoFuel requires simulator fuel data and an eligible race; manual amount changes disable it. |
| CTL-04 | Select/cancel tyre replacement per LF/RF/LR/RR | SDK `lf/rf/lr/rr`; macros and mapped controls | Some cars enforce side/set replacement. Individual cancellation uses macro/UI; this binding has no per-corner SDK clear mode. |
| CTL-05 | Replace all, left-side or right-side tyres | Compose individual SDK requests or use `#t/#l/#r` | Group operations must respect the car's replacement rules. |
| CTL-06 | Cancel all tyre changes | SDK `clear_tires`; macro `#cleartires` | Preserve unrelated services. |
| CTL-07 | Set next-stop cold pressure per tyre | SDK tyre request with pressure; tyre macros | SDK pressure is kPa and zero retains the requested pressure. A tyre request also selects replacement; preserve/verify the intended change flags. |
| CTL-08 | Select the next tyre compound, including dry/wet where available | SDK `tc`; macro `#tc` | Resolve the active car's compound choices. Macro numbering is ordered and starts at one; do not assume the same numerical convention for SDK/telemetry or a universal wet-compound ID. Verify the mapping. |
| CTL-09 | Request/cancel windshield cleaning or tear-off at the stop | SDK `ws/clear_ws`; macros | Distinct from a driver's immediate visor tear-off control. |
| CTL-10 | Request/cancel fast repair | SDK `fr/clear_fr`; macros | Only when the car/session permits it and allocation remains. This is not a generic repair-all or damage-percentage setter. |
| CTL-11 | Clear all pit-service checkboxes | SDK `clear`; macro `#clear` | Does not restore every chassis setting or constitute a complete pit-plan reset. |

The installed binding exposes all 13 pit modes:
`clear`, `ws`, `fuel`, `lf`, `rf`, `lr`, `rr`, `clear_tires`, `fr`,
`clear_ws`, `clear_fr`, `clear_fuel`, `tc`.
Its source states pit commands operate when the driver is in the car.

## Car-specific adjustments queued for a stop

The F7 pit-adjustment surface changes by car. These are profile-dependent
integration candidates, not additional dedicated SDK pit modes. The
[black-box guide](https://support.iracing.com/support/solutions/articles/31000167257-black-box-screen-information-and-controls)
distinguishes next-stop F7 changes from immediate F8 changes.

| ID | Change family | Route and availability |
| --- | --- | --- |
| CTL-12 | Wing angle/front flap or other exposed pit aero adjustment | Mapped/conditional UI; only the aero options offered for that car's pit stop. |
| CTL-13 | Pit-adjustable front/rear anti-roll-bar setting | Mapped/conditional UI; not interchangeable with a car's live ARB controls. |
| CTL-14 | Rear spring-perch/wedge adjustments | Mapped/conditional UI; validate each car's F7 rows and pending-state readback. Do not confuse these with the IR18's live weight jacker. |
| CTL-15 | Other F7 adjustments, potentially shocks/track bar/tape where actually exposed | Conditional UI, unverified per-car candidates. Do not advertise any of these solely because a similarly named garage setting or generic binding exists. |

An unrestricted "enable/cancel optional repairs" action is not established by
the inspected SDK or pit macro list. Reading a repair timer/needed flag is not
an execution route. Any future repair-choice control needs separate simulator
verification; required repairs cannot simply be cancelled by a speech command.

## Driver-only: immediate cockpit adjustments

**Excluded from Pitward control execution (CTL-16 through CTL-27).** The driver
adjusts these directly. Retain the inventory for read-only telemetry and advice;
do not build mapped-input or F8 execution for these settings. The technical
availability notes below describe iRacing controls, not planned Pitward actions.

| ID | Change family | Scope |
| --- | --- | --- |
| CTL-16 | Brake bias forward/rearward; an absolute target where measurable | Mapped input; translate clicks using the active car's step and verify readback. |
| CTL-17 | Brake migration and mapped brake-bias target controls | Car-specific. Other ramping/brake parameters may be garage-only. |
| CTL-18 | ABS setting | Only where fitted, adjustable and permitted. |
| CTL-19 | TC1/TC2, intervention/slip/gain, TC enable or temporary disable | Resolve meanings and toggle behavior by car. |
| CTL-20 | Stability-control/ESC/DSC setting | Only where exposed and mapped; some cars reuse TC control slots. |
| CTL-21 | Engine map, fuel mixture, throttle shape and engine braking | Separate controls/meanings per car; do not treat them as synonyms. |
| CTL-22 | Differential entry/middle/exit or preload adjustment | Only the differential settings explicitly available from the cockpit; many are garage-only. |
| CTL-23 | Front/rear anti-roll bars | Only cars with live adjustment. |
| CTL-24 | Live weight jacker | For the relevant car/track; the IR18 manual restricts its use to oval configurations. |
| CTL-25 | Power-steering assist | Where cockpit-adjustable. |
| CTL-26 | Hybrid deploy/regen level, deploy/recharge mode or trim | Car/build-specific. Do not promise manually adjustable modes on every hybrid or use obsolete deployment assumptions. |
| CTL-27 | Car-specific hybrid cornering setting | Example: Ferrari 499P front hybrid cornering control; validate its actual mapping. |

Sources: [iRacing controls](https://support.iracing.com/support/solutions/articles/31000133517-customizing-sim-hot-key-combinations),
[W12 manual](https://s100.iracing.com/wp-content/uploads/2023/10/UM-Mercedes-AMG-F1-W12-E-Performance-V2.pdf),
[MP4-30 guide](https://ir-core-sites.iracing.com/members/pdfs/MP4-30_guide.pdf),
[Dallara P217 manual](https://s100.iracing.com/wp-content/uploads/2025/03/UM-Dallara-P217-LMP2-V2.pdf),
[IR18 manual](https://s100.iracing.com/wp-content/uploads/2026/01/Dallara-IR18-V3.pdf),
[BMW bias/migration bindings](https://support.iracing.com/support/solutions/articles/31000176726-2025-season-3-patch-2-release-notes-2025-06-21-01-),
[DSC/recharge controls](https://support.iracing.com/support/solutions/articles/31000177173-2025-season-4-patch-1-release-notes-2025-09-12-01-),
[recharge binding](https://support.iracing.com/support/solutions/articles/31000177221-2025-season-4-patch-2-release-notes-2025-09-24-01-)
and [499P cornering control](https://support.iracing.com/support/solutions/articles/31000177376-2025-season-4-patch-3-release-notes-2025-10-16-01-).
Older manuals establish examples, not current universal control mappings.

## Other driver-side operations

**CTL-28 through CTL-35 and CTL-39 remain driver-only**, covering immediate vehicle
operations, recovery and force-feedback changes. They are retained as reference,
not proposed engineer actions. The other rows are separate optional utility ideas.
A toggle/held control is not necessarily an absolute on/off setter. Any future
utility integration must respect the driver-only boundary and known UI context.

| ID | Operation | Route |
| --- | --- | --- |
| CTL-28 | Pit speed limiter | Mapped input; not a dedicated SDK limiter setter. |
| CTL-29 | Ignition and starter | Mapped inputs, where separate controls exist. |
| CTL-30 | Wipers on/off, speed or one sweep | Car-specific mapped controls. |
| CTL-31 | Headlights/flash and exposed lighting controls | Car-specific mapped controls. |
| CTL-32 | Immediate visor tear-off | Mapped input; different from next-stop windshield service. |
| CTL-33 | DRS, push-to-pass, overtake or manual hybrid boost | Mapped input; car/session eligibility and held/pulsed behavior vary. |
| CTL-34 | Launch control, clutch bite point or dash-page/acknowledgment controls | Conditional mapped/UI candidates; validate actual car support. |
| CTL-35 | Enter/exit car or request tow/reset | Mapped context-dependent operation; not a generic SDK teleport/repair command. |
| CTL-36 | Black-box selection/navigation, in-scope pit-request changes and delta display | Optional mapped/UI utility. Never use generic navigation to change excluded cockpit settings. A keypress alone does not verify a change. |
| CTL-37 | Radio channel selection, channel add/remove, mute/unmute, scanning and voice/spotter volume | Official radio macros plus mapped/UI controls. Transmitting messages to others is a separate feature, not a side effect of settings adjustment. |
| CTL-38 | Mirrors, driving/virtual-mirror FOV, driver height/horizon and VR recenter | Mapped/UI controls where available. |
| CTL-39 | FFB maximum-force mapping and other supported FFB adjustments | SDK `ffb_command` for maximum force; mapped/UI for other settings. Simulator mapping is not arbitrary wheel-driver configuration. |
| CTL-40 | HUD/display toggles, UI scale/layout and camera view | Mapped/UI controls; useful conveniences rather than race strategy. |
| CTL-41 | Start/stop/restart iRacing telemetry recording; mark a telemetry event | SDK `telem_command` for recording; mapped input for markers. Restart starts another file, not an application restart. |
| CTL-42 | Replay play/pause, speed, seek, session/lap/frame/incident navigation and camera target/group/state | Dedicated SDK operations plus mapped controls. Erasing a replay tape is also exposed but is not the same as saving a replay. |
| CTL-43 | Screenshot, built-in video capture and video timer | SDK `video_capture`; conditional on simulator capture support/configuration. |
| CTL-44 | Reload all car textures or a selected car's textures | SDK texture-reload operations. |
| CTL-45 | Active Reset save/run and custom-sector boundaries for practice | Mapped controls, only in supported lone-practice sessions; not available as a normal race reset. |

References: [SDK command implementation](https://github.com/kutu/pyirsdk/blob/master/irsdk.py),
[driver shortcuts](https://www.iracing.com/keyboard-shortcuts/),
[custom controls](https://support.iracing.com/support/solutions/articles/31000133517-customizing-sim-hot-key-combinations),
[radio/pit macros](https://support.iracing.com/support/solutions/articles/31000170165-pit-macros-chat-commands)
and [Active Reset](https://support.iracing.com/support/solutions/articles/31000168816-how-to-use-active-reset).
Conditional items without an established mapping remain candidates for per-car
validation, not advertised Pitward capabilities.

## Garage and session administration boundaries

| ID | Area | Availability |
| --- | --- | --- |
| CTL-46 | Garage setup: fuel load, initial tyres/pressures, aero, springs/dampers, ride height, camber/toe/caster, gearing, differential and other car-specific setup parameters; loading/saving setups | Car-specific garage UI, rules and session state. No universal SDK setup editor was found. These are not automatically changes available while racing or at a stop. |
| CTL-47 | Hosted-session administration: session advance/grid timing, next-session track state, admin/chat permissions, removal/DQ, flags/penalties, pace-line/caution/restart and pit opening/closing | Official admin chat commands, only with existing host/admin privileges. Separate from normal driver engineering; no administrative functionality is proposed for the MVP. |

See [official session-admin commands](https://support.iracing.com/support/solutions/articles/31000133518).
Normal driver controls cannot change race weather, erase incident points or
override race-control rules. Likewise, mapped steering, throttle, brake, clutch,
gearshift and handbrake inputs are driving controls, not proposed engineer
settings or an autonomous-driving feature.

## Future action execution boundary

1. Interpret the driver's natural request as a typed action, with target,
   value/unit and next-stop scope for pit requests. A question or complaint is
   not automatically an instruction to change the car. Excluded cockpit actions
   remain driver-operated even if the request explicitly asks Pitward to change one.
2. Apply the agreed scope before resolving the active car/session profile.
   Never expose driver-only actions to an execution tool or generic input fallback.
   For supported pit requests, check current
   state, allowed values, units, available compounds and configured bindings.
   Do not let model-generated text become an arbitrary key sequence or chat command.
3. Use the dedicated SDK operation where available; otherwise use the explicitly
   configured macro/input adapter. Distinguish "add 20 litres" from "finish with
   20 litres" and check the units of pressure and other permitted pit adjustments.
4. Serialize actions and bound retries. Relative increments, toggles and service
   selections need different retry handling; do not blindly repeat a change after
   an uncertain result. Cancel obsolete actions on session changes/disconnection.
5. Verify supported readback before saying a target was achieved. A Windows
   broadcast returning successfully does not establish simulator acceptance. Where
   readback is unavailable, report that verification limit instead of claiming
   the change succeeded. For a next-stop request say "selected for the next stop",
   not "fitted" or "completed".

The first implementation candidate is next-stop fuel, tyre selection/compound,
pressures, tear-off and fast repair using dedicated pit commands. Verified
car-specific adjustments performed at a stop can follow. Brake bias, TC, ABS
and other cockpit execution are excluded from this backlog. This is a proposed
order, not a change to tomorrow's model trial.

## Verification status

- Inspected the installed binding (`VERSION = 1.3.6`), all pit modes and SDK
  broadcast helpers, and searched Pitward's current code for existing execution.
- Cross-checked official iRacing macros, controls/black boxes, crew-chief roles,
  AutoFuel, Active Reset, car manuals and relevant control-update release notes.
- No commands were sent to iRacing, no input bindings were changed, and no live
  command/readback or UI automation was tested. Every control implementation still
  needs simulator verification before being advertised in the application.
