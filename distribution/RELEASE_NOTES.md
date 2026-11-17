# Pitward 0.2.0

This release adds lap timing and race-progress data, deterministic fuel estimates,
and a source-only Ministral conversation preview. **The Windows application still
uses LFM3-1.7B. Ministral is not yet connected to the live desktop radio.**

## New telemetry and calculations

- iRacing capture now includes completed laps, last/best lap durations, the race
  lap limit, remaining race laps and the session clock. Missing channels and
  unlimited-session sentinels remain unknown rather than becoming fake readings.
- New core tools read last/best lap time, compare the last lap with the best,
  distinguish lap counts and estimate fuel for the full scheduled distance.
- Fuel-to-finish includes an explicit one-lap reserve at observed consumption.
  This is a conservative estimate, not a guaranteed finish or pit strategy.
  Timed-only races do not receive an invented exact remaining lap count or finish
  fuel prediction.
- Existing recordings remain readable; new fields require a new capture.

## Conversation preview for source users

The checked Ministral 3 3B Q4_K_M demo offers fresh native-tool readings, bounded
history and recall of delivered readings. It exposes 14 implemented core
capabilities through 26 read/conversation operations. English is the test focus.

Last/best lap durations use race-radio wording: **92.5 seconds becomes "one thirty
two point five"**. Fuel range explicitly says "laps of fuel". Instantaneous speed
and numerical per-second gap rates are excluded; recent closing movement is
qualitative. Lap differences and gaps keep seconds as their unit.

With the repository development environment installed, explicitly set up the
separate preview weights and then start typed chat:

```powershell
.\.venv\Scripts\python.exe scripts\setup_dialogue_comparison.py --model ministral
.\.venv\Scripts\python.exe scripts\try_ministral.py
```

This starts its own local GPU server and uses synthetic telemetry without audio.
`--backend cpu` selects an explicit CPU trial; CPU racing latency is not established.
Model weights, recordings and local diagnostic reports are not included in Git.

## Existing desktop path

The LFM social prompt now uses the current question and bounded dialogue context
more directly, and supports a bounded repair for invalid social output. Repeated
uses of one verified evidence binding no longer reject an otherwise valid reply.
These changes do not establish general conversation accuracy.

The installer component set, Qwen3-ASR, Chatterbox voice, PTT cues and driver
profiles are unchanged. The installer does not download Ministral or include the
source-only chat scripts. In-place upgrades retain downloaded models and driver
data; uninstalling first is unnecessary.

## Validation and known limitations

- Source: 965 tests passed, two NumPy-dependent audio tests skipped; Ruff and mypy
  passed. Windows packaging checks run separately in the tagged workflow.
- The expanded GPU text replay delivered 135/140 replies at a 348 ms median.
  Completion is not accuracy: remaining failures and misleading reserve/recall
  wording are documented. Timing excludes startup, STT, TTS and racing/VR load.
- The radio-format smoke check delivered eight replies with the requested lap
  pronunciation; a separate last-versus-best recall-selection error remains open.
- Opponent lap pace/fastest-driver comparisons, multi-lap clean-lap history and
  pit projections remain unavailable. Cockpit controls stay driver-operated.
- Turkish conversation is unvalidated. English/Turkish speech support is retained,
  but this release's source checks do not validate actual audio playback.
- Windows 11 x64 remains the packaging target. Builds are unsigned; packaged
  inference uses CPU and may miss racing deadlines. The source CUDA setup is separate.

See [known issues](https://github.com/furkannehir/race-engineer/blob/main/docs/known-issues.md)
and the [telemetry access matrix](https://github.com/furkannehir/race-engineer/blob/main/docs/telemetry-expansion.md).
