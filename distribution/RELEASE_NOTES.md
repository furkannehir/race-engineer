# Pitward 0.1.0

The first Pitward release brings local race assistance to iRacing on Windows.

- Live iRacing telemetry, deterministic automatic radio calls, and English LFM conversations.
- Local Qwen3-ASR push-to-talk and Chatterbox spoken replies.
- A native desktop panel with microphone/output selection, radio checks, keyboard,
  mouse and wheel bindings, mute and tray controls.
- Local driver preferences and decision history.
- A portable ZIP, Windows installer, and SHA-256 checksums.

## Getting started

Install the Windows x64 app, or extract the portable ZIP and open `Pitward.exe`.
Choose **Set up local AI components**, review the third-party terms, and allow about
5.9 GB of verified downloads to finish. Restart the app, select your radio devices,
bind push-to-talk, and perform a voice check before starting your engineer.

Models, profiles, settings, and logs stay under `%LOCALAPPDATA%\Pitward` and are
preserved when you replace or uninstall the application.

## Early-release limitations

- Windows 11 x64 is the clean-machine acceptance target.
- Packaged inference and speech workers use CPU; race-time speech latency is not yet
  established and may be too slow on some PCs. Source CUDA setup is separate.
- English conversation is the MVP. Turkish conversation remains unvalidated.
- Builds are unsigned; Windows may display a SmartScreen warning.
- An internet connection is required for the initial component downloads.

Source, third-party notices, and known issues are available in the
[repository](https://github.com/furkannehir/race-engineer).
