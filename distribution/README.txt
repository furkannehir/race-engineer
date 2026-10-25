Pitward for Windows
===================

1. Launch Pitward.exe and select Set up local AI components. Review and accept the
   third-party terms when prompted. Setup downloads about 7.3 GB of pinned local AI
   models and the CPU conversation runtime, then verifies every download.
2. Restart Pitward after the setup window closes.
3. Select your radio devices and push-to-talk control, then start the engineer.

All speech uses Chatterbox Multilingual's built-in voice; no reference recording
or installed system voice is needed. CPU speech may not meet racing deadlines.
PTT beep/static WAV cues remain separate and unchanged.

Advanced/manual setup:
  PitwardComponents.exe install --accept-third-party-licenses

Pitward stores profiles, settings, logs, and downloaded models under
%LOCALAPPDATA%\Pitward. Replacing or uninstalling the application does not remove them.

Run PitwardComponents.exe status to check for missing components, or
PitwardComponents.exe verify to re-hash installed model files.

Source and releases: https://github.com/furkannehir/race-engineer
