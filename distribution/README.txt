Pitward for Windows
===================

1. Launch Pitward.exe and select Set up local AI components. Review and accept the
   third-party terms when prompted. Setup downloads about 5.9 GB of pinned local AI
   models and the CPU conversation runtime, then verifies every download.
2. Pitward restarts automatically after setup succeeds. If setup fails or is
   cancelled, retry setup to resume the downloads.
3. Select your radio devices and push-to-talk control, then start the engineer.

All speech uses Chatterbox Multilingual's built-in voice; no reference recording
or installed system voice is needed. CPU speech may not meet racing deadlines.
PTT beep/static WAV cues remain separate and unchanged.

Advanced/manual setup:
  PitwardComponents.exe install --accept-third-party-licenses
  Restart Pitward manually after running the downloader directly from a terminal.

Updates:
  Installed builds check GitHub for newer stable releases after startup. Use
  Releases & updates for a manual check or to turn startup checks off.
  Update now downloads and verifies the installer, closes Pitward, and opens the
  normal installation wizard. Later keeps the current app running. Upgrade over
  the existing installation to keep models, settings, profiles, and history.
  If you cancel the installation wizard, open Pitward again manually.
  Portable and source launches use the GitHub release page for manual updates.

Pitward stores profiles, settings, logs, and downloaded models under
%LOCALAPPDATA%\Pitward. Upgrading the application preserves them. Explicit uninstall
removes downloaded models in components and downloaded installers in updates;
profiles, settings, and logs are retained. A custom PITWARD_DATA_DIR must be
cleaned up manually.

Run PitwardComponents.exe status to check for missing components, or
PitwardComponents.exe verify to re-hash installed model files.

Source and releases: https://github.com/furkannehir/race-engineer
