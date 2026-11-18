Pitward for Windows
===================

1. Launch Pitward.exe and select Set up local AI components. Review and accept the
   third-party terms when prompted. Setup downloads about 7.0 GB of pinned local AI
   models and the CPU conversation runtime, then verifies every download.
2. Pitward restarts automatically after setup succeeds. If setup fails or is
   cancelled, retry setup to resume the downloads.
3. Select your radio devices and push-to-talk control, then start the engineer.

Conversation uses local Ministral 3 3B Instruct 2512 Q4_K_M (Apache-2.0).
After successful Ministral startup and checksum verification, the obsolete
managed LFM weights are removed, freeing about 1.10 GB. Failed startup retains
LFM. Locked-file cleanup retries later; custom/source models are preserved.

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
  the existing installation to keep settings, profiles and history and reuse
  current models. Obsolete managed weights are removed after successful replacement.
  If you cancel the installation wizard, open Pitward again manually.
  Portable and source launches use the GitHub release page for manual updates.

Pitward stores profiles, settings, logs, and downloaded models under
%LOCALAPPDATA%\Pitward. Upgrading preserves settings/profiles/logs and reuses current models; successful
model migration removes only known obsolete weights. Explicit uninstall
removes downloaded models in components and downloaded installers in updates;
profiles, settings, and logs are retained. A custom PITWARD_DATA_DIR must be
cleaned up manually.

Run PitwardComponents.exe status to check for missing components, or
PitwardComponents.exe verify to re-hash installed model files.

Source and releases: https://github.com/furkannehir/race-engineer
