# Windows builds and releases

The version in `src/race_engineer/version.py` sets the Python metadata, executable
metadata, archive and installer names, and the Git tag. A release tag must be exactly
`v<version>`; a mismatch stops the build.

## Package layout

PyInstaller builds four executables:

- `Pitward.exe`: the windowed control panel
- `PitwardComponents.exe`: the console component downloader and verifier
- `workers/PitwardSTTWorker/PitwardSTTWorker.exe`: the Qwen3-ASR process
- `workers/PitwardTTSWorker/PitwardTTSWorker.exe`: the Chatterbox process

Models aren't in the archive. `PitwardComponents.exe` downloads the pinned component set
from `src/race_engineer/assets/components.v1.json` into
`%LOCALAPPDATA%\Pitward\components`, resumes partial downloads and verifies sha256.
Settings, logs and the SQLite database live under `%LOCALAPPDATA%\Pitward` (or
`PITWARD_DATA_DIR`).

**Component set 4** contains llama.cpp b12088 (CPU build), Ministral 3 3B Instruct 2512
Q4_K_M, Qwen3-ASR 0.6B and the five Chatterbox Multilingual V2 assets. A fresh setup
downloads about 7.0 GB.

**Installed builds run on CPU.** `Installation.apply_to` forces the CPU conversation
runtime, and the release workflow builds the TTS worker with `--backend cpu`. Source mode
uses CUDA. CPU race latency is not established.

**Model migrations.** The manifest's `model_migrations` list names retired artifacts for
every supported upgrade path. After the new model loads and verifies, Pitward deletes the
listed obsolete files from its managed cache and records it in `.model-migrations.json`.
Failed loads, custom paths, unknown files and locked files are left alone, and cleanup
retries on a later start. There is no wildcard deletion.

Licenses: [REVIEW-MINISTRAL.md](../distribution/REVIEW-MINISTRAL.md) covers the dialogue
model; [REVIEW-0.1.0.md](../distribution/REVIEW-0.1.0.md) covers the speech and radio
assets. Full license texts and runtime notices ship in each package.

## Local build

Install PyInstaller into the application, STT and TTS environments:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[desktop,build]"
.\data\stt-prototype\runtime\Scripts\python.exe -m pip install -e ".[asr,build]"
.\.venv\Scripts\python.exe scripts\setup_radio_tts.py --backend cpu
.\data\tts-chatterbox\runtime\Scripts\python.exe -m pip install -e ".[build]"
```

Build the portable archive, then check the frozen speech dependencies without loading
weights. The worker must return `runtime_ready: true`; the release workflow runs the same
check.

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts\build_windows.ps1
& .\build\package\dist\Pitward\workers\PitwardTTSWorker\PitwardTTSWorker.exe --check-runtime --diagnostics
if ($LASTEXITCODE -ne 0) { throw "Frozen speech dependencies failed initialization" }
```

Transformers 5 must be collected as external source files, because its import-time
filename inspection fails on shortened PYZ paths. Diffusers and Transformers also need
their dependencies' distribution metadata. Keep both rules in the TTS worker spec.

Then load both frozen workers against the development models and synthesize English and
Turkish replies and announcements without playback:

```powershell
.\.venv\Scripts\python.exe scripts\smoke_packaged_workers.py
```

Output: `dist/Pitward-<version>-windows-x64.zip` and `dist/SHA256SUMS.txt`. With Inno
Setup 6 installed, build the installer:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts\build_installer.ps1
```

The artifacts are unsigned, so SmartScreen may warn on first launch. Inno Setup is free
for non-commercial use; commercial use needs an
[Inno Setup license](https://jrsoftware.org/isorder.php).

## First launch and uninstall

On first launch, Start stays disabled until the component set is present. **Set up local
AI components** shows the third-party terms and runs the downloader without blocking the
UI. After a successful download, the panel saves settings, releases its single-instance
lock and restarts. Failed or cancelled setup stays open for retry. Running
`PitwardComponents.exe` directly needs a manual panel restart.

An explicit uninstall removes `%LOCALAPPDATA%\Pitward\components` and
`%LOCALAPPDATA%\Pitward\updates`. Settings, the profile and history database, and logs are
kept. In-place upgrades keep the models. A custom `PITWARD_DATA_DIR` is never cleaned up.

## Application updates

Installer-managed builds check the GitHub latest-release endpoint once after startup, once
components are present. **Releases & updates** has a manual check and a switch to turn
off startup checks. Source, preview and portable builds don't check. No audio, telemetry,
profile data or tokens are sent.

Only newer stable versions from `furkannehir/race-engineer` qualify. The updater needs the
versioned installer and `SHA256SUMS.txt`, checks the published checksum against the GitHub
asset digest when available, and verifies the size and sha256 of the download (at most
2 GiB, `updates._MAX_INSTALLER`). Offline startup checks are quiet; manual failures are
shown.

**Update now / Later** appears only when the engineer is stopped. Nothing downloads before
Update now. The download is cancellable and staged under `updates`. Pitward then saves
settings, releases its lock, closes and opens the Inno Setup wizard in the existing install
folder. The upgrade doesn't run the uninstaller, so models, settings and history stay.

## Release flow

1. Change the version in `src/race_engineer/version.py` and commit it.
2. Push the tag: `git tag v<version>` then `git push origin v<version>`. The
   `windows-package` workflow builds the files and, after clearance checks, creates a
   **draft** GitHub Release. Manual workflow runs produce a 14-day test artifact instead.
3. Run the clean-machine acceptance below on the draft's files.
4. Publish the draft. Published releases are immutable; every public release gets a new
   version.
5. Deploy the download site separately (see
   [the website README](../website/pitward-downloads/README.md)). It enables links only
   when the target release has all expected assets.

Release creation is blocked while either field in `distribution/release-clearance.json`
is false. Change a field only after the license review it refers to is complete. Each
release has a record in [releases/](releases/).

## Clean-machine acceptance

On a Windows 11 x64 machine with no Python and no repository checkout:

1. Portable ZIP extraction, installer upgrade and uninstall.
2. Component download, interruption and resume, and `verify`.
3. First launch, taskbar and tray icon, and About build identity.
4. Microphone and output selection; keyboard, mouse and wheel PTT.
5. English and Turkish replies against live iRacing.
6. Settings, profile and history database, and models preserved across an upgrade.
7. Update check: newer, equal and older versions; offline; Later; cancelled download;
   checksum failure; deferral during a live session.

Also check that uninstall removes only the components cache and the app, and keeps
settings, profiles, history and logs. A source test or `--check-runtime` doesn't replace
this or a listening test.
