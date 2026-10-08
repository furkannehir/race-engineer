# Pitward Windows builds and releases

Pitward uses the version in `src/race_engineer/version.py` for Python metadata, Windows
executable metadata, archive names, installer names and Git tags. A release tag must be
exactly `v<version>`; mismatches stop the build.

## Build layout

PyInstaller creates four executables:

- `Pitward.exe`, the windowed control panel;
- `PitwardComponents.exe`, the console component downloader and verifier;
- `PitwardSTTWorker.exe`, the isolated Qwen3-ASR process; and
- `PitwardTTSWorker.exe`, the isolated Piper process.

The speech workers are folder-mode bundles inside `Pitward/workers`. Models are not in the
application archive. `PitwardComponents.exe` downloads the pinned component set into
`%LOCALAPPDATA%\Pitward\components`, resumes partial downloads and verifies SHA-256 hashes.

Application settings, logs and SQLite profiles also live below `%LOCALAPPDATA%\Pitward` in
packaged builds. Source launches keep using the repository's ignored `data` and `logs`
directories.

## Local build

Install PyInstaller into the main, STT and TTS environments:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[desktop,build]"
.\data\stt-prototype\runtime\Scripts\python.exe -m pip install -e ".[asr,build]"
.\data\tts-prototype\runtime\Scripts\python.exe -m pip install -e ".[radio,build]"
```

Build the portable archive:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts\build_windows.ps1
```

After a local build, load both frozen speech workers against the development models and
synthesize a silent radio check:

```powershell
.\.venv\Scripts\python.exe scripts\smoke_packaged_workers.py
```

Output is written to `dist/Pitward-<version>-windows-x64.zip`, with its digest in
`dist/SHA256SUMS.txt`. If Inno Setup 6 is installed, build the installer with:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts\build_installer.ps1
```

The current artifacts are unsigned, so Windows SmartScreen may warn on first launch. Treat
code signing as a release-readiness step before presenting Pitward as a broadly trusted
stable download. Inno Setup is free for non-commercial use; a maintainer using its compiler
commercially must obtain the applicable [Inno Setup license](https://jrsoftware.org/isorder.php).

On a packaged first launch, the Start button remains disabled until the pinned component
set is present. **Set up local AI components** shows the relevant terms and launches the
checksum-verifying downloader. Update checks are also user-initiated: **Releases & updates**
opens the project's latest-release page; Pitward makes no background update request.

## Release flow

Manual runs of the `windows-package` workflow create a 14-day GitHub Actions artifact for
testing. Pushing a matching version tag builds the same files and, after clearance checks,
creates a draft GitHub Release. Review the clean-machine results and release notes before
publishing the draft. Published releases should be immutable.

Example release preparation:

```powershell
# Change the authoritative version, review and commit it first.
git tag v0.2.0a1
git push origin v0.2.0a1
```

Public release creation remains deliberately blocked while either field in
`distribution/release-clearance.json` is false. Change a field only after the referenced
license/distribution review is actually complete.

## Clean-machine acceptance

Before publishing, use a Windows 11 x64 machine without Python or this repository and test:

1. portable ZIP extraction and installer upgrade/uninstall;
2. component download, interruption/resume and `verify`;
3. first launch, Pitward taskbar/tray icon and About build identity;
4. microphone/output selection and keyboard, mouse and wheel PTT;
5. English and Turkish STT/replies against live iRacing telemetry; and
6. preservation of settings, profile/history database and models across an upgrade.
