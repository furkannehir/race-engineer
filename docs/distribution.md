# Pitward Windows builds and releases

Pitward uses the version in `src/race_engineer/version.py` for Python metadata, Windows
executable metadata, archive names, installer names and Git tags. A release tag must be
exactly `v<version>`; mismatches stop the build.

## Build layout

PyInstaller creates four executables:

- `Pitward.exe`, the windowed control panel;
- `PitwardComponents.exe`, the console component downloader and verifier;
- `PitwardSTTWorker.exe`, the isolated Qwen3-ASR process; and
- `PitwardTTSWorker.exe`, the isolated Chatterbox multilingual process.

The speech workers are folder-mode bundles inside `Pitward/workers`. Models are not in the
application archive. `PitwardComponents.exe` downloads the pinned component set into
`%LOCALAPPDATA%\Pitward\components`, resumes partial downloads and verifies SHA-256 hashes.

Application settings, logs and SQLite profiles also live below `%LOCALAPPDATA%\Pitward` in
packaged builds. Source launches keep using the repository's ignored `data` and `logs`
directories.

An explicit uninstall removes the downloaded AI cache at
`%LOCALAPPDATA%\Pitward\components`, plus downloaded installers in
`%LOCALAPPDATA%\Pitward\updates`. Settings, profiles/history and logs are retained;
in-place upgrades retain the models too. A custom `PITWARD_DATA_DIR` is outside this
installer cleanup rule and must be cleaned up manually. Existing 0.1.0 uninstallers
retain their old behavior until replaced by a rebuilt installer.

Development component set 4 selects Ministral 3 3B Instruct 2512 Q4_K_M and
llama.cpp b12088 CPU. Qwen3-ASR and the five multilingual V2 speech assets retain
their existing pins. Fresh setup downloads about 7.0 GB. Frozen paths match the
new model/runtime manifest. Upgrades reuse speech assets. After successful owned
Ministral startup and checksum verification, migration removes the known obsolete
LFM weights. Failed/cancelled starts, custom paths and unrecognized files retain
their old files. Locked-file cleanup retries on a later start. Published v0.2.0 files still contain component set 3 and do not change until a new version is built and published.

Ministral's pinned official model card declares Apache-2.0. The current review is
[REVIEW-MINISTRAL.md](../distribution/REVIEW-MINISTRAL.md); the earlier
[0.1.0 review](../distribution/REVIEW-0.1.0.md) still covers unchanged speech and
radio assets. See the [production migration guide](ministral-live.md).
Full model license texts and runtime wheel notices are included in each package.
The release workflow builds a CPU TTS worker; racing latency on CPU is not
established. CUDA builds can use `setup_radio_tts.py --backend cuda` instead.
Clean-Windows frozen-worker EN/TR synthesis and audio checks remain mandatory
before publishing; a source-worker smoke test does not validate the frozen bundle.

## Local build

Install PyInstaller into the main, STT and TTS environments:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[desktop,build]"
.\data\stt-prototype\runtime\Scripts\python.exe -m pip install -e ".[asr,build]"
.\.venv\Scripts\python.exe scripts\setup_radio_tts.py --backend cpu
.\data\tts-chatterbox\runtime\Scripts\python.exe -m pip install -e ".[build]"
```

Build the portable archive:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts\build_windows.ps1
```

After a local build, first exercise the frozen TTS dependency imports without loading or
downloading model weights. `--help` only checks the command-line parser:

```powershell
& .\build\package\dist\Pitward\workers\PitwardTTSWorker\PitwardTTSWorker.exe --check-runtime --diagnostics
if ($LASTEXITCODE -ne 0) { throw "Frozen speech dependencies failed initialization" }
```

The worker must return `runtime_ready: true`. The release workflow runs this check too.
Transformers 5 must be collected as external Python source: its import-time filename
inspection is incompatible with shortened filenames inside PYZ. Diffusers and Transformers
also require dependency distribution metadata; copying the importable modules alone is
insufficient. Keep these rules in the TTS worker spec.

Then load both frozen speech workers against the development models and synthesize English
and Turkish replies and announcements with one shared worker, without audio playback:

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
checksum-verifying downloader. The panel monitors setup without blocking the UI and
restarts automatically after a successful exit and a check that required files are
present. The restart saves panel settings, preserves config/settings paths and the data
directory, and releases the single-instance lock before relaunching. Failed or cancelled
setup stays open for retry; quitting the panel stops automatic restart. Running
`PitwardComponents.exe` directly still requires a manual panel restart.

## Application updates

Installer-managed Windows builds check the public GitHub latest-release endpoint once
after startup, after local components are present. **Releases & updates** provides a
manual check and a switch to disable startup checks. Source launches, design previews
and portable builds keep the release-page link and do not check automatically. This is
an explicit network exception to local speech/conversation: no audio, telemetry, profiles
or authentication tokens are sent.

Only newer published stable versions from `furkannehir/race-engineer` qualify. The
updater requires the exact versioned Windows installer and `SHA256SUMS.txt`, validates
the published checksum against the GitHub asset digest when available, and verifies
the size and SHA-256 of the entire download before offering it to Windows. Metadata and
download work runs off the UI thread. Offline startup checks are quiet; manual failures
are visible and leave the current installation running.

The **Update now / Later** prompt waits until the engineer is stopped and the panel is
visible. Nothing is downloaded without Update now. A cancellable download is staged
under the user's `updates` folder. Pitward then saves settings, releases the single-instance
lock, closes and opens the normal Inno Setup wizard in the existing installation folder.
The optional **Launch Pitward** step preserves custom config/settings paths. The upgrade
does not invoke the uninstaller, so existing models, settings and driver history remain.
Cancelling the installer itself requires reopening Pitward manually. Verified installers
can be reused on retry; explicit uninstall removes the standard update cache as well.

The public 0.1.0 binary predates this updater. Users need one manual installation of a
new build containing it before later releases can be offered inside the app. Keep
published releases immutable and increment the authoritative application version for
each future public release. Checksums verify download integrity; the current unsigned
installer still has the signing/SmartScreen limitation described above.

Before release, verify on Windows: newer/equal/older versions, offline and rate-limited
checks, Later, download cancellation and checksum failure, deferral during a live session,
and an actual installer upgrade retaining models, profiles and custom paths. See the
[GitHub release API](https://docs.github.com/en/rest/releases/releases#get-the-latest-release)
and [Inno Setup parameters](https://jrsoftware.org/inno-setup/documentation/topic_setupcmdline.htm).

## Release flow

The current feature release is **0.2.0**, published on 18 November 2026 and recorded
in [the release record](releases/0.2.0.md). It includes new timing/progress data
and a source-only Ministral preview; the installed model/component set remains
LFM. The tagged Windows build, public assets, updater detection and download
website publication were verified. Future version tags still create drafts;
public release publication and website deployment are separate operations.

The previous patch release is recorded in [the 0.1.1 release record](releases/0.1.1.md),
including verified public assets, the tagged workflow and upgrade instructions.

For the first release, use the [0.1.0 preparation record](releases/0.1.0.md) and
[prepared release notes](../distribution/RELEASE_NOTES.md). The
[download dashboard](../website/pitward-downloads/README.md) enables links only for
the published target release with all expected assets.

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
5. English LFM replies against live iRacing, plus isolated English/Turkish speech
   worker checks; Turkish LFM conversation remains unvalidated; and
6. preservation of settings, profile/history database and models across an upgrade.

Also verify that explicit uninstall removes only the standard components cache and app
installation, while retaining settings, profiles/history and logs. A source test or
`--check-runtime` cannot replace this clean-machine acceptance or a listening test.
