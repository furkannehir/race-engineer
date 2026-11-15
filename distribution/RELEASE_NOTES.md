# Pitward 0.1.1

This patch fixes speech startup in the Windows installer and makes setup, updates,
and microphone failures easier to handle.

## Changes

- Fixed Chatterbox initialization in packaged builds. Downloaded voice models are
  reused; no reference recording or replacement voice is needed.
- Pitward restarts automatically after successful local AI component setup.
- Installed builds can check for newer stable GitHub releases. **Update now**
  downloads and verifies the installer, then opens the normal Windows setup wizard.
  **Later** keeps the current app running. Prompts wait until the engineer is stopped.
- Added clearer speech-startup failures, microphone tests, and push-to-talk capture
  diagnostics. Normal recording releases play the closing cue even if capture fails.
- Explicit uninstall now removes downloaded models and cached update installers.
  Settings, driver profiles, history, and logs are retained.

## Installing or upgrading

Run `Pitward-Setup-0.1.1-windows-x64.exe` over the existing installation. Do not
uninstall first: an in-place upgrade keeps your downloaded models and driver data.
The public 0.1.0 app predates the updater, so this upgrade must be installed manually.
Later releases can be offered in the app. Startup checks can be disabled under
**Releases & updates**; portable/source launches use the GitHub release page.

For a fresh install, open **Set up local AI components**, review the third-party
terms, and allow about 5.9 GB of verified downloads. Pitward then restarts itself.
Select your microphone, output, and push-to-talk button, and use **Test mic** and
**Test voice** before starting the engineer.

## Known limitations

- Windows 11 x64 remains the release test target. Builds are unsigned and may
  display a SmartScreen prompt.
- Packaged inference and speech run on CPU; speech may miss racing deadlines.
  Source CUDA setup is separate.
- English conversation is the MVP; Turkish conversation remains unvalidated.
  LFM can still misunderstand questions or produce poor replies. This patch does
  not change the conversation model or claim to fix answer quality.
- Internet is used for component downloads and optional GitHub update checks.
  Speech, telemetry, and driver data remain local.
- If you use a custom `PITWARD_DATA_DIR`, its caches require manual cleanup on
  uninstall. Cancelling an update wizard requires reopening Pitward manually.

See [known issues](https://github.com/furkannehir/race-engineer/blob/main/docs/known-issues.md)
and the bundled third-party notices for details.
