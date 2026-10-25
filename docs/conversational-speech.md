# Local conversational speech

Conversational replies and automatic announcements now share **Chatterbox
Multilingual 0.1.7** and its built-in default voice. English and Turkish use the
same model/voice; no reference recording is required. Piper and Windows SAPI
are no longer runtime options.

See [the shared speech guide](text-to-speech.md) for installation, pinned versions,
CUDA/CPU behavior, configuration migration, tests and known limitations.
Only `[tts]` remains; remove the old `[radio_tts]` section in custom configs.
PTT beep/static WAV cues are unchanged. No voice uploads or emotion controls are
part of this migration.
