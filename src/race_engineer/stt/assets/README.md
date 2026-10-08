# Radio cues

The installed `radio-open.wav` layers the original `02-soft` squelch with the user-provided
`radio_beep.mp3`. `radio-close.wav` layers the soft closing squelch with the **reversed decoded
beep waveform**. The supplied MP3 is preserved unchanged.

Both installed cues are mono PCM16 at 48 kHz: opening 340 ms, closing 460 ms. The beep's first
20 ms of quiet preroll is trimmed and its edges faded. It starts 30 ms into the opening and
120 ms into the closing; the closing ends with a static tail. Mix peak amplitude is capped
at 0.23 full scale before the application's volume setting. Runtime playback uses only these
WAVs and does not need an MP3 decoder.

The procedural squelch components come from `scripts/build_radio_cues.py` under the
project's GPL-3.0 license. The MP3's external source/license has not been supplied; those
procedural components do not establish licensing for the recording or mixed assets.

Listen to the installed mixed pair without starting the engineer:

```powershell
Invoke-Item src\race_engineer\stt\assets\radio-open.wav
Invoke-Item src\race_engineer\stt\assets\radio-close.wav
```

Regenerate the mix with the existing STT runtime (its SoundFile decoder handles the MP3):

```powershell
.\data\stt-prototype\runtime\Scripts\python.exe scripts\build_radio_cues.py --replace
```

The experiment variants, intermediate audio and HTML previews have been removed. Only the
selected WAV pair, unchanged MP3 source and single reproducible builder are retained. The MP3
is a source asset, excluded from the installed runtime package; only the WAVs and this README
are packaged.
