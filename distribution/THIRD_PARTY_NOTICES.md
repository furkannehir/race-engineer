# Pitward third-party notices

Pitward is distributed under GPL-3.0. A packaged build also contains or downloads
third-party components governed by their own terms.

- Qt for Python / PySide6: LGPL-3.0/GPL/commercial licensing options. The distribution
  uses the open-source terms and keeps the Qt libraries dynamically linked in the
  application directory. Pitward itself is GPL-3.0; `LICENSE` contains the GPL text.
  Qt/PySide source is available from https://code.qt.io/cgit/pyside/pyside-setup.git/
  and https://download.qt.io/official_releases/qt/. Match the bundled runtime version
  recorded in `licenses/runtime-main/manifest.json`.
- PyInstaller bootloader: GPL-2.0-or-later with the PyInstaller bootloader exception.
- llama.cpp CPU runtime: MIT.
- LFM3-1.7B Q4_K_M: [LFM Open License v1.0](https://huggingface.co/LiquidAI/LFM3-1.7B/blob/main/LICENSE).
  These are separate model terms; Pitward's GPL license does not replace them.
  The full text is in `licenses/LFM-Open-License-v1.0.txt`. Its commercial-use
  revenue threshold applies separately to users of the model; review section 5.
- Qwen3-ASR-0.6B: Apache-2.0 model terms.
  The explicit source Qwen3-4B rollback also uses Apache-2.0 model terms; its
  weights are not part of component set 3.
- Chatterbox TTS 0.1.7 and the pinned ResembleAI/chatterbox multilingual V2 model: MIT.
  The built-in `conds.pt` voice is used; no reference recordings are uploaded.
- PyTorch/torchaudio, Transformers, Resemble Perth watermarking, sounddevice and
  dependencies retain their respective upstream licenses. Chatterbox watermarking
  remains enabled.
- QtAwesome code: MIT. The packaged Material Design Icons 6.9.96 font uses the
  Pictogrammers Free License / Apache-2.0; see `licenses/Material-Design-Icons.txt`.
  Unused QtAwesome font families are excluded from this release.
- pygame-ce: LGPL-2.1; sounddevice and pyirsdk: MIT. GNU LGPL 2.1 and 3.0 texts are
  included in `licenses`. Bundled wheel notices and metadata are preserved under
  `licenses/runtime-main`, `licenses/runtime-stt`, and `licenses/runtime-tts`.

Pinned model URLs/checksums are in `components.v1.json`. Old Piper engines/voices
are no longer shipped. Model license: https://huggingface.co/ResembleAI/chatterbox

The model license copies and their upstream URLs/checksums are in `licenses/sources.json`.
Pinned Qwen and Chatterbox model cards declare Apache-2.0 and MIT respectively; the
LFM license is copied from the exact quantized model revision. These model files
are downloaded from upstream after acceptance of the third-party terms; they are
not bundled or relicensed under the application's GPL license.

The original radio beep and procedural squelch are project assets under GPL-3.0.
The 0.1.0 review and maintainer provenance confirmation are in `REVIEW-0.1.0.md`.
