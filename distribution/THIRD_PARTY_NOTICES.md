# Pitward third-party notices

Pitward is distributed under GPL-3.0. A packaged build also contains or downloads
third-party components governed by their own terms.

- Qt for Python / PySide6: LGPL-3.0/GPL/commercial licensing options. The distribution
  keeps the Qt libraries dynamically linked in the application directory.
- PyInstaller bootloader: GPL-2.0-or-later with the PyInstaller bootloader exception.
- llama.cpp CPU runtime: MIT.
- Qwen3-4B-Instruct-2507 and Qwen3-ASR-0.6B: Apache-2.0 model terms.
- Chatterbox TTS 0.1.7 and the pinned ResembleAI/chatterbox multilingual V2 model: MIT.
  The built-in `conds.pt` voice is used; no reference recordings are uploaded.
- PyTorch/torchaudio, Transformers, Resemble Perth watermarking, sounddevice and
  dependencies retain their respective upstream licenses. Chatterbox watermarking
  remains enabled.

Pinned model URLs/checksums are in `components.v1.json`. Old Piper engines/voices
are no longer shipped. Model license: https://huggingface.co/ResembleAI/chatterbox

Release publication remains blocked until the project records clearance for the supplied
radio beep and completes its speech-model distribution review. Both clearance
gates remain blocked; migration does not clear them.
