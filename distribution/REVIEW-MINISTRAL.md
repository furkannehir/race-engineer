# Ministral component review

Recorded 2026-11-18 for development component set 4. The published v0.2.0 tag
and assets still use component set 3 and are unchanged.

The conversation model is the text-only Q4_K_M artifact from the official
[Ministral GGUF repository at the pinned revision](https://huggingface.co/mistralai/Ministral-3-3B-Instruct-2512-GGUF/blob/eb599d408350ea2bb60452cb86be7c7b2fc28227/README.md).
Its model card declares Apache-2.0 and identifies the original Ministral model.
The original model card at revision `b35d4dfe56c142746f54dbd64f579faab2744308`
also declares Apache-2.0. The GGUF revision's conventional `LICENSE` URL returned
404; the original model's metadata did not list an exact LICENSE/NOTICE filename.
The pinned Hub file inventories were checked again on 18 November: neither
repository contains a differently named license or notice file.
The canonical Apache-2.0 text is retained in
`licenses/Ministral-Apache-2.0.txt`; its URL, size and SHA-256 are in `sources.json`.
The weights are downloaded from upstream, never bundled or relicensed as GPL.

- File: `Ministral-3-3B-Instruct-2512-Q4_K_M.gguf`
- Revision: `eb599d408350ea2bb60452cb86be7c7b2fc28227`
- Bytes: `2147023008`
- SHA-256: `9ed150d4367e68df0ac8e1540f6ddc65b42d0ee26378329d1ecbca60f93fc5f8`
- Vision/projector files are not downloaded or loaded.
- Runtime remains llama.cpp b12088 CPU (MIT) in the installer. The source
  configuration explicitly selects the separately verified CUDA 13.4 runtime.

Qwen3-ASR and Chatterbox model pins, worker dependencies, the built-in voice,
watermarking and PTT cue provenance remain covered by `REVIEW-0.1.0.md`.
Legacy LFM license notices remain for the explicit source rollback profile;
component set 4 does not download LFM weights.

Fresh-install component downloads total 6,955,047,301 bytes (about 7.0 GB).
An upgrade reuses unchanged speech assets, settings and profiles. It downloads
the new 2.15 GB conversation model when missing. Once the owned model server
successfully loads verified Ministral, migration removes only the checksum-matched
obsolete managed LFM weights. Failed starts retain LFM; locked files defer cleanup.
Uninstall continues to remove the application-owned component directory.
