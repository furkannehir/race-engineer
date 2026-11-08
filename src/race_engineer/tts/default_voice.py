"""Placement adapter for the pinned Chatterbox cached-default-voice path.

No torch import in the application process. The worker supplies its torch module.
The reference encoders remain on CPU; generation weights stay resident on CUDA.
"""

from typing import Any, Never


def _decoder_device(decoder: Any) -> Any:
    # Upstream uses tokenizer.parameters(), although cached conditioning bypasses
    # that CPU-only reference tokenizer. Follow weights used for synthesis instead.
    return next(decoder.flow.parameters()).device


def _reject_reference(_decoder: Any, *args: Any, **kwargs: Any) -> Never:
    raise RuntimeError("tts_reference_audio_unsupported")


def place_default_voice_cuda(model: Any, torch: Any, *, precision: str) -> None:
    """Place only active synthesis modules on CUDA, before any warm-up or request.

    Called after CPU loading and default-conditioning validation in the 0.1.7
    worker. V2 and the pinned V3 source have the same S3Gen module layout. Check
    that boundary explicitly: an upstream layout change must fail visibly rather
    than accidentally send an unplaced generation module CPU inputs.
    """
    if precision not in {"bf16", "fp32"}:
        raise ValueError("tts_precision_invalid")
    decoder = model.s3gen
    children = dict(decoder.named_children())
    buffers = dict(decoder.named_buffers(recurse=False))
    if (
        set(children) != {"tokenizer", "speaker_encoder", "flow", "mel2wav"}
        or set(buffers) != {"trim_fade"}
        or list(decoder.parameters(recurse=False))
        or not isinstance(getattr(type(decoder), "device", None), property)
        or not isinstance(getattr(type(decoder), "dtype", None), property)
        or next(decoder.tokenizer.parameters()).device.type != "cpu"
        or next(decoder.speaker_encoder.parameters()).device.type != "cpu"
        or next(model.ve.parameters()).device.type != "cpu"
    ):
        raise RuntimeError("tts_default_voice_layout_unsupported")

    # Adapt this instance only; do not patch the upstream class or reconstruct its
    # weights. The model lives exclusively in this worker and is never pickled.
    decoder.__class__ = type(
        "DefaultVoiceS3Gen",
        (type(decoder),),
        {
            "__module__": __name__,
            "device": property(_decoder_device),
            "embed_ref": _reject_reference,
        },
    )
    if precision == "bf16":
        # Convert on CPU so full FP32 T3 weights never occupy CUDA temporarily.
        model.t3.to(dtype=torch.bfloat16)
        model.conds.t3.to(dtype=torch.bfloat16)
    model.t3.to(device="cuda")
    decoder.flow.to(device="cuda", dtype=torch.float32)
    decoder.mel2wav.to(device="cuda", dtype=torch.float32)
    for name, buffer in buffers.items():
        setattr(decoder, name, buffer.to(device="cuda"))
    model.conds.to("cuda")
    model.device = "cuda"
