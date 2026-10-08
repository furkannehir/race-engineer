"""Load packaged speech workers against local development models without playing audio."""

import argparse
import asyncio
import math
import struct
from pathlib import Path

from race_engineer.config import load_config
from race_engineer.core.conversation import ConversationReply
from race_engineer.core.speech_input import AudioClip
from race_engineer.stt.qwen import QwenSpeechRecognizer
from race_engineer.tts.piper import PiperConversationSpeaker


async def smoke(package: Path, config_path: Path, *, stt: bool, tts: bool) -> None:
    config = load_config(config_path)
    if stt:
        recognizer = QwenSpeechRecognizer(
            config.stt.model_copy(
                update={
                    "worker_path": package / "workers/PitwardSTTWorker/PitwardSTTWorker.exe",
                    "startup_timeout_s": 300.0,
                }
            )
        )
        try:
            sample_rate = 16_000
            pcm = bytearray(sample_rate * 2)
            for index in range(sample_rate):
                sample = int(math.sin(2 * math.pi * 440 * index / sample_rate) * 3_000)
                struct.pack_into("<h", pcm, index * 2, sample)
            result = await recognizer.transcribe(AudioClip(bytes(pcm), sample_rate))
            print(f"Packaged STT worker completed inference ({result.status}).")
        finally:
            await recognizer.aclose()
    if tts:
        speaker = PiperConversationSpeaker(
            config.radio_tts.model_copy(
                update={
                    "worker_path": package / "workers/PitwardTTSWorker/PitwardTTSWorker.exe",
                    "startup_timeout_s": 120.0,
                }
            )
        )
        try:
            result = await speaker.speak(
                ConversationReply(
                    language="en",
                    text="Radio check.",
                    status="answered",
                    session_id="package-smoke",
                    source_sequence=0,
                    mode="replay",
                ),
                play_audio=False,
            )
            if result.played or result.audio_duration_s <= 0:
                raise RuntimeError("packaged TTS worker returned an invalid smoke result")
            print("Packaged TTS worker synthesized audio without playback.")
        finally:
            await speaker.aclose()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--package", type=Path, default=Path("build/package/dist/Pitward")
    )
    parser.add_argument("--config", type=Path, default=Path("config/default.toml"))
    parser.add_argument("--skip-stt", action="store_true")
    parser.add_argument("--skip-tts", action="store_true")
    args = parser.parse_args()
    if args.skip_stt and args.skip_tts:
        parser.error("at least one worker must be selected")
    asyncio.run(
        smoke(
            args.package.resolve(),
            args.config.resolve(),
            stt=not args.skip_stt,
            tts=not args.skip_tts,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
