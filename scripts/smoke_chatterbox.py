"""Exercise EN/TR replies and automatic calls through one persistent default-voice worker."""

import argparse
import asyncio
import json
from pathlib import Path

from race_engineer.config import TtsConfig, load_config
from race_engineer.core.contracts import Utterance
from race_engineer.core.conversation import ConversationReply
from race_engineer.core.enums import PlaybackStatus
from race_engineer.tts.chatterbox import ChatterboxAnnouncementEngine, ChatterboxSpeechBackend


async def smoke(config_path: Path, *, play_audio: bool, output_device: int | None = None) -> None:
    config = load_config(config_path)
    speech = (
        config.tts
        if output_device is None
        else TtsConfig.model_validate(config.tts.model_dump() | {"output_device": output_device})
    )
    backend = ChatterboxSpeechBackend(speech)
    automatic = ChatterboxAnnouncementEngine(backend)
    try:
        await backend.start()
        pid = backend.worker_pid
        print(f"Chatterbox ready: device={backend.effective_device}, worker={pid}", flush=True)
        for language, text in (
            ("en", "Radio check. Your race engineer is ready."),
            ("tr", "Telsiz kontrolü. Yarış mühendisin hazır."),
        ):
            reply = ConversationReply(
                language=language,
                text=text,
                status="answered",
                session_id="smoke",
                source_sequence=0,
                mode="replay",
            )
            result = await backend.speak(reply, play_audio=play_audio)
            print(
                json.dumps({"path": "conversation", **result.model_dump()}, ensure_ascii=True),
                flush=True,
            )
            assert result.played == play_audio
        for index, (language, text) in enumerate((("en", "Green flag."), ("tr", "Sarı bayrak."))):
            result = await automatic.speak(
                Utterance(
                    intent_id=f"smoke-{index}", text=text, generator_metadata={"language": language}
                ),
                play_audio=play_audio,
            )
            print(json.dumps({"path": "automatic", **result.model_dump(mode="json")}), flush=True)
            if result.status is not PlaybackStatus.COMPLETED:
                raise RuntimeError("automatic_smoke_failed")
        if backend.worker_pid != pid:
            raise RuntimeError("speech_worker_was_reloaded")
        print(
            "Both paths and both languages completed with one unchanged worker/model.", flush=True
        )
    finally:
        await backend.aclose()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config/default.toml"))
    parser.add_argument("--no-playback", action="store_true")
    parser.add_argument("--output-device", type=int, help="index from list-output-devices")
    args = parser.parse_args()
    asyncio.run(
        smoke(args.config, play_audio=not args.no_playback, output_device=args.output_device)
    )


if __name__ == "__main__":
    main()
