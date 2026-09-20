import asyncio
import math
import struct
import wave

import pytest
from pydantic import ValidationError

from race_engineer.config import SttConfig
from race_engineer.core.speech_input import AudioClip, SpeechInputError, Transcription
from race_engineer.stt.audio import load_wav, silence_reason
from race_engineer.stt.capture import CaptureBuffer, PushToTalkMicrophone, virtual_key
from race_engineer.stt.worker import normalize_result


def tone(seconds=0.4, rate=16000):
    values = [
        int(6000 * math.sin(2 * math.pi * 440 * i / rate)) for i in range(int(rate * seconds))
    ]
    return AudioClip(struct.pack(f"<{len(values)}h", *values), rate)


def test_audio_limits_and_no_bytes_in_repr():
    with pytest.raises(ValueError):
        AudioClip(b"x")
    with pytest.raises(ValueError):
        AudioClip(b"\0\0", 12345)
    with pytest.raises(ValueError):
        AudioClip(b"\0" * (16000 * 2 * 31))
    assert "pcm16" not in repr(tone())


def test_energy_gate_handles_silence_dc_offset_and_short_clips():
    config = SttConfig()
    assert silence_reason(AudioClip(b"\0\0" * 16000), config) == "audio_below_threshold"
    assert (
        silence_reason(AudioClip(struct.pack("<h", 3000) * 16000), config)
        == "audio_below_threshold"
    )
    assert silence_reason(tone(0.1), config) == "audio_too_short"
    assert silence_reason(tone(), config) is None
    with pytest.raises(SpeechInputError, match="audio_too_long"):
        silence_reason(tone(2), SttConfig(max_capture_s=1))


def test_wav_stereo_downmix_and_duration_rejection(tmp_path):
    path = tmp_path / "stereo.wav"
    with wave.open(str(path), "wb") as output:
        output.setnchannels(2)
        output.setsampwidth(2)
        output.setframerate(48000)
        output.writeframes(struct.pack("<hh", 1000, 3000) * 48000)
    result = load_wav(path)
    assert result.sample_rate_hz == 48000
    assert result.duration_s == 1
    assert result.pcm16 == struct.pack("<h", 2000) * 48000
    with pytest.raises(SpeechInputError, match="audio_too_long"):
        load_wav(path, max_duration_s=0.5)


def test_wav_rejects_unsupported_formats(tmp_path):
    path = tmp_path / "bad.wav"
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(1)
        output.setframerate(16000)
        output.writeframes(b"\0" * 16000)
    with pytest.raises(SpeechInputError, match="pcm16"):
        load_wav(path)
    with pytest.raises(SpeechInputError, match="unreadable"):
        load_wav(tmp_path / "missing.wav")


def test_capture_never_keeps_idle_or_overlong_audio():
    buffer = CaptureBuffer(16000, 1)
    buffer.append(tone().pcm16)
    with pytest.raises(SpeechInputError, match="empty"):
        buffer.finish()
    buffer.begin()
    buffer.append(tone().pcm16)
    assert buffer.finish().duration_s == pytest.approx(0.4)
    buffer.begin()
    buffer.append(tone(2).pcm16)
    buffer.append(tone().pcm16)
    with pytest.raises(SpeechInputError, match="too_long"):
        buffer.finish()
    buffer.begin()
    buffer.append(tone().pcm16, overflow=True)
    with pytest.raises(SpeechInputError, match="overflow"):
        buffer.finish()


def test_key_mapping_and_config_validation():
    assert virtual_key("F8") == 0x77
    assert virtual_key("F24") == 0x87
    assert virtual_key("RCTRL") == 0xA3
    with pytest.raises(ValidationError):
        SttConfig(ptt_key="ESC")
    with pytest.raises(ValidationError):
        SttConfig(sample_rate_hz=12345)


def test_push_to_talk_starts_on_press_and_stops_on_release():
    class Stream:
        def __init__(self, **kwargs):
            self.callback = kwargs["callback"]
            self.started = self.stopped = self.closed = False

        def start(self):
            self.started = True
            self.callback(tone().pcm16, 6400, None, False)

        def stop(self):
            self.stopped = True

        def close(self):
            self.closed = True

    states = iter([False, True, True, False])
    streams = []

    def factory(**kwargs):
        stream = Stream(**kwargs)
        streams.append(stream)
        return stream

    mic = PushToTalkMicrophone(
        SttConfig(),
        stream_factory=factory,
        key_down=lambda key: next(states) if key == 0x77 else False,
    )
    assert not streams[0].started

    async def run():
        audio = await mic.next_clip(lambda message: None)
        await mic.aclose()
        return audio

    assert asyncio.run(run()).duration_s == pytest.approx(0.4)
    assert streams[0].started and streams[0].stopped and streams[0].closed


def test_escape_before_press_does_not_start_capture():
    class Stream:
        def __init__(self, **kwargs):
            pass

        def start(self):
            raise AssertionError("must not start")

        def close(self):
            pass

    mic = PushToTalkMicrophone(SttConfig(), stream_factory=Stream, key_down=lambda key: key == 0x1B)

    async def run():
        assert await mic.next_clip() is None
        await mic.aclose()

    asyncio.run(run())


def test_bilingual_model_output_and_language_filtering():
    clip = tone()
    english = normalize_result(
        {"language": "English", "transcription": "Where are we?"}, clip, 10, "auto"
    )
    turkish = normalize_result(
        {"language": "Turkish", "transcription": "Kaçıncıyız?"}, clip, 10, "auto"
    )
    assert english.language == "en" and turkish.language == "tr"
    forced = normalize_result({"language": None, "transcription": "Kaçıncıyız?"}, clip, 1, "tr")
    assert forced.language == "tr"
    silence = normalize_result({"language": "None", "transcription": ""}, clip, 1, "auto")
    assert silence.status == "no_speech" and not silence.text
    other = normalize_result({"language": "French", "transcription": "bonjour"}, clip, 1, "auto")
    assert other.status == "unsupported_language" and not other.text


def test_result_rejects_empty_success_and_text_on_silence():
    with pytest.raises(ValidationError):
        Transcription(status="transcribed", audio_duration_s=1)
    with pytest.raises(ValidationError):
        Transcription(status="no_speech", text="hallucination", audio_duration_s=1)


def test_cancelling_during_stream_start_waits_then_stops_before_returning():
    import threading

    starting = threading.Event()
    allow_start = threading.Event()
    steps = []

    class Stream:
        def __init__(self, **kwargs):
            pass

        def start(self):
            starting.set()
            assert allow_start.wait(timeout=2)
            steps.append("started")

        def stop(self):
            steps.append("stopped")

        def close(self):
            steps.append("closed")

    keys = iter((False, True))
    microphone = PushToTalkMicrophone(
        SttConfig(),
        stream_factory=Stream,
        key_down=lambda key: next(keys, True) if key == 0x77 else False,
    )

    async def run():
        capture = asyncio.create_task(microphone.next_clip())
        await asyncio.to_thread(starting.wait, 1)
        capture.cancel()
        await asyncio.sleep(0)
        assert not steps
        allow_start.set()
        with pytest.raises(asyncio.CancelledError):
            await capture
        assert steps == ["started", "stopped"]
        await microphone.aclose()

    asyncio.run(run())
    assert steps == ["started", "stopped", "closed"]
