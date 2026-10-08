import asyncio
import struct
import threading
from types import SimpleNamespace

import pytest

from race_engineer.stt.radio_cues import RadioCuePlayer, render_radio_cue


def test_packaged_cues_are_distinct_bounded_pcm_and_preserve_original_gain():
    opened = render_radio_cue("open")
    closed = render_radio_cue("close")
    assert len(opened) == round(48_000 * 0.34) * 2
    assert len(closed) == round(48_000 * 0.46) * 2
    assert opened != closed and any(opened) and any(closed)
    original = struct.unpack(f"<{len(opened) // 2}h", opened)
    quieter = render_radio_cue("open", 0.5)
    assert struct.unpack(f"<{len(quieter) // 2}h", quieter) == tuple(
        round(value * 0.5) for value in original
    )
    assert render_radio_cue("open", 0) == bytes(len(opened))
    for invalid in (-0.1, 1.1, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="volume"):
            render_radio_cue("open", invalid)


def test_cue_player_uses_selected_output_and_drops_muted_or_disabled_audio(monkeypatch):
    streams = []
    pcm = struct.pack("<h", 2400) * 1920
    monkeypatch.setattr("race_engineer.stt.radio_cues._load_cue_pcm", lambda cue: pcm)

    class Stream:
        def __init__(self, **kwargs):
            self.arguments = kwargs
            self.chunks = []
            streams.append(self)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.closed = True

        def write(self, data):
            self.chunks.append(data)
            return False

    monkeypatch.setattr(
        "race_engineer.stt.radio_cues._sounddevice",
        lambda: SimpleNamespace(RawOutputStream=Stream),
    )
    muted = [False]

    async def run():
        player = RadioCuePlayer(output_device=7, volume=0.5, muted=lambda: muted[0])
        assert await player.play("open")
        muted[0] = True
        assert not await player.play("close")
        assert not await RadioCuePlayer(volume=0).play("open")
        assert not await RadioCuePlayer(enabled=False).play("open")

    asyncio.run(run())
    assert len(streams) == 1
    assert streams[0].arguments["device"] == 7
    assert streams[0].arguments["samplerate"] == 48_000
    assert streams[0].arguments["channels"] == 1
    assert streams[0].closed
    assert b"".join(streams[0].chunks) == struct.pack("<h", 1200) * 1920


def test_cue_failure_is_logged_once_and_does_not_break_capture(monkeypatch, caplog):
    attempts = []
    monkeypatch.setattr("race_engineer.stt.radio_cues._load_cue_pcm", lambda cue: b"\0\0")

    def unavailable():
        attempts.append(True)
        raise OSError("output unavailable")

    monkeypatch.setattr("race_engineer.stt.radio_cues._sounddevice", unavailable)

    async def run():
        player = RadioCuePlayer()
        assert not await player.play("open")
        assert not await player.play("close")

    asyncio.run(run())
    assert len(attempts) == 1
    assert [record.event for record in caplog.records] == ["radio_cue_unavailable"]


def test_cancelling_cue_aborts_and_joins_own_stream_before_returning(monkeypatch):
    writing = threading.Event()
    finish_write = threading.Event()
    events = []
    monkeypatch.setattr(
        "race_engineer.stt.radio_cues._load_cue_pcm", lambda cue: b"\0\0" * 48_000
    )

    class Stream:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            events.append("started")
            return self

        def __exit__(self, *args):
            events.append("closed")

        def write(self, data):
            events.append("write")
            writing.set()
            assert finish_write.wait(timeout=2)
            return False

        def abort(self):
            events.append("aborted")

    monkeypatch.setattr(
        "race_engineer.stt.radio_cues._sounddevice",
        lambda: SimpleNamespace(RawOutputStream=Stream),
    )

    async def run():
        task = asyncio.create_task(RadioCuePlayer().play("open"))
        assert await asyncio.to_thread(writing.wait, 1)
        task.cancel()
        await asyncio.sleep(0)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
        assert events == ["started", "write"]
        finish_write.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert events == ["started", "write", "aborted", "closed"]

    asyncio.run(run())


def test_muting_during_playback_aborts_remaining_cue(monkeypatch):
    muted = [False]
    events = []
    monkeypatch.setattr(
        "race_engineer.stt.radio_cues._load_cue_pcm", lambda cue: b"\0\0" * 1920
    )

    class Stream:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            events.append("closed")

        def write(self, data):
            events.append("write")
            muted[0] = True
            return False

        def abort(self):
            events.append("aborted")

    monkeypatch.setattr(
        "race_engineer.stt.radio_cues._sounddevice",
        lambda: SimpleNamespace(RawOutputStream=Stream),
    )
    assert not asyncio.run(RadioCuePlayer(muted=lambda: muted[0]).play("open"))
    assert events == ["write", "aborted", "closed"]


def test_live_capture_keeps_radio_claimed_through_both_cues():
    from race_engineer.application.live_conversation import _capture
    from race_engineer.core.speech_input import AudioClip, SpeechInputError
    from race_engineer.tts.live_radio import LiveRadio

    events = []
    clip = AudioClip(b"\0\0" * 1600)

    class Microphone:
        async def next_clip(self, *, before_capture, after_stream_started, after_capture):
            before_capture()
            events.append("input_started")
            await after_stream_started()
            events.extend(("recording", "input_stopped"))
            await after_capture()
            return clip

    async def run():
        radio = LiveRadio(None)
        radio.reset(7)

        class Cues:
            async def play(self, cue):
                with pytest.raises(SpeechInputError, match="radio_busy"):
                    radio.claim_capture()
                events.append(cue)
                return True

        audio, epoch = await _capture(
            Microphone(), SimpleNamespace(epoch=7, available=True), radio, cues=Cues()
        )
        assert audio is clip and epoch == 7
        # Once both cues and input cleanup finish, the next user can claim the radio.
        radio.claim_capture()
        radio.release_capture()
        await radio.aclose()

    asyncio.run(run())
    assert events == ["input_started", "open", "recording", "input_stopped", "close"]
