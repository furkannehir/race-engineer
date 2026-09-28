import asyncio
from types import SimpleNamespace

import pytest

from race_engineer.stt.radio_cues import RadioCuePlayer, render_radio_cue


def test_packaged_radio_cues_are_distinct_pcm16_at_selected_lengths() -> None:
    opened = render_radio_cue("open", 0.2)
    closed = render_radio_cue("close", 0.2)
    assert len(opened) == round(48_000 * 0.25) * 2
    assert len(closed) == round(48_000 * 0.30) * 2
    assert opened != closed and any(opened) and any(closed)
    assert render_radio_cue("open", 0) == bytes(len(opened))
    with pytest.raises(ValueError):
        render_radio_cue("open", 2)


def test_cue_player_respects_output_device_and_mute(monkeypatch) -> None:
    writes = []

    class Stream:
        def __init__(self, **kwargs):
            writes.append((kwargs, None))

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def write(self, pcm):
            writes[-1] = (writes[-1][0], pcm)

    monkeypatch.setattr(
        "race_engineer.stt.radio_cues._sounddevice",
        lambda: SimpleNamespace(RawOutputStream=Stream),
    )
    muted = [False]
    player = RadioCuePlayer(output_device=7, volume=0.2, muted=lambda: muted[0])

    async def run():
        assert await player.play("open")
        muted[0] = True
        assert not await player.play("close")

    asyncio.run(run())
    assert len(writes) == 1
    assert writes[0][0]["device"] == 7
    assert writes[0][0]["samplerate"] == 48_000
    assert writes[0][1] == render_radio_cue("open", 0.2)


def test_cue_failure_disables_future_attempts_without_raising(monkeypatch) -> None:
    attempts = []

    def unavailable():
        attempts.append(True)
        raise OSError("missing output")

    monkeypatch.setattr("race_engineer.stt.radio_cues._sounddevice", unavailable)
    player = RadioCuePlayer()

    async def run():
        assert not await player.play("open")
        assert not await player.play("close")

    asyncio.run(run())
    assert len(attempts) == 1
