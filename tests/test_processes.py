import asyncio

import pytest
from test_stt_worker import FakeProcess

from race_engineer.processes import start_owned_process


def test_cancel_during_spawn_waits_for_child_and_kills_it(monkeypatch):
    async def run():
        spawning = asyncio.Event()
        release = asyncio.Event()
        child = FakeProcess([])

        async def create(*args, **kwargs):
            spawning.set()
            await release.wait()
            return child

        monkeypatch.setattr("race_engineer.processes.asyncio.create_subprocess_exec", create)
        task = asyncio.create_task(start_owned_process("local-worker"))
        await spawning.wait()
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert child.killed

    asyncio.run(run())


def test_failed_spawn_after_cancellation_preserves_cancellation(monkeypatch):
    async def run():
        spawning = asyncio.Event()
        release = asyncio.Event()

        async def create(*args, **kwargs):
            spawning.set()
            await release.wait()
            raise OSError("spawn failed")

        monkeypatch.setattr("race_engineer.processes.asyncio.create_subprocess_exec", create)
        task = asyncio.create_task(start_owned_process("local-worker"))
        await spawning.wait()
        task.cancel()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(run())
