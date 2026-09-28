import asyncio
import subprocess
from types import SimpleNamespace

import pytest

from race_engineer import worker_lifecycle


@pytest.mark.parametrize(
    "platform,finished,killer_timeout",
    [
        ("nt", False, False),
        ("nt", False, True),
        ("nt", True, False),
        ("posix", False, False),
    ],
)
def test_only_owned_worker_tree_is_stopped(monkeypatch, platform, finished, killer_timeout):
    async def run():
        calls = []

        class Process:
            pid = 4242
            returncode = 0 if finished else None
            terminated = False
            killed = False

            def terminate(self):
                self.terminated = True
                self.returncode = 0

            def kill(self):
                self.killed = True
                self.returncode = 1

            async def wait(self):
                return self.returncode

        class Killer(Process):
            returncode = None

            async def wait(self):
                if killer_timeout and not self.killed:
                    raise TimeoutError()
                self.returncode = 0
                return 0

        async def create(*arguments, **kwargs):
            calls.append((arguments, kwargs))
            return Killer()

        monkeypatch.setattr(worker_lifecycle, "os", SimpleNamespace(name=platform))
        monkeypatch.setattr(subprocess, "CREATE_NO_WINDOW", 0x08000000, raising=False)
        monkeypatch.setattr(asyncio, "create_subprocess_exec", create)
        owned = Process()
        await worker_lifecycle.stop_owned_worker(owned)
        if platform == "nt" and not finished:
            assert calls[0][0] == ("taskkill.exe", "/PID", "4242", "/T", "/F")
            assert calls[0][1]["creationflags"] == subprocess.CREATE_NO_WINDOW
            assert owned.killed
        else:
            assert calls == []
            assert owned.terminated == (not finished)

    asyncio.run(run())
