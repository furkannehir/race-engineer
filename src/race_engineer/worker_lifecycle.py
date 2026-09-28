"""Stop only an owned inference worker and its Windows venv-launcher children."""

import asyncio
import os
import subprocess
from contextlib import suppress


async def stop_owned_worker(process: asyncio.subprocess.Process) -> None:
    if process.returncode is None and os.name == "nt":
        killer: asyncio.subprocess.Process | None = None
        try:
            # Windows venv python.exe redirects to a child interpreter. Terminating
            # only the launcher may leave expensive inference alive until pipe EOF.
            killer = await asyncio.create_subprocess_exec(
                "taskkill.exe",
                "/PID",
                str(process.pid),
                "/T",
                "/F",
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
                creationflags=subprocess.CREATE_NO_WINDOW,
            )
            await asyncio.wait_for(killer.wait(), 3)
        except (OSError, TimeoutError):
            if killer is not None and killer.returncode is None:
                with suppress(ProcessLookupError):
                    killer.kill()
                await killer.wait()
        finally:
            # Also reap the launched process, even when taskkill reported failure.
            if process.returncode is None:
                with suppress(ProcessLookupError):
                    process.kill()
    elif process.returncode is None:
        with suppress(ProcessLookupError):
            process.terminate()
    try:
        await asyncio.wait_for(process.wait(), 3)
    except TimeoutError:
        with suppress(ProcessLookupError):
            process.kill()
        await process.wait()
