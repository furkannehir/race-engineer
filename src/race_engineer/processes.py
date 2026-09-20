"""Cancellation-safe ownership of local child processes."""

import asyncio
from contextlib import suppress
from typing import Any


async def start_owned_process(*arguments: str, **options: Any) -> asyncio.subprocess.Process:
    """Never lose a newly spawned child if cancellation arrives during process creation."""
    launch = asyncio.create_task(asyncio.create_subprocess_exec(*arguments, **options))
    try:
        return await asyncio.shield(launch)
    except asyncio.CancelledError:
        try:
            process = await launch
        except Exception:
            raise asyncio.CancelledError from None
        if process.returncode is None:
            with suppress(ProcessLookupError):
                process.kill()
        await process.wait()
        raise
