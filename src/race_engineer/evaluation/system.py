"""Best-effort, content-free metadata for explicit local evaluation runs."""

import json
import os
import platform
import shutil
import statistics
import subprocess
from hashlib import sha256
from pathlib import Path
from typing import Any


def _creation_flags() -> int:
    return subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0


def distribution(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "min": None, "mean": None, "p50": None, "p95": None, "max": None}
    ordered = sorted(values)

    def percentile(fraction: float) -> float:
        position = (len(ordered) - 1) * fraction
        lower = int(position)
        upper = min(lower + 1, len(ordered) - 1)
        weight = position - lower
        return ordered[lower] * (1 - weight) + ordered[upper] * weight

    return {
        "count": len(ordered),
        "min": round(ordered[0], 3),
        "mean": round(statistics.fmean(ordered), 3),
        "p50": round(percentile(0.5), 3),
        "p95": round(percentile(0.95), 3),
        "max": round(ordered[-1], 3),
    }


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def files_fingerprint(root: Path, paths: tuple[Path, ...]) -> str:
    """Hash file names and bytes so a dirty-tree run still identifies its exact inputs."""
    resolved_root = root.resolve()
    digest = sha256()
    for path in sorted({item.resolve() for item in paths}):
        if not path.is_file() or not path.is_relative_to(resolved_root):
            raise ValueError("fingerprinted files must be regular files inside the repository")
        relative = path.relative_to(resolved_root).as_posix().encode("utf-8")
        digest.update(relative)
        digest.update(b"\0")
        digest.update(bytes.fromhex(file_sha256(path)))
        digest.update(b"\0")
    return digest.hexdigest()


def directory_fingerprint(root: Path) -> str:
    files = tuple(path for path in root.rglob("*") if path.is_file())
    return files_fingerprint(root, files)


def _run(
    arguments: list[str],
    timeout_s: float = 5,
    *,
    cwd: Path | None = None,
) -> subprocess.CompletedProcess[str] | None:
    try:
        return subprocess.run(
            arguments,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_s,
            cwd=cwd,
            creationflags=_creation_flags(),
        )
    except (OSError, subprocess.TimeoutExpired):
        return None


def runtime_version(executable: Path, timeout_s: float) -> str | None:
    if not executable.is_file():
        return None
    result = _run([str(executable), "--version"], timeout_s)
    if result is None or result.returncode != 0:
        return None
    text = (result.stdout + "\n" + result.stderr).strip().replace("\r", "")
    return text[:1000] or None


def git_revision(root: Path) -> str | None:
    result = _run(["git", "rev-parse", "HEAD"], cwd=root)
    if result is None or result.returncode != 0:
        return None
    revision = result.stdout.strip()
    return revision if len(revision) == 40 else None


def git_dirty(root: Path) -> bool | None:
    result = _run(["git", "status", "--porcelain"], cwd=root)
    if result is None or result.returncode != 0:
        return None
    return bool(result.stdout.strip())


def _windows_hardware() -> dict[str, object] | None:
    powershell = shutil.which("powershell.exe") or shutil.which("powershell")
    if powershell is None:
        return None
    command = (
        "$cpu=((Get-ItemProperty -LiteralPath "
        "'Registry::HKEY_LOCAL_MACHINE\\HARDWARE\\DESCRIPTION\\System\\CentralProcessor\\0'"
        ").ProcessorNameString).Trim();"
        "Add-Type -AssemblyName Microsoft.VisualBasic;"
        "$ram=(New-Object Microsoft.VisualBasic.Devices.ComputerInfo).TotalPhysicalMemory;"
        "$video='Registry::HKEY_LOCAL_MACHINE\\SYSTEM\\CurrentControlSet\\Control\\Video';"
        "$gpus=@(Get-ChildItem -LiteralPath $video | ForEach-Object {"
        "Get-ChildItem -LiteralPath $_.PSPath -ErrorAction SilentlyContinue} | ForEach-Object {"
        "Get-ItemProperty -LiteralPath $_.PSPath -ErrorAction SilentlyContinue} | "
        "Where-Object {$_.DriverDesc -and $_.MatchingDeviceId -match '^pci\\\\'} | "
        "Select-Object "
        "@{Name='name';Expression={$_.DriverDesc}},"
        "@{Name='driver_version';Expression={$_.DriverVersion}} -Unique);"
        "[pscustomobject]@{cpu=$cpu;memory_bytes=[long]$ram;gpus=$gpus} | "
        "ConvertTo-Json -Compress -Depth 4"
    )
    result = _run([powershell, "-NoProfile", "-NonInteractive", "-Command", command], 10)
    if result is None or result.returncode != 0:
        return None
    try:
        value = json.loads(result.stdout)
    except (json.JSONDecodeError, TypeError):
        return None
    return value if isinstance(value, dict) else None


def machine_metadata() -> dict[str, object]:
    hardware = _windows_hardware() if os.name == "nt" else None
    return {
        "os": platform.platform(aliased=True, terse=True),
        "architecture": platform.machine(),
        "cpu": (hardware or {}).get("cpu") or platform.processor() or "unknown",
        "logical_cpus": os.cpu_count(),
        "memory_bytes": (hardware or {}).get("memory_bytes"),
        "gpus": (hardware or {}).get("gpus", []),
    }


def process_metrics(pid: int) -> dict[str, object]:
    """Return lifetime metrics without sampling during timed inference."""
    if os.name != "nt":
        return {"pid": pid, "available": False, "reason": "platform_not_implemented"}
    powershell = shutil.which("powershell.exe") or shutil.which("powershell")
    if powershell is None:
        return {"pid": pid, "available": False, "reason": "powershell_unavailable"}
    command = (
        f"Get-Process -Id {int(pid)} -ErrorAction Stop | "
        "Select-Object Id,CPU,WorkingSet64,PeakWorkingSet64 | ConvertTo-Json -Compress"
    )
    result = _run([powershell, "-NoProfile", "-NonInteractive", "-Command", command])
    if result is None or result.returncode != 0:
        return {"pid": pid, "available": False, "reason": "process_query_failed"}
    try:
        value: Any = json.loads(result.stdout)
        return {
            "pid": pid,
            "available": True,
            "cpu_seconds": value.get("CPU"),
            "working_set_bytes": value.get("WorkingSet64"),
            "peak_working_set_bytes": value.get("PeakWorkingSet64"),
        }
    except (json.JSONDecodeError, AttributeError, TypeError):
        return {"pid": pid, "available": False, "reason": "process_query_invalid"}


def gpu_process_memory(pid: int) -> dict[str, object]:
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return {"available": False, "reason": "nvidia_smi_unavailable"}
    result = _run(
        [
            executable,
            "--query-compute-apps=pid,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ]
    )
    if result is None or result.returncode != 0:
        return {"available": False, "reason": "gpu_process_query_failed"}
    memory_mib = 0
    for line in result.stdout.splitlines():
        columns = [part.strip() for part in line.split(",")]
        if len(columns) != 2:
            continue
        try:
            if int(columns[0]) == pid:
                memory_mib += int(columns[1])
        except ValueError:
            continue
    return {
        "available": True,
        "current_memory_mib": memory_mib,
        "peak_memory_mib": None,
        "utilization_percent": None,
    }
