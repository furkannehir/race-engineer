"""Explicit Windows worker RAM probe, separate from frozen semantic quality runs.

Windows venv python.exe is a launcher; its PID is not the inference process.
Resolve only its immediate owned child. No system-wide process dump is recorded.
"""

import argparse
import asyncio
import json
from datetime import timedelta
from pathlib import Path

from race_engineer.config import DialogueConfig
from race_engineer.conversation.context_view import ConversationContextAssembler
from race_engineer.conversation.judges import EncoderSemanticJudge
from race_engineer.core.dialogue import DialogueState
from race_engineer.evaluation.dialogue import EvalWorld
from race_engineer.evaluation.system import _run, process_metrics

ROOT = Path(__file__).resolve().parents[1]


def worker_pid(launcher_pid):
    command = (
        f'Get-CimInstance Win32_Process -Filter "ParentProcessId={int(launcher_pid)}" | '
        "Select-Object ProcessId,Name | ConvertTo-Json -Compress"
    )
    result = _run(["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", command], 10)
    if result is None or result.returncode or not result.stdout.strip():
        raise RuntimeError("Cannot identify owned Windows worker for RAM measurement")
    children = json.loads(result.stdout)
    children = children if isinstance(children, list) else [children]
    if len(children) != 1 or children[0]["Name"].lower() != "python.exe":
        raise RuntimeError("Unexpected owned process tree; refuse to guess worker PID")
    return int(children[0]["ProcessId"])


async def run(args):
    if args.output.exists():
        raise ValueError("Refusing to overwrite a memory probe")
    encoder = EncoderSemanticJudge(
        python=ROOT / "data/stt-prototype/runtime/Scripts/python.exe",
        worker=ROOT / "scripts/semantic_worker.py",
        model_path=ROOT / "data/dialogue-prototype" / args.candidate,
        package_path=ROOT / "data/dialogue-prototype/packages",
        candidate=args.candidate,
    )
    actual_pid = None
    try:
        await encoder.start()
        actual_pid = worker_pid(encoder.process.pid)
        loaded = process_metrics(actual_pid)
        world = EvalWorld()
        request = ConversationContextAssembler(DialogueConfig()).assemble(
            world.snapshot(),
            DialogueState(session_id="dialogue-evaluation"),
            turn_id="ram-probe",
            question="How many liters are left in our tank?",
            received_at=world.now,
            deadline=world.now + timedelta(seconds=30),
            asr_language=None,
            reply_language=None,
        )
        await encoder.judge(request)
        report = {
            "schema_version": "semantic-memory-probe.v1",
            "candidate": args.candidate,
            "scope": "Model load plus one calibration utterance, not full-suite peak RAM.",
            "startup_ms": encoder.startup_ms,
            "loaded_worker": loaded,
            "after_one_turn": process_metrics(actual_pid),
            "runtime": encoder.metadata,
        }
    finally:
        await encoder.aclose()
    if actual_pid is not None:
        for _ in range(10):
            remaining = process_metrics(actual_pid)
            if not remaining.get("available"):
                break
            await asyncio.sleep(0.2)
        report["worker_exited"] = not remaining.get("available")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", choices=["minilm", "laya"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    asyncio.run(run(parser.parse_args()))
