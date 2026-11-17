"""Replay Ministral follow-ups with changing synthetic telemetry and saved traces."""

import argparse
import asyncio
import json
import statistics
from datetime import timedelta
from pathlib import Path

from race_engineer.conversation.runtime import ConversationServer
from race_engineer.core.enums import RaceFlag, SessionPhase
from race_engineer.evaluation.system import file_sha256
from race_engineer.fixtures import load_fixture
from race_engineer.intelligence.telemetry_memory import BoundedTelemetryMemory
from race_engineer.policy import DefaultRaceContextBuilder

if __package__:
    from .benchmark_qwen_cache import require_free_port
    from .compare_dialogue_models import CANDIDATES, ROOT, RawConversation, configuration
    from .lfm3_dialogue import LocalDialogueModel, NativeToolDialogue
    from .ministral_dialogue import MinistralConversation
else:
    from benchmark_qwen_cache import require_free_port
    from compare_dialogue_models import CANDIDATES, ROOT, RawConversation, configuration
    from lfm3_dialogue import LocalDialogueModel, NativeToolDialogue
    from ministral_dialogue import MinistralConversation

DATASET = ROOT / "fixtures/intelligence-evaluation/ministral-laps.json"


def advance(memory, update, *, builder=None):
    if not update:
        return  # Fixed snapshot, rather than fabricating flat gap history between turns.
    context = memory.latest_context()
    frame = context.frame
    position = update.get("position", frame.player.position)
    opponents = []
    positions = update.get("opponent_positions")
    if positions is not None and len(positions) != len(frame.opponents):
        raise ValueError("opponent position update must match the fixture's field")
    for index, opponent in enumerate(frame.opponents):
        gap = opponent.gap_to_player_s
        changes = {}
        if gap is not None and gap != 0:
            direction = "ahead" if gap < 0 else "behind"
            if direction in update:
                changes["gap_to_player_s"] = update[direction] * (-1 if gap < 0 else 1)
            if "position" in update:
                changes["position"] = max(1, position + (-1 if gap < 0 else 1))
        if positions is not None:
            changes["position"] = positions[index]
        opponents.append(opponent.model_copy(update=changes))
    player_changes = {key: update[key] for key in (
        "fuel_l", "speed_mps", "lap_number", "in_pit_lane", "laps_completed",
        "last_lap_time_s", "best_lap_time_s",
    ) if key in update}
    frame_changes = {}
    for name in ("session_type", "session_laps_total", "session_laps_remaining",
                 "session_time_remaining_s", "session_time_total_s"):
        if name in update:
            frame_changes[name] = update[name]
    if "flags" in update:
        frame_changes["flags"] = tuple(RaceFlag(value) for value in update["flags"])
    if "session_phase" in update:
        frame_changes["session_phase"] = SessionPhase(update["session_phase"])
    next_frame = frame.model_copy(update={
        "sequence": frame.sequence + 1, "observed_at": frame.observed_at + timedelta(seconds=1),
        "session_time_s": frame.session_time_s + 1,
        "player": frame.player.model_copy(update={"position": position, **player_changes}),
        "opponents": tuple(opponents),
        **frame_changes,
    })
    memory.update(builder.update(next_frame, ()) if builder is not None
                  else context.model_copy(update={"frame": next_frame}))


async def evaluate(args):
    output = args.output.resolve()
    if not output.is_relative_to(ROOT / "data") or output.exists():
        raise ValueError("Choose a new output JSON file under data/")
    candidate = CANDIDATES["ministral"]
    if file_sha256(candidate["path"]) != candidate["sha256"]:
        raise ValueError("Ministral checksum mismatch")
    config, settings = configuration(candidate, args.backend, args.port)
    require_free_port(config.port)
    dataset_path = (getattr(args, "dataset", None) or DATASET).resolve()
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    fixture_path = ROOT / dataset.get("fixture", "fixtures/synthetic/intelligence-capabilities")
    fixture = load_fixture(fixture_path)
    server = ConversationServer(ROOT, config)
    report = {"mode": "raw" if args.raw else "checked", "dataset": dataset["revision"],
              "model_sha256": candidate["sha256"], "rows": [],
              "scope": "typed_changing_synthetic_telemetry_no_audio_no_race"}
    sources = [Path(__file__), dataset_path, ROOT / "scripts/ministral_dialogue.py",
               ROOT / "scripts/ministral_telemetry.py",
               ROOT / "scripts/lfm3_dialogue.py", ROOT / "src/race_engineer/core/intelligence.py",
               ROOT / "src/race_engineer/intelligence/grounding.py",
               ROOT / "src/race_engineer/intelligence/capabilities.py",
               ROOT / "src/race_engineer/intelligence/race_progress.py",
               ROOT / "src/race_engineer/language/lap_times.py",
               ROOT / "src/race_engineer/core/contracts.py",
               ROOT / "src/race_engineer/intelligence/telemetry_memory.py",
               ROOT / "src/race_engineer/policy/context.py",
               *sorted(fixture_path.glob("*.json*"))]
    report["input_hashes"] = {str(path.relative_to(ROOT)): file_sha256(path) for path in sources}
    output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    try:
        await server.start()
        if server.process is None:
            raise ValueError("Evaluation requires an owned server")
        for repetition in range(args.repeats):
            for sequence in dataset["sequences"]:
                if args.language and sequence["language"] != args.language:
                    continue
                memory = BoundedTelemetryMemory()
                builder = DefaultRaceContextBuilder(settings)
                for frame in fixture.frames:
                    memory.update(builder.update(frame, ()))
                if args.raw:
                    dialogue = RawConversation(LocalDialogueModel(config, generation={
                        "temperature": 0.1, "seed": 7,
                    }), NativeToolDialogue(config, memory))
                else:
                    dialogue = MinistralConversation(config, memory)
                for case in sequence["turns"]:
                    advance(memory, case.get("update", {}), builder=builder)
                    row = await dialogue.respond(
                        case["question"], sequence["language"], f"followup-{len(report['rows'])}",
                    )
                    row.update(sequence=sequence["id"], repetition=repetition + 1,
                               expected=case, frame_sequence=memory.latest_context().frame.sequence)
                    row["delivered_reply"] = row.get("reply", row["raw_reply"])
                    report["rows"].append(row)
                    save()
                    print(json.dumps({
                        "n": len(report["rows"]), "question": row["question"],
                        "reply": row["delivered_reply"], "ms": row["duration_ms"],
                        "failure": row.get("failure"),
                    }, ensure_ascii=True), flush=True)
                    if any(word in row.get("failure", "") for word in ("timeout", "unreachable")):
                        raise RuntimeError("Transport failure; no subsequent requests queued")
    finally:
        await server.aclose()
        report["server_closed"] = server.process is None
        report["inputs_unchanged"] = report["input_hashes"] == {
            str(path.relative_to(ROOT)): file_sha256(path) for path in sources
        }
        report["completed"] = sum(row["delivered_reply"] is not None for row in report["rows"])
        if report["rows"]:
            report["median_ms"] = statistics.median(row["duration_ms"] for row in report["rows"])
        save()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("gpu", "cpu"), default="gpu")
    parser.add_argument("--raw", action="store_true", help="Original comparison dialogue control")
    parser.add_argument("--language", choices=("en", "tr"), help="Limit the development replay")
    parser.add_argument("--repeats", type=int, choices=range(1, 4), default=1)
    parser.add_argument("--port", type=int, default=18198)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset", type=Path,
                        help="Development corpus; defaults to the expanded telemetry workload")
    args = parser.parse_args()
    report = asyncio.run(evaluate(args))
    print(f"Completed {report['completed']}/{len(report['rows'])}; completion is not accuracy.")


if __name__ == "__main__":
    main()
