"""Opt-in real-model evaluation; synthetic questions only, no user recording."""

import argparse
import asyncio
import json
import time
from pathlib import Path

from race_engineer.config import load_config
from race_engineer.conversation.factory import conversation_planner
from race_engineer.conversation.replay import ReplayRaceState
from race_engineer.conversation.session import ConversationSession
from race_engineer.fixtures import load_fixture

ROOT = Path(__file__).resolve().parents[1]


async def evaluate(config_path: Path, cases_path: Path) -> int:
    config = load_config(config_path)
    cases = json.loads(cases_path.read_text(encoding="utf-8"))
    planner = conversation_planner(config.conversation)
    results = []
    for case in cases:
        state = ReplayRaceState(load_fixture(ROOT / "fixtures/synthetic/conversation"))
        session = ConversationSession(planner, state.snapshot, config.conversation)
        for index, turn in enumerate(case["turns"]):
            if "frame_index" in turn:
                state.seek(turn["frame_index"])
            started = time.perf_counter()
            reply = await session.ask(turn["question"])
            elapsed_ms = round((time.perf_counter() - started) * 1000)
            queries = sorted(answer.query.value for answer in reply.answers)
            passed = (
                queries == sorted(turn["queries"])
                and reply.language == turn["language"]
                and reply.status == turn["status"]
            )
            result = {
                "case": case["id"],
                "turn": index + 1,
                "passed": passed,
                "elapsed_ms": elapsed_ms,
                "queries": queries,
                "reply": reply.model_dump(mode="json"),
            }
            results.append(result)
            print(json.dumps(result, ensure_ascii=False), flush=True)
    passed_count = sum(result["passed"] for result in results)
    print(json.dumps({"passed": passed_count, "total": len(results)}, sort_keys=True))
    return 0 if passed_count == len(results) else 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config/default.toml")
    parser.add_argument("--cases", type=Path, default=ROOT / "fixtures/conversation/cases.json")
    args = parser.parse_args()
    raise SystemExit(asyncio.run(evaluate(args.config, args.cases)))


if __name__ == "__main__":
    main()
