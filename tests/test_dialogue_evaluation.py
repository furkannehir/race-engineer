"""Evaluation and adapter tests use no model downloads or optional ML dependencies."""

import asyncio
import json
from datetime import timedelta
from pathlib import Path

import pytest

from race_engineer.config import ConversationConfig, DialogueConfig
from race_engineer.conversation.context_view import ConversationContextAssembler
from race_engineer.conversation.judges import EncoderSemanticJudge, QwenSemanticJudge
from race_engineer.conversation.router import JudgeRouter
from race_engineer.conversation.semantic import compose_scores, hypotheses, semantic_view
from race_engineer.core.dialogue import DialogueState, SemanticJudgeError, SemanticProposal
from race_engineer.evaluation.dialogue import (
    EvalWorld,
    audit_splits,
    evaluate_dialogues,
    load_dialogues,
    ratio_interval,
    semantic_correct,
)
from race_engineer.testing.dialogue import ScriptedSemanticJudge

ROOT = Path(__file__).resolve().parents[1]


def request():
    world = EvalWorld()
    return ConversationContextAssembler(DialogueConfig()).assemble(
        world.snapshot(),
        DialogueState(session_id="dialogue-evaluation"),
        turn_id="turn-1",
        question="Where are we?",
        received_at=world.now,
        deadline=world.now + timedelta(seconds=30),
        asr_language=None,
        reply_language=None,
    )


class FixtureOracle:
    def __init__(self, dataset):
        self.turns = [
            turn for scenario in dataset.scenarios for turn in (*scenario.en, *scenario.tr)
        ]

    async def judge(self, item):
        matches = [t for t in self.turns if t.question == item.question]
        preferred = [t for t in matches if t.acceptable[0].language == item.default_language]
        return (preferred or matches)[0].acceptable[0]


class AlwaysAbstain:
    async def judge(self, item):
        return SemanticProposal(
            language=item.reply_language or item.default_language, model_id="control", abstain=True
        )


@pytest.mark.parametrize("split,count", [("calibration", 48), ("locked", 70)])
def test_frozen_oracle_is_perfect_and_content_free(split, count):
    dataset = load_dialogues(ROOT / f"fixtures/dialogue/{split}.json")
    rows, summary = asyncio.run(evaluate_dialogues(dataset, FixtureOracle(dataset)))
    assert len(rows) == count
    assert all(row["passed"] for row in rows)
    assert summary["whole_dialogues"]["rate"] == 1
    output = json.dumps(rows)
    assert dataset.scenarios[0].en[0].question not in output
    assert summary["overall"]["guard_only_turns"] == (2 if split == "locked" else 0)


def test_abstain_cannot_qualify_through_accuracy():
    dataset = load_dialogues(ROOT / "fixtures/dialogue/locked.json")
    _, summary = asyncio.run(evaluate_dialogues(dataset, AlwaysAbstain()))
    assert summary["overall"]["answerable_coverage"]["rate"] == 0
    assert summary["overall"]["accepted_plan_accuracy"]["rate"] is None
    assert summary["overall"]["abstentions"] == 68


def test_split_audit_rejects_relabeling_and_literal_leakage():
    cal = load_dialogues(ROOT / "fixtures/dialogue/calibration.json")
    locked = load_dialogues(ROOT / "fixtures/dialogue/locked.json")
    assert audit_splits((cal, locked))["literal_cross_split_duplicates"] == 0
    with pytest.raises(ValueError, match="family"):
        audit_splits((cal, cal.model_copy(update={"split": "locked"})))
    scenario = cal.scenarios[0].model_copy(update={"family": "brand-new-id"})
    with pytest.raises(ValueError, match="literal"):
        audit_splits((cal, locked.model_copy(update={"scenarios": (scenario,)})))
    with pytest.raises(ValueError, match="encode"):
        audit_splits(
            (
                cal.model_copy(
                    update={
                        "scenarios": (scenario.model_copy(update={"family": "locked-new-name"}),)
                    }
                ),
            )
        )


def test_joint_semantics_does_not_ignore_dropped_or_extra_parts():
    dataset = load_dialogues(ROOT / "fixtures/dialogue/locked.json")
    turn = next(s for s in dataset.scenarios if s.family == "range-versus-quantity").en[0]
    expected = turn.acceptable[0]
    assert semantic_correct(expected, turn)
    assert not semantic_correct(
        expected.model_copy(update={"requests": expected.requests[:1]}), turn
    )
    assert not semantic_correct(expected.model_copy(update={"acts": ("acknowledge",)}), turn)
    assert semantic_correct(
        expected.model_copy(
            update={
                "requests": tuple(reversed(expected.requests)),
                "model_id": "other",
                "calibration_id": "v9",
            }
        ),
        turn,
    )


def test_independent_truth_rejects_wrong_values_and_car():
    from race_engineer.core.dialogue import GroundedAnswer, OpponentReference

    world = EvalWorld()
    answer = GroundedAnswer(
        part_id="p",
        query="gap_behind",
        status="available",
        value=2.4,
        unit="s",
        source_sequence=1,
        opponent=OpponentReference(side="behind", driver_id="car-b"),
    )
    assert world.facts_valid([answer])
    assert not world.facts_valid([answer.model_copy(update={"value": 99})])
    assert not world.facts_valid(
        [
            answer.model_copy(
                update={
                    "opponent": OpponentReference(side="behind", driver_id="different"),
                }
            )
        ]
    )
    world.stale = True
    assert not world.facts_valid([answer])


def test_wilson_reports_counts_and_uncertainty():
    assert ratio_interval(0, 0)["rate"] is None
    result = ratio_interval(10, 10)
    assert result["rate"] == 1 and result["wilson95"][0] < 0.95


def good_scores():
    scores = dict.fromkeys(hypotheses(), 0.01)
    for key in (
        "mode/request",
        "act/none",
        "language/en",
        "reference/none",
        "clarification/none",
        "query/position",
    ):
        scores[key] = 0.99
    return scores


def compose(scores):
    return compose_scores(
        request(),
        scores,
        model_id="fake",
        threshold=0.65,
        margin=0.1,
        calibration_id="test-calibration",
    )


def test_encoder_composition_and_whole_plan_abstention():
    assert compose(good_scores()).requests[0].query == "position"
    scores = good_scores()
    scores["query/lap"] = 0.6
    assert compose(scores).abstain
    scores = good_scores()
    scores["act/close"], scores["act/none"] = 0.99, 0.01
    assert compose(scores).abstain  # cannot close and answer position


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -1, 2])
def test_encoder_rejects_invalid_scores(invalid):
    scores = good_scores()
    scores["query/position"] = invalid
    with pytest.raises(ValueError):
        compose(scores)


def test_context_projection_never_includes_fixture_expectations_or_old_numbers():
    view = semantic_view(request())
    assert "acceptable" not in view and "value" not in json.dumps(view)
    assert view["current_utterance"] == "Where are we?"
    assert "topic" in view and "capabilities" in view


def test_router_fallback_once_only_on_uncertainty_or_invalid():
    async def run():
        req = request()
        accepted = compose(good_scores())
        fallback = ScriptedSemanticJudge(accepted)
        router = JudgeRouter(
            ScriptedSemanticJudge(
                SemanticProposal(language="en", model_id="uncertain", abstain=True)
            ),
            fallback,
        )
        assert await router.judge(req) == accepted
        assert router.attempts == 2 and router.fallbacks == 1
        router = JudgeRouter(ScriptedSemanticJudge(accepted), fallback)
        assert await router.judge(req) == accepted
        assert router.attempts == 1 and router.fallbacks == 0

    asyncio.run(run())


@pytest.mark.parametrize(
    "proposal",
    [
        {"clarification": "topic"},
        {"requests": [{"part_id": "q1", "query": "fuel_to_finish"}]},
        {"requests": [{"part_id": "q1", "query": "gap", "reference": "unspecified"}]},
    ],
)
def test_router_does_not_fallback_for_missing_reference_or_data(proposal):
    primary = SemanticProposal.model_validate({"language": "en", "model_id": "test", **proposal})
    router = JudgeRouter(ScriptedSemanticJudge(primary), ScriptedSemanticJudge())
    assert asyncio.run(router.judge(request())) == primary
    assert router.fallbacks == 0


def test_router_deadline_no_attempts():
    req = request()
    req = req.model_copy(update={"deadline": req.received_at - timedelta(seconds=1)})
    router = JudgeRouter(ScriptedSemanticJudge())
    with pytest.raises(SemanticJudgeError, match="deadline"):
        asyncio.run(router.judge(req))
    assert router.attempts == 0


class FakeWriter:
    def __init__(self):
        self.written = b""
        self.closed = False

    def write(self, data):
        self.written += data

    async def drain(self):
        pass

    def close(self):
        self.closed = True

    async def wait_closed(self):
        pass


@pytest.mark.parametrize(
    "failure", [None, "redirect", "duplicate", "truncated", "extra", "incomplete"]
)
def test_qwen_loopback_protocol_validation(monkeypatch, failure):
    async def run():
        result = compose(good_scores()).model_dump(
            mode="json",
            exclude={
                "schema_version",
                "model_id",
                "calibration_id",
            },
        )
        if failure == "extra":
            result["text"] = "invented radio text"
        content = json.dumps(result)
        if failure == "duplicate":
            content = content.replace('"language": "en"', '"language":"en","language":"tr"')
        body = json.dumps(
            {
                "choices": [
                    {
                        "finish_reason": "length" if failure == "incomplete" else "stop",
                        "message": {"content": content},
                    }
                ]
            }
        ).encode()
        reader = asyncio.StreamReader()
        status = b"302 Found" if failure == "redirect" else b"200 OK"
        reader.feed_data(
            b"HTTP/1.1 "
            + status
            + b"\r\nContent-Length: "
            + str(len(body)).encode()
            + b"\r\n\r\n"
            + (body[:5] if failure == "truncated" else body)
        )
        reader.feed_eof()
        writer = FakeWriter()

        async def connect(host, port):
            assert host == "127.0.0.1"
            return reader, writer

        monkeypatch.setattr(asyncio, "open_connection", connect)
        judge = QwenSemanticJudge(ConversationConfig())
        if failure:
            with pytest.raises(SemanticJudgeError):
                await judge.judge(request())
        else:
            assert (await judge.judge(request())).requests[0].query == "position"
        assert writer.closed
        assert b"POST /v1/chat/completions" in writer.written

    asyncio.run(run())


def test_encoder_missing_artifacts_do_not_download(tmp_path):
    encoder = EncoderSemanticJudge(
        python=tmp_path / "missing",
        worker=tmp_path / "worker",
        model_path=tmp_path / "weights",
        candidate="minilm",
    )
    with pytest.raises(SemanticJudgeError, match="artifacts_missing"):
        asyncio.run(encoder.start())


@pytest.mark.parametrize("mode", ["normal", "context", "timeout", "cancel", "mismatch"])
def test_owned_encoder_worker_lifecycle(tmp_path, mode, monkeypatch):
    async def run():
        class Process:
            returncode = None
            stdin = FakeWriter()
            stdout = asyncio.StreamReader()
            terminated = False

            def terminate(self):
                self.terminated = True
                self.returncode = 1

            async def wait(self):
                return self.returncode

        proc = Process()

        async def stop_owned(process):
            process.terminate()
            await process.wait()

        monkeypatch.setattr(
            "race_engineer.conversation.judges.stop_owned_worker", stop_owned
        )
        if mode in {"normal", "context", "mismatch"}:
            payload = {"id": "wrong" if mode == "mismatch" else "turn-1", "scores": good_scores()}
            if mode == "context":
                payload = {"id": "turn-1", "error": "context_limit"}
            proc.stdout.feed_data(json.dumps(payload).encode() + b"\n")
        encoder = EncoderSemanticJudge(
            python=tmp_path,
            worker=tmp_path,
            model_path=tmp_path,
            candidate="minilm",
            timeout_s=0.02,
        )
        encoder.process = proc
        if mode == "cancel":
            task = asyncio.create_task(encoder.judge(request()))
            await asyncio.sleep(0)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        elif mode == "normal":
            assert (await encoder.judge(request())).requests[0].query == "position"
        else:
            with pytest.raises(SemanticJudgeError):
                await encoder.judge(request())
        assert proc.terminated == (mode in {"timeout", "cancel", "mismatch"})
        if mode in {"normal", "context"}:
            await encoder.aclose()
            assert proc.terminated

    asyncio.run(run())


def test_routing_assessment_never_promotes_oracle_or_small_pilot():
    from race_engineer.evaluation.routing import assess_candidate

    dataset = load_dialogues(ROOT / "fixtures/dialogue/locked.json")
    _, summary = asyncio.run(evaluate_dialogues(dataset, FixtureOracle(dataset)))
    assessment = assess_candidate(
        {"candidate": "oracle", "split": "locked", "synthetic_control": True, "summary": summary}
    )
    assert not assessment["promote_to_live"]
    assert "not_locked_model_evidence" in assessment["reasons"]
    assert "en:insufficient_independent_screening_volume" in assessment["reasons"]
    assessment = assess_candidate(
        {"candidate": "minilm", "split": "locked", "synthetic_control": False, "summary": summary}
    )
    assert not assessment["promote_to_live"]
    assert "no_qualifying_calibration" in assessment["reasons"]


def test_optional_model_packages_are_not_imported_by_adapters():
    import subprocess
    import sys

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import race_engineer.conversation.judges; "
            "assert not {'torch', 'transformers', 'laya'} & set(sys.modules)",
        ],
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0
