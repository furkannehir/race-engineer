import asyncio
import json
import logging
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from race_engineer.config import ConversationConfig, DriverInterpreterConfig, load_config
from race_engineer.core.contracts import PlayerState, RaceContext, TelemetryFrame
from race_engineer.core.intelligence import ContextPlan, DriverTurn
from race_engineer.intelligence.driver_action_client import (
    DriverInterpreterError,
    LocalDriverActionClassifier,
)
from race_engineer.intelligence.driver_action_planner import (
    ActionAssistedContextPlanner,
    live_action_input,
    social_fast_action,
)
from race_engineer.intelligence.driver_actions import (
    LABELS,
    ActionPrediction,
    ActionRequest,
    load_action_metadata,
)
from race_engineer.intelligence.factory import live_intelligence
from race_engineer.intelligence.local_model import LocalIntelligenceError
from race_engineer.intelligence.telemetry_memory import BoundedTelemetryMemory
from race_engineer.radio_diagnostics import RadioTrace

ROOT = Path(__file__).parents[1]


def turn(text="Are you there?", history=(), language="en"):
    return DriverTurn(
        turn_id="trial-turn",
        transcript=text,
        session_id="trial-session",
        generation=1,
        received_at=datetime.now(UTC),
        asr_language=language,
        recent_dialogue=history,
    )


def prediction(*, truncated=False, **scores):
    return ActionPrediction(
        scores=[float(scores.get(label, 0.01)) for label in LABELS],
        threshold=0.7,
        duration_ms=8.0,
        input_truncated=truncated,
    )


class Classifier:
    def __init__(self, result=None, error=None):
        self.result = result or prediction(radio_check=0.99)
        self.error = error
        self.starts = self.closes = 0
        self.requests = []

    async def start(self):
        self.starts += 1
        if self.error:
            raise DriverInterpreterError(self.error)

    async def predict(self, request):
        self.requests.append(request)
        return self.result

    async def aclose(self):
        self.closes += 1


class Planner:
    def __init__(self):
        self.calls = []

    async def plan(self, request, signals, capabilities):
        self.calls.append((request, signals, capabilities))
        return ContextPlan(turn_id=request.turn_id, planner_id="fallback")


@pytest.mark.parametrize("action", ["radio_check", "greeting", "vent"])
def test_clear_single_social_action_skips_only_the_context_planner(action):
    async def run():
        classifier = Classifier(prediction(**{action: 0.99}))
        fallback = Planner()
        hybrid = ActionAssistedContextPlanner(classifier, fallback)
        await hybrid.start()
        plan = await hybrid.plan(turn(), (), ())
        assert plan.temporal_scope == "social"
        assert plan.situation == ("driver_social_turn", f"driver_{action}")
        assert plan.queries == plan.capability_requests == plan.unknowns == ()
        assert not fallback.calls
        await hybrid.plan(turn(), (), ())
        assert classifier.starts == 1
        assert len(classifier.requests) == 2
        await hybrid.aclose()
        assert classifier.closes == 1

    asyncio.run(run())


@pytest.mark.parametrize(
    "scores",
    [
        {"request_information": 0.99},
        {"vent": 0.99, "request_information": 0.8},
        {"vent": 0.99, "request_information": 0.4},
        {"radio_check": 0.99, "needs_clarification": 0.5},
        {"radio_check": 0.99, "needs_clarification": 0.9},
        {"vent": 0.99, "report_problem": 0.8},
        {"vent": 0.8},
        {"greeting": 0.9, "correction": 0.4},
        {"greeting": 0.9, "communication_feedback": 0.4},
        {"greeting": 0.9, "preference_change": 0.4},
        {"greeting": 0.9, "out_of_scope": 0.4},
        {"greeting": 0.9, "unclear": 0.4},
        {"report_problem": 0.99},
        {"acknowledge": 0.99},
        {"correction": 0.99},
        {"preference_change": 0.99},
        {"communication_feedback": 0.99},
        {"out_of_scope": 0.99},
        {},
    ],
)
def test_unsupported_uncertain_or_mixed_actions_preserve_full_qwen_request(scores):
    async def run():
        fallback = Planner()
        hybrid = ActionAssistedContextPlanner(Classifier(prediction(**scores)), fallback)
        request = turn("That was dirty. How much fuel do we have?")
        signals, capabilities = [object()], [object()]
        result = await hybrid.plan(request, signals, capabilities)
        assert result.planner_id == "fallback"
        assert fallback.calls == [(request, signals, capabilities)]

    asyncio.run(run())


def test_truncated_input_always_uses_full_planner():
    action, reason = social_fast_action(prediction(radio_check=0.99, truncated=True))
    assert action is None and reason == "input_truncated"


def test_declared_guard_allows_v2_development_vent_without_treating_it_as_verified_fact():
    action, _ = social_fast_action(
        prediction(
            vent=0.8778,
            report_problem=0.6312,
            request_information=0.1061,
            unclear=0.2167,
            needs_clarification=0.2625,
        )
    )
    assert action == "vent"


def test_history_format_matches_training_with_a_bounded_role_separated_tail():
    history = tuple(
        f"{role}: message {index}" for index in range(10) for role in ("driver", "engineer")
    )
    text = live_action_input(turn("Can we make it?", history))
    assert text.startswith("Recent dialogue:\ndriver: message 7\n")
    assert text.endswith("engineer: message 9\nCurrent driver message:\nCan we make it?")
    assert "message 6" not in text
    assert live_action_input(turn(history=("arbitrary unknown role",))) is None


def test_unknown_history_format_uses_qwen_not_a_context_free_classifier_guess():
    async def run():
        classifier, fallback = Classifier(), Planner()
        hybrid = ActionAssistedContextPlanner(classifier, fallback)
        request = turn(history=("unrecognized history entry",))
        assert (await hybrid.plan(request, (), ())).planner_id == "fallback"
        assert not classifier.requests

    asyncio.run(run())


def test_optional_startup_failure_falls_back_once_and_does_not_end_the_session():
    async def run():
        classifier, fallback = (
            Classifier(error="driver_interpreter_installation_missing"),
            Planner(),
        )
        hybrid = ActionAssistedContextPlanner(classifier, fallback)
        for _ in range(2):
            assert (await hybrid.plan(turn(), (), ())).planner_id == "fallback"
        assert classifier.starts == 1 and not classifier.requests
        assert len(fallback.calls) == 2
        await hybrid.aclose()
        assert classifier.closes == 1

    asyncio.run(run())


def test_inference_failure_disables_experiment_and_falls_back_without_retrying_each_turn():
    class FailingClassifier(Classifier):
        async def predict(self, request):
            self.requests.append(request)
            raise DriverInterpreterError("driver_interpreter_timeout")

    async def run():
        classifier, fallback = FailingClassifier(), Planner()
        hybrid = ActionAssistedContextPlanner(classifier, fallback)
        for _ in range(2):
            assert (await hybrid.plan(turn(), (), ())).planner_id == "fallback"
        assert len(classifier.requests) == 1
        assert len(fallback.calls) == 2

    asyncio.run(run())


def test_action_and_route_timings_are_logged_without_driver_or_engineer_text(caplog):
    caplog.set_level(logging.INFO)

    async def run():
        with RadioTrace("run", "trace").bind():
            hybrid = ActionAssistedContextPlanner(Classifier(), Planner())
            await hybrid.plan(turn("private driver text", ("engineer: private reply",)), (), ())

    asyncio.run(run())
    stages = [row for row in caplog.records if getattr(row, "stage", None) == "driver_interpreter"]
    assert stages[-1].duration_ms >= 0
    route = next(
        row for row in caplog.records if getattr(row, "event", None) == "radio_interpreter_route"
    )
    assert route.route == "social_fast_path" and route.context_planner_skipped
    for record in caplog.records:
        assert "private driver text" not in str(record.__dict__)
        assert "private reply" not in str(record.__dict__)


def test_config_is_opt_in_but_repository_race_trial_is_enabled():
    assert ConversationConfig().driver_interpreter.enabled is False
    settings = load_config(ROOT / "config/default.toml", environ={}).conversation.driver_interpreter
    assert settings.enabled and settings.model_path == Path(
        "data/driver-interpreter-pilot/run-v2-seed17"
    )


def test_disabled_factory_never_constructs_optional_cpu_runtime(monkeypatch):
    def forbidden(*args):
        pytest.fail("disabled classifier must not be constructed")

    monkeypatch.setattr("race_engineer.intelligence.factory.LocalDriverActionClassifier", forbidden)
    intelligence = live_intelligence(ConversationConfig(), BoundedTelemetryMemory())
    asyncio.run(intelligence.start())
    asyncio.run(intelligence.aclose())


def test_prediction_rejects_malformed_numeric_protocol():
    raw = prediction(radio_check=0.9).model_dump()
    for scores in ([0.1], [float("nan")] * 12, [float("inf")] * 12, [1.1] * 12, [True] * 12):
        with pytest.raises(ValidationError):
            ActionPrediction.model_validate({**raw, "scores": scores})


def worker_config(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    data = tmp_path / "data"
    data.mkdir()
    python = data / "python.exe"
    python.touch()
    base, model = data / "base", data / "model"
    base.mkdir()
    model.mkdir()
    metadata = {
        "schema_version": "driver-action-pilot.v1",
        "labels": list(LABELS),
        "base_repo": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        "base_revision": "e8f8c211226b894fcb81acc59f3b34ba3efd5f42",
        "selected_candidate": "fine_tuned_encoder",
        "selected_by": "validation_exact_match_ties_prefer_frozen",
        "threshold": 0.7,
        "adapter_file": "selected-adapter.safetensors",
        "trained_scope": "conversational_actions_and_clarification_only",
        "draft_labels": True,
        "promoted": False,
    }
    (model / "model.json").write_text(json.dumps(metadata), encoding="utf-8")
    return DriverInterpreterConfig(
        enabled=True,
        python_path=python,
        base_model_path=base,
        model_path=model,
        timeout_s=0.02,
        startup_timeout_s=0.02,
    ), metadata


class Process:
    def __init__(self, metadata, *, ready=True, respond=True, wrong_id=False):
        self.returncode = None
        self.killed = self.waited = False
        self.lines = asyncio.Queue()
        self.stdout = self.stdin = self
        self.requests = []
        self.respond, self.wrong_id = respond, wrong_id
        if ready:
            self.lines.put_nowait(
                json.dumps({"ready": True, "metadata": metadata}).encode() + b"\n"
            )

    async def readline(self):
        return await self.lines.get()

    def write(self, payload):
        request = json.loads(payload)
        self.requests.append(request)
        if self.respond:
            self.lines.put_nowait(
                json.dumps(
                    {
                        "request_id": "wrong" if self.wrong_id else request["request_id"],
                        "result": prediction(radio_check=0.99).model_dump(),
                    }
                ).encode()
                + b"\n"
            )

    async def drain(self):
        pass

    def kill(self):
        self.killed = True
        self.returncode = 0

    async def wait(self):
        self.waited = True
        return self.returncode


def test_persistent_worker_is_offline_hidden_and_reused_until_owned_shutdown(monkeypatch, tmp_path):
    config, metadata = worker_config(monkeypatch, tmp_path)
    launches = []

    async def run():
        process = Process(metadata)

        async def spawn(*args, **options):
            launches.append((args, options))
            return process

        monkeypatch.setattr(
            "race_engineer.intelligence.driver_action_client.start_owned_process", spawn
        )
        client = LocalDriverActionClassifier(config)
        await client.start()
        for index in range(2):
            await client.predict(ActionRequest(request_id=str(index), input_text="Seni duyuyorum."))
        assert len(launches) == 1 and len(process.requests) == 2
        assert process.requests[0]["input_text"] == "Seni duyuyorum."
        assert launches[0][1]["env"]["HF_HUB_OFFLINE"] == "1"
        assert launches[0][1]["env"]["TRANSFORMERS_OFFLINE"] == "1"
        await client.aclose()
        assert process.killed and process.waited

    asyncio.run(run())


@pytest.mark.parametrize(
    ("ready", "respond", "wrong_id", "reason"),
    [
        (False, True, False, "driver_interpreter_startup_timeout"),
        (True, False, False, "driver_interpreter_timeout"),
        (True, True, True, "driver_interpreter_protocol_error"),
    ],
)
def test_worker_failures_kill_and_reap_owned_process(
    monkeypatch,
    tmp_path,
    ready,
    respond,
    wrong_id,
    reason,
):
    config, metadata = worker_config(monkeypatch, tmp_path)

    async def run():
        process = Process(metadata, ready=ready, respond=respond, wrong_id=wrong_id)

        async def spawn(*args, **options):
            return process

        monkeypatch.setattr(
            "race_engineer.intelligence.driver_action_client.start_owned_process", spawn
        )
        client = LocalDriverActionClassifier(config)
        with pytest.raises(DriverInterpreterError, match=reason):
            await client.predict(ActionRequest(request_id="trial", input_text="Radio check."))
        assert process.killed and process.waited and client._process is None

    asyncio.run(run())


def test_worker_cancelled_inference_kills_process_and_does_not_route_to_qwen(monkeypatch, tmp_path):
    config, metadata = worker_config(monkeypatch, tmp_path)

    async def run():
        process = Process(metadata, respond=False)

        async def spawn(*args, **options):
            return process

        monkeypatch.setattr(
            "race_engineer.intelligence.driver_action_client.start_owned_process", spawn
        )
        client = LocalDriverActionClassifier(config)
        await client.start()
        task = asyncio.create_task(
            client.predict(ActionRequest(request_id="trial", input_text="Hey"))
        )
        while not process.requests:
            await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert process.killed and process.waited

    asyncio.run(run())


def test_live_metadata_rejects_label_order_and_path_injection(monkeypatch, tmp_path):
    config, metadata = worker_config(monkeypatch, tmp_path)
    for changed in ({"labels": list(reversed(LABELS))}, {"adapter_file": "../other.safetensors"}):
        (config.model_path / "model.json").write_text(
            json.dumps(metadata | changed), encoding="utf-8"
        )
        with pytest.raises(ValueError):
            load_action_metadata(config.model_path)


@pytest.mark.parametrize("social", [True, False])
def test_live_factory_keeps_generated_replies_and_refreshes_factual_evidence(monkeypatch, social):
    class Model:
        def __init__(self):
            self.calls = []

        async def request(self, **kwargs):
            self.calls.append(kwargs)
            if not social and len(self.calls) == 1:
                return {
                    "social_comment": False,
                    "requested_facts": ["fuel level"],
                    "temporal_scope": "current",
                    "capability_ids": [],
                    "queries": [{"signal_id": "player.fuel_l", "operation": "latest"}],
                    "missing_information": "none",
                }
            return {
                "goal": "acknowledge" if social else "inform",
                "speech": "I'm here. Go ahead." if social else "Fuel is {{a}} liters.",
            }

    async def run():
        model = Model()
        classifier = Classifier(
            prediction(
                **{
                    "radio_check" if social else "request_information": 0.99,
                }
            )
        )
        monkeypatch.setattr("race_engineer.intelligence.factory.LocalJsonModel", lambda _: model)
        monkeypatch.setattr(
            "race_engineer.intelligence.factory.LocalDriverActionClassifier", lambda _: classifier
        )
        memory = BoundedTelemetryMemory()

        def frame(sequence, fuel):
            return RaceContext(
                frame=TelemetryFrame(
                    source="test",
                    session_id="trial-session",
                    sequence=sequence,
                    observed_at=datetime.now(UTC),
                    session_time_s=float(sequence),
                    player=PlayerState(driver_id="player", fuel_l=fuel),
                    capabilities=("fuel",),
                )
            )

        memory.update(frame(1, 30.0))
        config = ConversationConfig(driver_interpreter=DriverInterpreterConfig(enabled=True))
        engineer = live_intelligence(config, memory)
        try:
            await engineer.start()
            prepared = await engineer.prepare(turn("Are you there?" if social else "Fuel?"))
            assert len(model.calls) == (1 if social else 2)
            memory.update(frame(2, 29.0))
            reply = await engineer.ground(prepared)
            assert reply.source_sequence == 2
            assert reply.text == ("I'm here. Go ahead." if social else "Fuel is 29 liters.")
            core_payload = json.loads(model.calls[-1]["content"])
            if social:
                assert not prepared.context.evidence and not reply.evidence_ids
                assert "driver_radio_check" in core_payload["situation"]
            else:
                assert prepared.context.evidence and reply.evidence_ids
        finally:
            await engineer.aclose()
        assert classifier.starts == classifier.closes == 1

    asyncio.run(run())


def test_social_fast_path_does_not_authorize_invented_telemetry(monkeypatch):
    class Model:
        async def request(self, **kwargs):
            return {"goal": "inform", "speech": "You have 30 liters of fuel."}

    async def run():
        classifier = Classifier()
        monkeypatch.setattr("race_engineer.intelligence.factory.LocalJsonModel", lambda _: Model())
        monkeypatch.setattr(
            "race_engineer.intelligence.factory.LocalDriverActionClassifier", lambda _: classifier
        )
        memory = BoundedTelemetryMemory()
        memory.update(
            RaceContext(
                frame=TelemetryFrame(
                    source="test",
                    session_id="trial-session",
                    sequence=1,
                    observed_at=datetime.now(UTC),
                    session_time_s=1.0,
                    player=PlayerState(driver_id="player"),
                )
            )
        )
        engineer = live_intelligence(
            ConversationConfig(driver_interpreter=DriverInterpreterConfig(enabled=True)),
            memory,
        )
        try:
            with pytest.raises(LocalIntelligenceError):
                await engineer.respond(turn())
        finally:
            await engineer.aclose()

    asyncio.run(run())
