"""Composition for the evidence-driven local intelligence path."""

from race_engineer.config import ConversationConfig
from race_engineer.core.interfaces import ContextQueryPlanner
from race_engineer.intelligence.capabilities import DeterministicRaceCapabilities
from race_engineer.intelligence.context_engine import QueryDrivenContextEngineer
from race_engineer.intelligence.context_planner import QwenContextQueryPlanner
from race_engineer.intelligence.driver_action_client import LocalDriverActionClassifier
from race_engineer.intelligence.driver_action_planner import ActionAssistedContextPlanner
from race_engineer.intelligence.grounding import StrictEvidenceGrounder
from race_engineer.intelligence.local_model import LocalJsonModel
from race_engineer.intelligence.orchestrator import EngineerOrchestrator
from race_engineer.intelligence.portable_engineer import PortableQwenEngineer
from race_engineer.intelligence.telemetry_memory import BoundedTelemetryMemory


def live_intelligence(
    config: ConversationConfig,
    memory: BoundedTelemetryMemory,
) -> EngineerOrchestrator:
    """Build grounded Qwen intelligence, optionally with the experimental social fast path."""

    model = LocalJsonModel(config)
    planner: ContextQueryPlanner = QwenContextQueryPlanner(config, model=model)
    runtime = None
    if config.driver_interpreter.enabled:
        runtime = ActionAssistedContextPlanner(
            LocalDriverActionClassifier(config.driver_interpreter), planner,
        )
        planner = runtime
    context = QueryDrivenContextEngineer(
        memory,
        planner,
        DeterministicRaceCapabilities(memory),
    )
    engineer = PortableQwenEngineer(config, model=model)
    return EngineerOrchestrator(
        context, engineer, engineer, StrictEvidenceGrounder(), runtime=runtime,
    )
