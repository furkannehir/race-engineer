"""Composition for the evidence-driven local intelligence path."""

from race_engineer.config import ConversationConfig
from race_engineer.intelligence.capabilities import DeterministicRaceCapabilities
from race_engineer.intelligence.context_engine import QueryDrivenContextEngineer
from race_engineer.intelligence.context_planner import QwenContextQueryPlanner
from race_engineer.intelligence.grounding import StrictEvidenceGrounder
from race_engineer.intelligence.local_model import LocalJsonModel
from race_engineer.intelligence.orchestrator import EngineerOrchestrator
from race_engineer.intelligence.portable_engineer import PortableQwenEngineer
from race_engineer.intelligence.telemetry_memory import BoundedTelemetryMemory


def live_intelligence(
    config: ConversationConfig,
    memory: BoundedTelemetryMemory,
) -> EngineerOrchestrator:
    """Build the portable two-inference Context/Core/Qwen composition."""

    model = LocalJsonModel(config)
    context = QueryDrivenContextEngineer(
        memory,
        QwenContextQueryPlanner(config, model=model),
        DeterministicRaceCapabilities(memory),
    )
    engineer = PortableQwenEngineer(config, model=model)
    return EngineerOrchestrator(context, engineer, engineer, StrictEvidenceGrounder())
