"""Composition for the evidence-driven local intelligence path."""

from race_engineer.config import ConversationConfig
from race_engineer.intelligence.capabilities import DeterministicRaceCapabilities
from race_engineer.intelligence.context_engine import QueryDrivenContextEngineer
from race_engineer.intelligence.context_planner import LFM_NATIVE_ID, QwenContextQueryPlanner
from race_engineer.intelligence.grounding import StrictEvidenceGrounder
from race_engineer.intelligence.local_model import LocalJsonModel
from race_engineer.intelligence.native_lfm import LfmNativePlanner
from race_engineer.intelligence.orchestrator import EngineerOrchestrator
from race_engineer.intelligence.portable_engineer import PortableQwenEngineer
from race_engineer.intelligence.telemetry_memory import BoundedTelemetryMemory


def live_intelligence(
    config: ConversationConfig,
    memory: BoundedTelemetryMemory,
) -> EngineerOrchestrator:
    """Build the selected local Context/Core composition and shared grounding."""

    # Native functions select facts; the reply still needs decoder-enforced
    # placeholder constraints. Both clients use this same configured model/server.
    model = LocalJsonModel(config)
    context = QueryDrivenContextEngineer(
        memory,
        LfmNativePlanner(config)
        if config.context_planner == LFM_NATIVE_ID
        else QwenContextQueryPlanner(config, model=model, planner_id=config.context_planner),
        DeterministicRaceCapabilities(
            memory, include_rank_checks=config.context_planner == LFM_NATIVE_ID
        ),
    )
    engineer = PortableQwenEngineer(config, model=model)
    return EngineerOrchestrator(context, engineer, engineer, StrictEvidenceGrounder())
