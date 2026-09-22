"""Composition of replaceable local conversation planners."""

from race_engineer.config import ConversationConfig
from race_engineer.conversation.local_model import LocalLlamaCppPlanner
from race_engineer.core.interfaces import ConversationPlanner


def conversation_planner(config: ConversationConfig) -> ConversationPlanner:
    if config.adapter == "llama-cpp":
        return LocalLlamaCppPlanner(config)
    raise ValueError("unsupported_conversation_adapter")
