"""Composition of replaceable local conversation planners."""

from race_engineer.config import ConversationConfig
from race_engineer.conversation.local_model import LocalLlamaCppPlanner
from race_engineer.conversation.qwen_v1_dialogue import QwenV1DialogueJudge
from race_engineer.core.interfaces import ConversationPlanner, SemanticJudge


def conversation_planner(config: ConversationConfig) -> ConversationPlanner:
    if config.adapter == "llama-cpp":
        return LocalLlamaCppPlanner(config)
    raise ValueError("unsupported_conversation_adapter")


def semantic_judge(config: ConversationConfig) -> SemanticJudge:
    if config.dialogue.adapter == "qwen-v1-hybrid":
        return QwenV1DialogueJudge(config)
    raise ValueError("unsupported_semantic_judge")
