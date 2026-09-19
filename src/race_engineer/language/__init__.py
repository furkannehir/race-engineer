"""Fact-bound language generators and deterministic fallback behavior."""

from race_engineer.config import LanguageConfig
from race_engineer.core.interfaces import LanguageGenerator
from race_engineer.language.deterministic import (
    DeterministicLanguageGenerator,
    UnsupportedIntentError,
)
from race_engineer.language.fallback import FallbackLanguageGenerator


def language_factory(config: LanguageConfig) -> LanguageGenerator:
    if config.adapter == "deterministic":
        return DeterministicLanguageGenerator()
    raise AssertionError(f"unsupported language adapter: {config.adapter}")


__all__ = [
    "DeterministicLanguageGenerator",
    "FallbackLanguageGenerator",
    "UnsupportedIntentError",
    "language_factory",
]
