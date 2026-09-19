"""Timeout and validation boundary for replaceable language generators."""

import asyncio
import logging

from race_engineer.core.contracts import SpeechIntent, Utterance
from race_engineer.core.interfaces import LanguageGenerator


class FallbackLanguageGenerator:
    """Uses deterministic wording when a primary generator fails or violates its contract."""

    def __init__(
        self,
        primary: LanguageGenerator,
        fallback: LanguageGenerator,
        timeout_s: float,
    ) -> None:
        if timeout_s <= 0:
            raise ValueError("language generation timeout must be positive")
        self._primary = primary
        self._fallback = fallback
        self._timeout_s = timeout_s
        self._logger = logging.getLogger(__name__)

    @staticmethod
    def _validate(intent: SpeechIntent, utterance: Utterance) -> None:
        if utterance.intent_id != intent.intent_id:
            raise ValueError("generator returned an utterance for a different intent")
        if len(utterance.text.split()) > intent.max_words:
            raise ValueError("generator exceeded the intent word limit")

    async def generate(self, intent: SpeechIntent) -> Utterance:
        try:
            utterance = await asyncio.wait_for(
                self._primary.generate(intent),
                timeout=self._timeout_s,
            )
            self._validate(intent, utterance)
            return utterance
        except Exception as error:
            self._logger.warning(
                "language generator fallback activated",
                extra={
                    "event": "language_fallback",
                    "intent_id": intent.intent_id,
                    "reason": type(error).__name__,
                },
            )
        fallback = await self._fallback.generate(intent)
        self._validate(intent, fallback)
        return fallback
