"""Bounded optional fallback. This is not connected to the live panel in CE-04."""

import asyncio
from datetime import UTC, datetime

from race_engineer.core.dialogue import DialogueTurnInput, SemanticJudgeError, SemanticProposal
from race_engineer.core.interfaces import SemanticJudge


class JudgeRouter:
    def __init__(self, primary: SemanticJudge, fallback: SemanticJudge | None = None) -> None:
        if primary is fallback:
            raise ValueError("fallback must be a distinct judge")
        self.primary, self.fallback = primary, fallback
        self.attempts = 0
        self.fallbacks = 0

    async def judge(self, request: DialogueTurnInput) -> SemanticProposal:
        last: SemanticProposal | None = None
        for index, judge in enumerate((self.primary, self.fallback)):
            if judge is None:
                break
            remaining = (request.deadline - datetime.now(UTC)).total_seconds()
            if remaining <= 0:
                raise SemanticJudgeError("judge_deadline")
            self.attempts += 1
            self.fallbacks += int(index == 1)
            try:
                async with asyncio.timeout(remaining):
                    result = await judge.judge(request)
                    last = SemanticProposal.model_validate(result.model_dump())
                # Clarification/missing capabilities aren't model failures. Only an
                # explicit semantic abstention or invalid result is fallback eligible.
                if not last.abstain:
                    return last
            except (SemanticJudgeError, ValueError):
                continue
        if last is not None:
            return last
        raise SemanticJudgeError("judge_routes_failed")
