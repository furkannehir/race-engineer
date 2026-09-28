"""Scripted semantic meanings for orchestration tests, not a language-understanding model."""

from collections import deque
from collections.abc import Awaitable, Callable

from race_engineer.core.dialogue import DialogueTurnInput, SemanticProposal

type JudgeStep = (
    SemanticProposal | Exception | Callable[[DialogueTurnInput], Awaitable[SemanticProposal]]
)


class ScriptedSemanticJudge:
    def __init__(self, *steps: JudgeStep) -> None:
        self._steps = deque(steps)
        self.requests: list[DialogueTurnInput] = []

    async def judge(self, request: DialogueTurnInput) -> SemanticProposal:
        self.requests.append(request)
        if not self._steps:
            raise AssertionError("scripted judge has no remaining steps")
        step = self._steps.popleft()
        if isinstance(step, Exception):
            raise step
        if isinstance(step, SemanticProposal):
            return step
        return await step(request)
