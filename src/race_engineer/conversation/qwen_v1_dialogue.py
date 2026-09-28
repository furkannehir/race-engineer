"""The retained Qwen-v1 factual route with CE-05 dialogue fields.

This keeps the same local model/runtime and established query vocabulary. The
additional bounded fields express corrections, repeats and social acts; they do not
allow generated wording, race values, tools, or preference writes.
"""

import json
from copy import deepcopy
from typing import Literal

from pydantic import Field, model_validator

from race_engineer.conversation.facts import DEFAULT_FACT_CATALOG
from race_engineer.conversation.judges import QwenSemanticJudge
from race_engineer.conversation.local_model import SYSTEM_PROMPT
from race_engineer.core.contracts import ContractModel
from race_engineer.core.conversation import FieldRelation, RaceQuery, RadioLanguage
from race_engineer.core.dialogue import DialogueTurnInput, SemanticProposal, SocialAct


class QwenV1DialoguePlan(ContractModel):
    """Bounded private plan for the retained Qwen compatibility adapter."""

    schema_version: Literal["qwen-v1-dialogue-plan.v2"] = "qwen-v1-dialogue-plan.v2"
    language: RadioLanguage
    mode: Literal["request", "correction", "repeat"] = "request"
    queries: tuple[RaceQuery, ...] = Field(default=(), max_length=6)
    field_relation: FieldRelation | None = None
    clarification: Literal["none", "topic", "opponent"] = "none"
    acts: tuple[SocialAct, ...] = Field(default=(), max_length=1)

    @model_validator(mode="after")
    def validate_plan(self) -> "QwenV1DialoguePlan":
        if len(self.queries) != len(set(self.queries)):
            raise ValueError("queries must be unique")
        if (RaceQuery.FIELD_STATUS in self.queries) != (self.field_relation is not None):
            raise ValueError("field status requires exactly one requested relation")
        if self.clarification != "none" and (self.queries or self.acts):
            raise ValueError("clarification cannot discard another understood part")
        if self.mode == "repeat" and (self.queries or self.acts or self.clarification != "none"):
            raise ValueError("repeat has no new meaning")
        if "close" in self.acts and (self.queries or self.mode != "request"):
            raise ValueError("close must be the only meaning")
        if not (self.queries or self.acts or self.clarification != "none" or self.mode == "repeat"):
            raise ValueError("plan needs a meaning")
        return self


def _schema() -> dict[str, object]:
    schema = deepcopy(QwenV1DialoguePlan.model_json_schema())
    # Defaults are application conveniences; constrained decoding must emit every field.
    schema["required"] = list(schema["properties"])
    return schema


_DIALOGUE_RULES = """
The output also has mode and acts. Preserve the existing factual-query rules above.
Every output also has field_relation. It MUST be null unless queries contains field_status.
For field_status it identifies exactly what the driver asked: 'first' for leading/P1,
'last' for last place, 'cars_ahead' or 'cars_behind' for a count on that side, and
'position_of_total' for place out of the complete field. Do not infer this relation from
telemetry. It comes only from the driver's wording. A yes/no answer is calculated later.
mode='correction' only when the driver corrects an earlier meaning ('no, I meant...');
otherwise mode='request'. mode='repeat' only for asking to hear the last reply again and
then queries=[], clarification='none', acts=[]. The application refreshes repeated facts.

acts=[] normally. acts=['acknowledge'] when the driver reports frustration, unfair/dirty
driving, a mistake, or a difficult moment and a calm teammate should briefly acknowledge
it. It may accompany factual queries. This act never verifies contact, blame, a penalty,
or the driver's emotion. acts=['close'] only for a standalone thanks/end of exchange;
then queries=[] and clarification='none'. Do not turn ordinary factual/unsupported requests
into acknowledgments. Retain every factual or unsupported part of a mixed request.

Examples:
User: That was dirty. Gap behind?
{"schema_version":"qwen-v1-dialogue-plan.v2","language":"en","mode":"request","queries":["gap_behind"],"field_relation":null,"clarification":"none","acts":["acknowledge"]}
User: That was dirty. Where are we?
{"schema_version":"qwen-v1-dialogue-plan.v2","language":"en","mode":"request","queries":["position"],"field_relation":null,"clarification":"none","acts":["acknowledge"]}
User: Are we first?
{"schema_version":"qwen-v1-dialogue-plan.v2","language":"en","mode":"request","queries":["field_status"],"field_relation":"first","clarification":"none","acts":[]}
User: Lider miyiz?
{"schema_version":"qwen-v1-dialogue-plan.v2","language":"tr","mode":"request","queries":["field_status"],"field_relation":"first","clarification":"none","acts":[]}
User: Sonuncu muyuz?
{"schema_version":"qwen-v1-dialogue-plan.v2","language":"tr","mode":"request","queries":["field_status"],"field_relation":"last","clarification":"none","acts":[]}
User: Bu yaptığı hiç hoş değildi.
{"schema_version":"qwen-v1-dialogue-plan.v2","language":"tr","mode":"request","queries":[],"field_relation":null,"clarification":"none","acts":["acknowledge"]}
User: Thanks.
{"schema_version":"qwen-v1-dialogue-plan.v2","language":"en","mode":"request","queries":[],"field_relation":null,"clarification":"none","acts":["close"]}
User: Say that again.
{"schema_version":"qwen-v1-dialogue-plan.v2","language":"en","mode":"repeat","queries":[],"field_relation":null,"clarification":"none","acts":[]}
"""


_LEGACY_FIELD_RULE = (
    "field_status: whether the driver is currently last in the complete overall "
    "classification,\n"
    "their place out of the classified field, or how many classified cars are behind. Use "
    "this\n"
    "for natural variants such as 'Am I last?', 'Anyone behind us?', or 'Sonuncu muyuz?'. "
    "It\n"
    "already includes the current position, so do not add position for the same request."
)
_RELATIONAL_FIELD_RULE = """field_status: a relationship to the complete overall classification.
Use it for first/leading, last, car counts ahead/behind, and place out of the field. The
separate field_relation says which relationship the driver actually requested. It already
includes current position, so do not add position for the same request."""

HYBRID_PROMPT = (
    SYSTEM_PROMPT.replace(_LEGACY_FIELD_RULE, _RELATIONAL_FIELD_RULE).replace(
        "Return ONLY a JSON ConversationPlan matching the provided schema.",
        "Return ONLY JSON matching qwen-v1-dialogue-plan.v2 and its provided schema.",
    )
    + _DIALOGUE_RULES
)


class QwenV1DialogueJudge(QwenSemanticJudge):
    """One local constrained Qwen call; no extra model stage on ordinary questions."""

    async def judge(self, request: DialogueTurnInput) -> SemanticProposal:
        context = self._context(request)
        plan = QwenV1DialoguePlan.model_validate(
            await self._request_json(
                prompt=self._instructions(request),
                content=json.dumps(context, ensure_ascii=False),
                schema=_schema(),
                max_tokens=384,
            )
        )
        plan_queries = plan.queries
        if RaceQuery.FIELD_STATUS in plan_queries:
            plan_queries = tuple(item for item in plan_queries if item is not RaceQuery.POSITION)
        requests = tuple(
            DEFAULT_FACT_CATALOG.part_for_query(
                item,
                part_id=f"q{index}",
                field_relation=(plan.field_relation if item is RaceQuery.FIELD_STATUS else None),
            )
            for index, item in enumerate(plan_queries, 1)
        )
        return SemanticProposal(
            language=request.reply_language or plan.language,
            mode=plan.mode,
            requests=requests,
            acts=plan.acts,
            clarification=plan.clarification,
            model_id=f"{self.config.model}:v1-hybrid",
        )

    @staticmethod
    def _context(request: DialogueTurnInput) -> dict[str, object]:
        focus = request.dialogue.topic
        return {
            "current_utterance": request.question,
            "default_language": request.default_language,
            "active_topic": [
                {
                    "query": item.part.query,
                    "reference": item.part.reference,
                    "field_relation": item.part.field_relation,
                }
                for item in focus.requests
            ]
            if focus
            else [],
            "pending_clarification": request.dialogue.pending.model_dump(mode="json")
            if request.dialogue.pending
            else None,
            "last_response": {
                "requests": [
                    {
                        "query": item.part.query,
                        "reference": item.part.reference,
                        "field_relation": item.part.field_relation,
                    }
                    for item in request.dialogue.responses[-1].requests
                ],
                "delivery": request.dialogue.responses[-1].delivery,
            }
            if request.dialogue.responses
            else None,
        }

    @staticmethod
    def _instructions(request: DialogueTurnInput) -> str:
        instructions = HYBRID_PROMPT
        if request.reply_language is not None:
            instructions += f"\nFor this turn, language MUST be '{request.reply_language}'."
        return instructions
