"""Interactive/benchmark adapter over the packaged Ministral implementation."""

import time
from datetime import UTC, datetime

from race_engineer.core.intelligence import DriverTurn
from race_engineer.intelligence.ministral_dialogue import MinistralDialogue as MinistralDialogue
from race_engineer.radio_diagnostics import error_code


class MinistralConversation:
    """Demo adapter: only accepted text is delivered, rejected drafts stay in traces."""

    def __init__(self, config, memory, *, model=None):
        self.engine = MinistralDialogue(config, memory, model=model)
        self.memory = memory
        self.generation = 0

    @property
    def history(self):
        return self.engine.history

    def clear(self):
        self.engine.clear()

    async def respond(self, question, language, turn_id):
        frame = self.memory.latest_context().frame
        turn = DriverTurn(
            turn_id=turn_id,
            transcript=question,
            session_id=frame.session_id,
            received_at=datetime.now(UTC),
            generation=self.generation,
            asr_language=language,
            reply_language=language,
        )
        row = {
            "question": question,
            "language": language,
            "turn_id": turn_id,
            "reply": None,
            "raw_reply": None,
        }
        started = time.perf_counter()
        try:
            answer = await self.engine.respond(turn)
            row.update(
                reply=answer.text,
                source_sequence=answer.source_sequence,
                evidence_ids=list(answer.evidence_ids),
            )
            row["application_validation"] = {"status": "accepted"}
        except Exception as error:
            row["failure"] = error_code(error)
            row["application_validation"] = {"status": "rejected", "reason": error_code(error)}
        row["trace"] = self.engine.last_trace
        drafts = [item for item in row["trace"] if "reply" in item]
        row["requests"] = [item for item in drafts if not item.get("direct", False)]
        row["model_calls"] = len(row["requests"])
        row["raw_reply"] = next(
            (item["reply"]["content"] for item in reversed(drafts) if item["reply"].get("content")),
            None,
        )
        row["duration_ms"] = round((time.perf_counter() - started) * 1000, 3)
        return row
