from __future__ import annotations

import uuid
from typing import Any, Literal

from pydantic import BaseModel, Field


class EpisodeView(BaseModel):
    id: uuid.UUID
    task_type: str | None = None
    summary: str
    outcome: str
    outcome_detail: str | None = None
    content: dict[str, Any] = Field(default_factory=dict)


class ExtractedEntity(BaseModel):
    name: str
    type: str = "thing"


class ExtractedFact(BaseModel):
    subject: str
    predicate: str
    object: str
    object_is_entity: bool = False
    confidence: float = 0.7


class Extraction(BaseModel):
    episode_index: int
    entities: list[ExtractedEntity] = Field(default_factory=list)
    facts: list[ExtractedFact] = Field(default_factory=list)


class FactView(BaseModel):
    id: uuid.UUID
    subject: str
    predicate: str
    object: str


Verdict = Literal["consistent", "duplicate", "contradiction"]


class PairVerdict(BaseModel):
    pair_index: int
    verdict: Verdict
    reason: str = ""


class RuleDraft(BaseModel):
    condition: str
    avoid: str
    action: str
    rationale: str

    def text(self) -> str:
        return f"When {self.condition}: avoid {self.avoid}; instead {self.action}."


class RuleValidation(BaseModel):
    explains_failures: bool
    contradicts_success: bool
    notes: str = ""
