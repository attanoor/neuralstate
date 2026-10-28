"""The judgement calls of consolidation, behind one interface.

`LLMReasoner` asks any `LLMProvider`. `HeuristicReasoner` uses deterministic rules and needs
no model, so tests and the offline demo can run without API keys.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Protocol

from neuralstate.consolidation import prompts
from neuralstate.consolidation.types import (
    EpisodeView,
    Extraction,
    ExtractedEntity,
    ExtractedFact,
    FactView,
    PairVerdict,
    RuleDraft,
    RuleValidation,
)
from neuralstate.providers.base import LLMProvider


class Reasoner(Protocol):
    name: str

    async def extract(self, episodes: list[EpisodeView]) -> list[Extraction]: ...

    async def judge_pairs(self, pairs: list[tuple[FactView, FactView]]) -> list[PairVerdict]: ...

    async def synthesize_rule(self, failures: list[EpisodeView], successes: list[EpisodeView]) -> RuleDraft | None: ...

    async def validate_rule(self, rule: RuleDraft, failures: list[EpisodeView],
                            successes: list[EpisodeView]) -> RuleValidation: ...


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


class LLMReasoner:
    def __init__(self, llm: LLMProvider):
        self.llm = llm
        self.name = f"llm:{llm.name}"

    async def extract(self, episodes: list[EpisodeView]) -> list[Extraction]:
        if not episodes:
            return []
        data = await self.llm.complete_json(
            system=prompts.EXTRACT_SYSTEM, prompt=prompts.extract_prompt(episodes), schema=prompts.EXTRACT_SCHEMA)
        out = [Extraction.model_validate(x) for x in data.get("extractions", [])]
        return [x for x in out if 0 <= x.episode_index < len(episodes)]

    async def judge_pairs(self, pairs: list[tuple[FactView, FactView]]) -> list[PairVerdict]:
        if not pairs:
            return []
        data = await self.llm.complete_json(
            system=prompts.JUDGE_SYSTEM, prompt=prompts.judge_prompt(pairs), schema=prompts.JUDGE_SCHEMA)
        out = [PairVerdict.model_validate(x) for x in data.get("verdicts", [])]
        return [v for v in out if 0 <= v.pair_index < len(pairs)]

    async def synthesize_rule(self, failures: list[EpisodeView], successes: list[EpisodeView]) -> RuleDraft | None:
        data = await self.llm.complete_json(
            system=prompts.SYNTH_SYSTEM, prompt=prompts.synth_prompt(failures, successes),
            schema=prompts.SYNTH_SCHEMA)
        if not data.get("has_rule") or not data.get("avoid") or not data.get("action"):
            return None
        return RuleDraft.model_validate(data)

    async def validate_rule(self, rule: RuleDraft, failures: list[EpisodeView],
                            successes: list[EpisodeView]) -> RuleValidation:
        data = await self.llm.complete_json(
            system=prompts.VALIDATE_SYSTEM, prompt=prompts.validate_prompt(rule, failures, successes),
            schema=prompts.VALIDATE_SCHEMA)
        return RuleValidation.model_validate(data)


class HeuristicReasoner:
    """Model-free reasoner.

    Extraction reads structured hints the agent can put in an episode's `content`:
    ``{"facts": [{"subject", "predicate", "object"}], "entities": [...], "approach": "..."}``.
    Rule synthesis uses the most common failing `approach` and the most common succeeding one.
    """

    name = "heuristic"

    async def extract(self, episodes: list[EpisodeView]) -> list[Extraction]:
        out = []
        for i, ep in enumerate(episodes):
            facts, entities = [], []
            for f in ep.content.get("facts", []) or []:
                if isinstance(f, dict) and {"subject", "predicate", "object"} <= f.keys():
                    facts.append(ExtractedFact(
                        subject=str(f["subject"]), predicate=str(f["predicate"]), object=str(f["object"]),
                        object_is_entity=bool(f.get("object_is_entity", False)),
                        confidence=float(f.get("confidence", 0.7))))
            for e in ep.content.get("entities", []) or []:
                if isinstance(e, str):
                    entities.append(ExtractedEntity(name=e))
                elif isinstance(e, dict) and "name" in e:
                    entities.append(ExtractedEntity(name=str(e["name"]), type=str(e.get("type", "thing"))))
            out.append(Extraction(episode_index=i, entities=entities, facts=facts))
        return out

    async def judge_pairs(self, pairs: list[tuple[FactView, FactView]]) -> list[PairVerdict]:
        verdicts = []
        for i, (a, b) in enumerate(pairs):
            same_key = norm(a.subject) == norm(b.subject) and norm(a.predicate) == norm(b.predicate)
            if same_key and norm(a.object) == norm(b.object):
                verdicts.append(PairVerdict(pair_index=i, verdict="duplicate", reason="identical triple"))
            elif same_key:
                verdicts.append(PairVerdict(pair_index=i, verdict="contradiction",
                                            reason="same subject and predicate, different object"))
            else:
                verdicts.append(PairVerdict(pair_index=i, verdict="consistent", reason="different claims"))
        return verdicts

    @staticmethod
    def _approach(ep: EpisodeView) -> str | None:
        value = ep.content.get("approach")
        return str(value) if value else None

    async def synthesize_rule(self, failures: list[EpisodeView], successes: list[EpisodeView]) -> RuleDraft | None:
        failing = Counter(a for a in map(self._approach, failures) if a)
        if not failing:
            return None
        avoid, count = failing.most_common(1)[0]
        if count < max(2, len(failures) // 2):
            return None  # no dominant failing approach: nothing actionable to learn
        working = Counter(a for a in map(self._approach, successes) if a and a != avoid)
        action = working.most_common(1)[0][0] if working else "try a different approach and verify it before committing"
        task = failures[0].task_type or "this kind of task"
        reasons = Counter(ep.outcome_detail for ep in failures if ep.outcome_detail)
        why = reasons.most_common(1)[0][0] if reasons else "it repeatedly failed"
        return RuleDraft(
            condition=f"working on {task}",
            avoid=avoid,
            action=action,
            rationale=f"Failed in {count} of {len(failures)} similar episodes ({why}).",
        )

    async def validate_rule(self, rule: RuleDraft, failures: list[EpisodeView],
                            successes: list[EpisodeView]) -> RuleValidation:
        broke = [ep for ep in successes if self._approach(ep) == rule.avoid]
        return RuleValidation(
            explains_failures=any(self._approach(ep) == rule.avoid for ep in failures),
            contradicts_success=bool(broke),
            notes=f"{len(broke)} successful episodes used the avoided approach" if broke else "",
        )
