import uuid
from typing import Any

import pytest

from neuralstate.consolidation import prompts
from neuralstate.consolidation.reasoner import HeuristicReasoner, LLMReasoner
from neuralstate.consolidation.types import EpisodeView, FactView, RuleDraft


def ep(outcome="failure", approach="npm install", task="deploy", detail="lockfile stale", **content):
    return EpisodeView(id=uuid.uuid4(), task_type=task, summary=f"{task} something", outcome=outcome,
                       outcome_detail=detail, content={"approach": approach, **content})


def fact(s, p, o):
    return FactView(id=uuid.uuid4(), subject=s, predicate=p, object=o)


# ------------------------------------------------------------- heuristic
async def test_heuristic_extracts_structured_hints_only():
    r = HeuristicReasoner()
    [x, y] = await r.extract([
        ep(facts=[{"subject": "acme", "predicate": "uses", "object": "pg"}, {"bad": 1}], entities=["acme"]),
        ep(),
    ])
    assert [(f.subject, f.object) for f in x.facts] == [("acme", "pg")]
    assert [e.name for e in x.entities] == ["acme"]
    assert y.facts == [] and y.episode_index == 1


async def test_heuristic_judges_pairs():
    r = HeuristicReasoner()
    verdicts = await r.judge_pairs([
        (fact("acme", "uses", "pg"), fact("ACME", "uses ", "PG")),
        (fact("acme", "primary db", "pg"), fact("acme", "primary db", "mysql")),
        (fact("acme", "uses", "pg"), fact("acme", "located in", "berlin")),
    ])
    assert [v.verdict for v in verdicts] == ["duplicate", "contradiction", "consistent"]


async def test_heuristic_rule_uses_dominant_failure_and_success_approach():
    r = HeuristicReasoner()
    fails = [ep() for _ in range(5)]
    oks = [ep("success", approach="npm ci")]
    draft = await r.synthesize_rule(fails, oks)
    assert draft.avoid == "npm install" and draft.action == "npm ci"
    check = await r.validate_rule(draft, fails, oks)
    assert check.explains_failures and not check.contradicts_success


async def test_heuristic_rejects_rule_that_breaks_a_success():
    r = HeuristicReasoner()
    fails = [ep() for _ in range(5)]
    draft = await r.synthesize_rule(fails, [])
    check = await r.validate_rule(draft, fails, [ep("success", approach="npm install")])
    assert check.contradicts_success


async def test_heuristic_finds_no_rule_without_a_common_cause():
    r = HeuristicReasoner()
    assert await r.synthesize_rule([ep(approach=f"a{i}") for i in range(5)], []) is None


# ------------------------------------------------------------- LLM reasoner, with a scripted model
class ScriptedLLM:
    name = "scripted"

    def __init__(self, *responses: dict[str, Any]):
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    async def complete_json(self, *, system, prompt, schema):
        self.calls.append({"system": system, "prompt": prompt, "schema": schema})
        return self.responses.pop(0)


async def test_llm_extract_drops_out_of_range_indexes():
    llm = ScriptedLLM({"extractions": [
        {"episode_index": 0, "entities": [], "facts": [
            {"subject": "acme", "predicate": "uses", "object": "pg", "object_is_entity": True, "confidence": 0.9}]},
        {"episode_index": 7, "entities": [], "facts": []},
    ]})
    out = await LLMReasoner(llm).extract([ep()])
    assert len(out) == 1 and out[0].facts[0].object_is_entity
    assert llm.calls[0]["schema"] is prompts.EXTRACT_SCHEMA
    assert '<episode index="0">' in llm.calls[0]["prompt"]


async def test_llm_synthesize_respects_has_rule_false():
    llm = ScriptedLLM({"has_rule": False, "condition": "", "avoid": "", "action": "", "rationale": ""})
    assert await LLMReasoner(llm).synthesize_rule([ep()] * 5, []) is None


async def test_llm_synthesize_and_validate():
    llm = ScriptedLLM(
        {"has_rule": True, "condition": "deploying node apps", "avoid": "npm install", "action": "npm ci",
         "rationale": "stale lockfile"},
        {"explains_failures": True, "contradicts_success": False, "notes": ""},
    )
    r = LLMReasoner(llm)
    draft = await r.synthesize_rule([ep()] * 5, [ep("success")])
    assert isinstance(draft, RuleDraft) and draft.action == "npm ci"
    assert (await r.validate_rule(draft, [ep()], [])).explains_failures
    assert "<rule>" in llm.calls[1]["prompt"]


def test_long_episode_content_is_shortened_visibly():
    text = prompts.render_episode(0, ep(blob="x" * 10_000))
    assert "content shortened" in text and len(text) < 5000


def _strict(schema):
    if schema.get("type") == "object":
        assert schema["additionalProperties"] is False
        assert set(schema["required"]) == set(schema["properties"])
        for sub in schema["properties"].values():
            _strict(sub)
    elif schema.get("type") == "array":
        _strict(schema["items"])


@pytest.mark.parametrize("schema", [prompts.EXTRACT_SCHEMA, prompts.JUDGE_SCHEMA, prompts.SYNTH_SCHEMA,
                                    prompts.VALIDATE_SCHEMA])
def test_schemas_are_strict_mode_compatible(schema):
    _strict(schema)
