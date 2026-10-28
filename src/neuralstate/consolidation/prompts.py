"""Prompts and JSON schemas for every LLM call the consolidation engine makes.

Schemas follow the strict-mode rules shared by Anthropic structured outputs and OpenAI
`json_schema` (every property required, `additionalProperties: false`), so one schema
works for every provider.
"""

from __future__ import annotations

import json
from typing import Any

from neuralstate.consolidation.types import EpisodeView, FactView, RuleDraft

MAX_EPISODE_CHARS = 4000


def _obj(properties: dict[str, Any]) -> dict[str, Any]:
    return {"type": "object", "properties": properties, "required": list(properties), "additionalProperties": False}


def _arr(items: dict[str, Any]) -> dict[str, Any]:
    return {"type": "array", "items": items}


_STR = {"type": "string"}
_NUM = {"type": "number"}
_INT = {"type": "integer"}
_BOOL = {"type": "boolean"}


def render_episode(i: int, ep: EpisodeView) -> str:
    content = json.dumps(ep.content, ensure_ascii=False, default=str)
    if len(content) > MAX_EPISODE_CHARS:
        content = content[:MAX_EPISODE_CHARS] + " …[content shortened for this prompt]"
    lines = [f"<episode index=\"{i}\">", f"task_type: {ep.task_type or 'unspecified'}",
             f"summary: {ep.summary}", f"outcome: {ep.outcome}"]
    if ep.outcome_detail:
        lines.append(f"outcome_detail: {ep.outcome_detail}")
    lines += [f"content: {content}", "</episode>"]
    return "\n".join(lines)


# ---------------------------------------------------------------- extraction
EXTRACT_SYSTEM = """You maintain the long-term memory of an autonomous software agent.
You read the agent's interaction logs ("episodes") and extract durable knowledge worth
remembering across sessions: stable facts about the world, the clients, the systems the
agent works with, and what each approach led to.

Guidelines:
- Extract facts as (subject, predicate, object) triples. Use short, canonical names for
  subjects and objects (e.g. "acme-corp", "deploy pipeline"), and lowercase verb phrases
  for predicates (e.g. "uses", "requires", "failed because").
- Prefer facts that will still be true next week. Skip pleasantries, one-off chatter and
  anything already implied by another extracted fact.
- Set object_is_entity to true when the object is itself a named thing that other facts
  could refer to, and false when it is a value or description.
- confidence is your belief, from 0 to 1, that the fact is true and durable, given only
  what the episode shows.
- Return one extraction per episode, keyed by its index, even when it yields nothing."""

EXTRACT_SCHEMA = _obj({
    "extractions": _arr(_obj({
        "episode_index": _INT,
        "entities": _arr(_obj({"name": _STR, "type": _STR})),
        "facts": _arr(_obj({
            "subject": _STR, "predicate": _STR, "object": _STR,
            "object_is_entity": _BOOL, "confidence": _NUM,
        })),
    })),
})


def extract_prompt(episodes: list[EpisodeView]) -> str:
    body = "\n\n".join(render_episode(i, ep) for i, ep in enumerate(episodes))
    return f"Extract durable knowledge from these {len(episodes)} episodes.\n\n{body}"


# ---------------------------------------------------------------- reconciliation
JUDGE_SYSTEM = """You check an agent's long-term memory for consistency. For each pair of
facts, decide whether they are:
- "duplicate": they state the same thing in different words;
- "contradiction": they cannot both be true at the same time (for example a client that
  "uses postgres" versus "uses mysql" as its only database);
- "consistent": both can be true together (for example "uses postgres" and "uses redis").
Many predicates allow several values at once; only answer "contradiction" when the facts
are genuinely incompatible. Give a one-sentence reason."""

JUDGE_SCHEMA = _obj({
    "verdicts": _arr(_obj({
        "pair_index": _INT,
        "verdict": {"type": "string", "enum": ["consistent", "duplicate", "contradiction"]},
        "reason": _STR,
    })),
})


def judge_prompt(pairs: list[tuple[FactView, FactView]]) -> str:
    rows = [
        f"<pair index=\"{i}\">\nA: {a.subject} | {a.predicate} | {a.object}\n"
        f"B: {b.subject} | {b.predicate} | {b.object}\n</pair>"
        for i, (a, b) in enumerate(pairs)
    ]
    return "Judge each pair.\n\n" + "\n\n".join(rows)


# ---------------------------------------------------------------- heuristic synthesis
SYNTH_SYSTEM = """You help an autonomous agent learn from its own mistakes. You are given a
cluster of episodes where the agent failed in a similar way, plus some episodes of similar
work that succeeded. Write one heuristic rule that would have prevented these failures.

The rule has four parts:
- condition: the situation in which the rule applies, specific enough that it does not
  fire on unrelated work, general enough to cover every failure in the cluster;
- avoid: the approach that kept failing;
- action: what to do instead, grounded in the successful episodes when there are any;
- rationale: why the avoided approach fails, citing the pattern across the failures.

If the failures do not share a real, actionable cause (they only look alike on the
surface), set "has_rule" to false and leave the other fields empty."""

SYNTH_SCHEMA = _obj({
    "has_rule": _BOOL,
    "condition": _STR,
    "avoid": _STR,
    "action": _STR,
    "rationale": _STR,
})


def synth_prompt(failures: list[EpisodeView], successes: list[EpisodeView]) -> str:
    fail = "\n\n".join(render_episode(i, ep) for i, ep in enumerate(failures))
    ok = "\n\n".join(render_episode(i, ep) for i, ep in enumerate(successes)) or "(none recorded)"
    return (f"<failures count=\"{len(failures)}\">\n{fail}\n</failures>\n\n"
            f"<successes count=\"{len(successes)}\">\n{ok}\n</successes>")


VALIDATE_SYSTEM = """You review a proposed heuristic rule before it is written into an
autonomous agent's permanent memory. Check it against the evidence:
- explains_failures: does following the rule plausibly prevent most of the listed failures?
- contradicts_success: would following the rule have prevented or broken any of the
  listed successful episodes?
Be strict: a rule that is too vague to act on does not explain the failures."""

VALIDATE_SCHEMA = _obj({"explains_failures": _BOOL, "contradicts_success": _BOOL, "notes": _STR})


def validate_prompt(rule: RuleDraft, failures: list[EpisodeView], successes: list[EpisodeView]) -> str:
    return (f"<rule>\ncondition: {rule.condition}\navoid: {rule.avoid}\naction: {rule.action}\n"
            f"rationale: {rule.rationale}\n</rule>\n\n" + synth_prompt(failures, successes))
