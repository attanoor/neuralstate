"""Pure scoring functions: confidence, decay, contradiction strength, rule effectiveness."""

from __future__ import annotations

import math


def recency(days: float, half_life_days: float = 30.0) -> float:
    """1.0 for brand-new evidence, halving every `half_life_days`."""
    return 0.5 ** (max(0.0, days) / half_life_days)


def reinforce(base: float, new_evidence_confidence: float, damping: float = 0.5) -> float:
    """Noisy-OR update: each new piece of evidence closes part of the gap to 1.0."""
    return min(1.0, base + (1.0 - base) * new_evidence_confidence * damping)


def decayed(base: float, idle_days: float, lam: float, grace_days: float) -> float:
    """Time decay of an unused memory. Recomputed from `base` every cycle, so it never compounds."""
    return base * math.exp(-lam * max(0.0, idle_days - grace_days))


def strength(evidence_count: int, confidence: float, days_since_evidence: float) -> float:
    """How strongly memory backs a fact, for settling contradictions. Range [0, 1]."""
    evidence = min(1.0, math.log1p(evidence_count) / math.log1p(10))
    return 0.40 * recency(days_since_evidence) + 0.35 * evidence + 0.25 * confidence


def contradiction_margin(strength_a: float, strength_b: float) -> tuple[str, float]:
    """Return ('a' | 'b', relative margin). Margin 0 means a tie, 1 means one side has no support."""
    winner = "a" if strength_a >= strength_b else "b"
    top = max(strength_a, strength_b)
    margin = abs(strength_a - strength_b) / top if top > 0 else 0.0
    return winner, margin


def evidence_confidence(evidence_count: int, min_evidence: int) -> float:
    """Confidence from repetition alone: ~0.63 at `min_evidence` observations, approaching 1."""
    return 1.0 - math.exp(-evidence_count / max(1, min_evidence))


def rule_effectiveness(success_after: int, failure_after: int) -> float:
    """Laplace-smoothed success rate of episodes where the rule was in context (prior = 0.5)."""
    return (success_after + 1) / (success_after + failure_after + 2)


def rule_confidence(evidence_count: int, success_after: int, failure_after: int, min_evidence: int) -> float:
    """Evidence-based confidence, scaled up when the rule helps and down when it doesn't."""
    eff = rule_effectiveness(success_after, failure_after)
    return max(0.0, min(1.0, evidence_confidence(evidence_count, min_evidence) * 2.0 * eff))


def rule_should_retire(success_after: int, failure_after: int, min_trials: int, threshold: float) -> bool:
    """A rule that has been tried enough and still mostly fails stops being recalled."""
    return (success_after + failure_after) >= min_trials and rule_effectiveness(success_after, failure_after) < threshold
