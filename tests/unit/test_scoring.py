import math

import pytest

from neuralstate.consolidation import scoring


def test_recency_halves_each_half_life():
    assert scoring.recency(0) == 1.0
    assert scoring.recency(30) == pytest.approx(0.5)
    assert scoring.recency(60) == pytest.approx(0.25)


def test_reinforce_moves_toward_one_and_never_exceeds_it():
    c = 0.5
    for _ in range(50):
        nxt = scoring.reinforce(c, 0.9)
        assert c < nxt <= 1.0
        c = nxt
    assert c == pytest.approx(1.0, abs=1e-6)


def test_decay_is_flat_during_grace_then_exponential():
    assert scoring.decayed(0.8, idle_days=5, lam=0.1, grace_days=7) == 0.8
    assert scoring.decayed(0.8, idle_days=17, lam=0.1, grace_days=7) == pytest.approx(0.8 * math.exp(-1))


def test_decay_does_not_compound():
    # Recomputed from base each cycle: running it twice gives the same value.
    once = scoring.decayed(0.8, 40, 0.02, 7)
    assert scoring.decayed(0.8, 40, 0.02, 7) == once


def test_more_evidence_wins_contradiction():
    strong = scoring.strength(evidence_count=5, confidence=0.9, days_since_evidence=0)
    weak = scoring.strength(evidence_count=1, confidence=0.7, days_since_evidence=0)
    side, margin = scoring.contradiction_margin(weak, strong)
    assert side == "b" and margin > 0.2


def test_fresh_observation_beats_stale_single_one():
    fresh = scoring.strength(1, 0.7, 0)
    stale = scoring.strength(1, 0.7, 30)
    side, margin = scoring.contradiction_margin(fresh, stale)
    assert side == "a" and margin >= 0.2


def test_equal_support_is_a_tie():
    s = scoring.strength(2, 0.7, 1)
    assert scoring.contradiction_margin(s, s) == ("a", 0.0)


def test_rule_confidence_tracks_evidence_and_effectiveness():
    base = scoring.rule_confidence(5, 0, 0, min_evidence=5)
    assert base == pytest.approx(1 - math.exp(-1))
    assert scoring.rule_confidence(5, 4, 0, 5) > base
    assert scoring.rule_confidence(5, 0, 4, 5) < base
    assert 0 <= scoring.rule_confidence(50, 100, 0, 5) <= 1


def test_rule_retires_only_after_enough_failed_trials():
    assert not scoring.rule_should_retire(0, 2, min_trials=3, threshold=0.3)
    assert scoring.rule_should_retire(0, 3, min_trials=3, threshold=0.3)
    assert not scoring.rule_should_retire(3, 3, min_trials=3, threshold=0.3)
