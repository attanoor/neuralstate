import numpy as np

from neuralstate.consolidation.cluster import centroid, cluster


def _unit(v):
    v = np.asarray(v, dtype=float)
    return v / np.linalg.norm(v)


def test_empty_and_single():
    assert cluster([], 0.5) == []
    assert cluster([_unit([1, 0])], 0.5) == [[0]]


def test_separates_two_groups():
    rng = np.random.default_rng(0)
    a = [_unit([1, 0, 0] + rng.normal(0, 0.05, 3)) for _ in range(6)]
    b = [_unit([0, 1, 0] + rng.normal(0, 0.05, 3)) for _ in range(4)]
    groups = cluster(a + b, threshold=0.8)
    assert groups == [list(range(6)), list(range(6, 10))]


def test_threshold_one_keeps_distinct_points_apart():
    pts = [_unit([1, 0]), _unit([0.9, 0.1]), _unit([0, 1])]
    assert sorted(map(len, cluster(pts, 0.9999))) == [1, 1, 1]


def test_average_linkage_resists_chaining():
    # A chain where neighbors are similar but the ends are not: average linkage stops early.
    angles = np.linspace(0, np.pi / 2, 8)
    pts = [[np.cos(t), np.sin(t)] for t in angles]
    groups = cluster(pts, threshold=0.9)
    assert len(groups) > 1


def test_centroid_is_unit_length():
    c = centroid([_unit([1, 0]), _unit([0, 1])])
    assert abs(np.linalg.norm(c) - 1) < 1e-9
    assert c[0] == c[1]
