"""Average-linkage agglomerative clustering over unit vectors (cosine similarity)."""

from __future__ import annotations

import numpy as np


def cluster(vectors: np.ndarray | list[list[float]], threshold: float) -> list[list[int]]:
    """Group row indices so every merge joins clusters whose mean pairwise similarity >= threshold.

    Uses the Lance-Williams update (no pairwise recomputation), O(n^2) memory and one
    vectorized O(n^2) argmax per merge. That is fine for the few hundred failures per task
    type the synthesizer feeds it. Returns clusters largest first.
    """
    x = np.asarray(vectors, dtype=np.float64)
    n = len(x)
    if n == 0:
        return []
    if n == 1:
        return [[0]]

    sim = x @ x.T
    np.fill_diagonal(sim, -np.inf)
    sizes = np.ones(n)
    members: list[list[int] | None] = [[i] for i in range(n)]
    alive = np.ones(n, dtype=bool)

    while alive.sum() > 1:
        masked = np.where(alive[:, None] & alive[None, :], sim, -np.inf)
        i, j = np.unravel_index(int(np.argmax(masked)), masked.shape)
        if masked[i, j] < threshold:
            break
        # Merge j into i: the average similarity to every other cluster is size-weighted.
        si, sj = sizes[i], sizes[j]
        merged = (si * sim[i] + sj * sim[j]) / (si + sj)
        sim[i, :] = merged
        sim[:, i] = merged
        sim[i, i] = -np.inf
        sizes[i] = si + sj
        members[i] = members[i] + members[j]  # type: ignore[operator]
        members[j] = None
        alive[j] = False
        sim[j, :] = -np.inf
        sim[:, j] = -np.inf

    groups = [sorted(m) for m in members if m is not None]
    return sorted(groups, key=len, reverse=True)


def centroid(vectors: np.ndarray | list[list[float]]) -> list[float]:
    c = np.asarray(vectors, dtype=np.float64).mean(axis=0)
    norm = np.linalg.norm(c) or 1.0
    return (c / norm).tolist()
