"""NSGA-II building blocks: non-dominated sorting, crowding distance, binary tournament and
truncation.

Objective direction
-------------------
Every public function takes the two objectives as they are measured: performance is
maximized and objective 2 (USD cost or token count) is minimized. The direction is handled
here, once, so callers never negate or weight their scores.

The numba kernels (leading underscore) assume clean input; the public wrappers check it
first. A NaN would compare False against everything, so it would be neither dominated nor
dominating and would land on the front -- the wrappers reject non-finite values outright.
"""

from typing import List, Tuple

import numpy as np
import numpy.typing as npt
from numba import njit
from typeguard import typechecked


@njit(cache=True)
def _non_dominated_sorting(performances: npt.NDArray[np.float64],
                           objective2: npt.NDArray[np.float64]) -> npt.NDArray[np.int64]:
    """Rank every individual by front; 0 is the non-dominated front.

    Both arrays are maximized here: the caller passes objective 2 negated.
    """
    n = len(performances)
    ranks = np.zeros(n, dtype=np.int64)
    domination_count = np.zeros(n, dtype=np.int64)
    # dominates[i, j] = 1 iff i dominates j
    dominates = np.zeros((n, n), dtype=np.int8)

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            # i dominates j: at least as good on both objectives, strictly better on one.
            if (performances[i] >= performances[j] and objective2[i] >= objective2[j]
                    and (performances[i] > performances[j] or objective2[i] > objective2[j])):
                dominates[i, j] = 1
                domination_count[j] += 1

    current_front = np.where(domination_count == 0)[0]
    current_rank = 0
    while len(current_front) > 0:
        for idx in current_front:
            ranks[idx] = current_rank
        next_front = []
        for i in current_front:
            for j in range(n):
                if dominates[i, j] == 1:
                    domination_count[j] -= 1
                    # Reaches 0 exactly once per j, so no duplicates.
                    if domination_count[j] == 0:
                        next_front.append(j)
        current_front = np.array(next_front, dtype=np.int64)
        current_rank += 1

    return ranks


@njit(cache=True)
def _crowding_distance_one_front(performances: npt.NDArray[np.float64],
                                 objective2: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Crowding distance among the members of a single front.

    The two boundary points of each objective get infinity; every interior point gets
    the normalized gap between its neighbours, summed over both objectives. Gaps are
    absolute, so the sign convention of either objective does not matter.

    Ties are left to argsort order on purpose. Within one front a tie on one objective
    implies a tie on both, so tied members are identical points; one copy then gets the
    smaller distance and is dropped first by truncation, which keeps duplicates from
    crowding out distinct solutions. Exact ties are rare in practice: failed teams are
    excluded before sorting, and ROC-AUC and cost/token counts are continuous.
    """
    n = len(performances)
    distances = np.zeros(n, dtype=np.float64)
    for values in (performances, objective2):
        order = np.argsort(values)
        ordered = values[order]
        distances[order[0]] = np.inf
        distances[order[-1]] = np.inf
        value_range = ordered[-1] - ordered[0]
        if value_range > 0:
            for k in range(1, n - 1):
                distances[order[k]] += (ordered[k + 1] - ordered[k - 1]) / value_range
    return distances


def _check_objectives(performances: npt.ArrayLike,
                      objective2: npt.ArrayLike) -> Tuple[np.ndarray, np.ndarray]:
    """Validate and convert to contiguous 1-D float64 arrays of equal length."""
    perf = np.ascontiguousarray(performances, dtype=np.float64)
    obj2 = np.ascontiguousarray(objective2, dtype=np.float64)
    assert perf.ndim == 1 and obj2.ndim == 1, (
        f"Objectives must be 1-D: performances {perf.shape}, objective2 {obj2.shape}.")
    assert perf.size == obj2.size, (
        f"performances ({perf.size}) and objective2 ({obj2.size}) must be the same length.")
    assert perf.size > 0, "Objectives cannot be empty."
    assert np.isfinite(perf).all() and np.isfinite(obj2).all(), (
        "Objectives must be finite; a NaN is never dominated and would sit on the front.")
    return perf, obj2


@typechecked
def non_dominated_sorting(performances: npt.ArrayLike,
                          objective2: npt.ArrayLike) -> npt.NDArray[np.int64]:
    """Front rank of each individual (0 = non-dominated).

    Args:
        performances: to maximize.
        objective2: to minimize, as measured (USD or token count; not negated). Kept
            float: an int cast would truncate every cost under $1 to 0, so every
            individual would tie and the front would collapse to argmax(performance).
    """
    perf, obj2 = _check_objectives(performances, objective2)
    return _non_dominated_sorting(perf, -obj2)


@typechecked
def crowding_distance(performances: npt.ArrayLike, objective2: npt.ArrayLike,
                      ranks: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Crowding distance of each individual, measured within its own front.

    Each front is handled on its own, so the extremes of every front are infinite
    and a point's neighbours are always members of its front -- as NSGA-II defines
    it. ``ranks`` comes from ``non_dominated_sorting`` on the same objectives.
    """
    perf, obj2 = _check_objectives(performances, objective2)
    ranks = np.asarray(ranks, dtype=np.int64)
    assert ranks.shape == perf.shape, (
        f"ranks ({ranks.size}) must match the objectives ({perf.size}).")
    distances = np.empty(perf.size, dtype=np.float64)
    for rank in np.unique(ranks):
        members = np.where(ranks == rank)[0]
        distances[members] = _crowding_distance_one_front(perf[members], obj2[members])
    return distances


@typechecked
def nds_tournament(ranks: npt.NDArray[np.int64], distances: npt.NDArray[np.float64],
                   t1: int, t2: int) -> int:
    """Binary tournament: the lower rank wins, then the larger crowding distance.

    A full tie returns ``t2``; ``t1`` and ``t2`` are drawn at random, so this
    introduces no bias toward either position.
    """
    if ranks[t1] != ranks[t2]:
        return t1 if ranks[t1] < ranks[t2] else t2
    return t1 if distances[t1] > distances[t2] else t2


@typechecked
def nds_truncate(ranks: npt.NDArray[np.int64], distances: npt.NDArray[np.float64],
                 n_keep: int, rng: np.random.Generator) -> npt.NDArray[np.int64]:
    """Indices of the ``n_keep`` best individuals (all of them if there are fewer).

    Whole fronts are kept in rank order; the first front that does not fit is split
    by crowding distance, largest first, so its boundary points are kept first.
    Individuals tied on distance at the cut are chosen at random with ``rng`` (the
    EA's generator, so a run stays reproducible from its seed) rather than by
    position. Plain Python, not numba: a numpy Generator cannot be passed into a
    jitted function, and the arrays are only population-sized.
    """
    assert n_keep > 0, f"n_keep must be positive: {n_keep}."
    assert ranks.shape == distances.shape, (
        f"ranks ({ranks.size}) and distances ({distances.size}) must be the same length.")
    if n_keep >= ranks.size:
        return np.arange(ranks.size)

    selected: List[int] = []
    for rank in np.unique(ranks):  # np.unique returns sorted values
        front = np.where(ranks == rank)[0]
        room = n_keep - len(selected)
        if len(front) <= room:
            selected.extend(front.tolist())
        else:
            # Shuffle first, then a stable sort by descending distance: distinct
            # distances still go largest first, and ties keep the random order.
            shuffled = rng.permutation(front)
            order = shuffled[np.argsort(-distances[shuffled], kind="stable")]
            selected.extend(order[:room].tolist())
        if len(selected) == n_keep:
            break
    return np.asarray(selected, dtype=np.int64)


@typechecked
def front_indices(performances: npt.ArrayLike,
                  objective2: npt.ArrayLike) -> npt.NDArray[np.int64]:
    """Positions of the non-dominated (rank 0) individuals, in input order."""
    return np.where(non_dominated_sorting(performances, objective2) == 0)[0]
