"""A finished (or interrupted) run, read back from its folder for the animations.

Everything comes from two files the EA already writes: generation_members.csv (which
teams competed in each generation, where each came from, its parents, its NSGA-II front
and whether it succeeded and survived) and archive.json (each team's scores). Nothing
here touches the EA, so the animations can be rebuilt for any run folder, at any time,
with different styling.

Fronts come from the CSV as the EA ranked them. Each team is placed at the medians the EA
ranked on: the median across its evaluations, with 0.0 median performance meaning failed.
Keep these in step with EA.objective_2_of and EA.viable_positions; the loader checks the
failures against the CSV's successful column.
"""

import csv
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from typeguard import typechecked

from ..Optimizer.objectives import OBJECTIVE_COST_USD, VALID_OBJECTIVES
from .generation_summary import MEMBER_COLUMNS


@dataclass(frozen=True)
class TeamPoint:
    """One archived team, as plotted."""
    id: int
    performance: float
    objective2: float
    failed: bool


@dataclass(frozen=True)
class GenerationFrame:
    """One generation's teams, by archive id.

    - candidates: every team that competed, failures included: the initial teams for
      generation -1, otherwise the carried population plus the offspring.
    - origins: each candidate's origin ('init', 'carried', 'm', 'c' or 'cm').
    - parents: each candidate's parent ids this generation; None unless bred.
    - fronts: each candidate's NSGA-II rank among the successful candidates; None
      for a failed team.
    - carried: the candidates that were the previous generation's survivors.
    - survivors: the population this generation hands on.
    - front: the candidates on front zero, survivors or not, ordered by objective 2.
    - survivor_front: the survivors on front zero (front zero of the population),
      ordered by objective 2.
    - dominated: the remaining survivors.
    """
    generation: int
    candidates: List[int]
    origins: Dict[int, str]
    parents: Dict[int, Optional[Tuple[int, int]]]
    fronts: Dict[int, Optional[int]]
    carried: List[int]
    survivors: List[int]
    front: List[int]
    survivor_front: List[int]
    dominated: List[int]


@dataclass(frozen=True)
class RunHistory:
    objective_2: str
    teams: Dict[int, TeamPoint]
    generations: List[GenerationFrame]


@typechecked
def load_run_history(run_dir: str, objective_2: str) -> RunHistory:
    """Read ``archive.json`` and ``generation_members.csv`` from ``run_dir``.

    ``objective_2`` must be the one the run optimized; the archive keeps both cost
    and token count, so the files alone do not say which was ranked on.
    """
    assert objective_2 in VALID_OBJECTIVES, f"Unknown objective_2: {objective_2}."
    with open(os.path.join(run_dir, "archive.json")) as f:
        archive = json.load(f)
    teams = {int(entry["id"]): _team_point(entry, objective_2) for entry in archive["teams"]}

    members_path = os.path.join(run_dir, "generation_members.csv")
    rows: Dict[int, List[Dict[str, str]]] = {}
    with open(members_path, newline="") as f:
        reader = csv.DictReader(f)
        assert tuple(reader.fieldnames or ()) == MEMBER_COLUMNS, (
            f"{members_path} has columns {reader.fieldnames}, expected {list(MEMBER_COLUMNS)}; "
            f"it was written by an older version of the EA.")
        for row in reader:
            rows.setdefault(int(row["generation"]), []).append(row)
    assert rows, f"{members_path} has no generations."
    gens = sorted(rows)
    assert gens == list(range(-1, gens[-1] + 1)), (
        f"generation_members.csv must hold generations -1 .. N without gaps: {gens}.")

    frames: List[GenerationFrame] = []
    previous: List[int] = []
    for g in gens:
        frame = _frame(g, rows[g], teams)
        if g >= 0:
            assert set(frame.carried) == set(previous), (
                f"Generation {g} carries {sorted(frame.carried)}, but generation {g - 1} "
                f"kept {sorted(previous)}.")
            outside = sorted({p for ids in frame.parents.values() if ids for p in ids}
                             - set(previous))
            assert not outside, (
                f"Generation {g} cites parents {outside} that generation {g - 1} did not keep.")
        frames.append(frame)
        previous = frame.survivors
    return RunHistory(objective_2=objective_2, teams=teams, generations=frames)


def _frame(g: int, rows: List[Dict[str, str]], teams: Dict[int, TeamPoint]) -> GenerationFrame:
    """One generation's rows, checked against the archive."""
    candidates = [int(r["id"]) for r in rows]
    missing = sorted(set(candidates) - set(teams))
    assert not missing, f"Generation {g} lists ids missing from archive.json: {missing}."
    origins = {int(r["id"]): r["origin"] for r in rows}
    parents = {int(r["id"]): ((int(r["parent_1"]), int(r["parent_2"])) if r["parent_1"] else None)
               for r in rows}
    fronts = {int(r["id"]): (int(r["front"]) if r["front"] else None) for r in rows}
    for r in rows:
        i = int(r["id"])
        assert (r["successful"] == "1") == (not teams[i].failed), (
            f"Generation {g}, team {i}: successful={r['successful']} in the CSV, but its "
            f"archived median performance is {teams[i].performance}.")
    survivors = [int(r["id"]) for r in rows if r["survived"] == "1"]

    def by_objective2(ids: List[int]) -> List[int]:
        return sorted(ids, key=lambda i: teams[i].objective2)  # stable: ties keep CSV order

    survivor_front = by_objective2([i for i in survivors if fronts[i] == 0])
    return GenerationFrame(
        generation=g, candidates=candidates, origins=origins, parents=parents, fronts=fronts,
        carried=[i for i in candidates if origins[i] == "carried"],
        survivors=survivors,
        front=by_objective2([i for i in candidates if fronts[i] == 0]),
        survivor_front=survivor_front,
        dominated=[i for i in survivors if i not in set(survivor_front)])


def _team_point(entry: Dict, objective_2: str) -> TeamPoint:
    """Median performance and objective 2 of one archive.json entry."""
    assert entry["performance"], f"Team {entry['id']} in archive.json was never evaluated."
    performance = float(np.median(entry["performance"]))
    if objective_2 == OBJECTIVE_COST_USD:
        objective2 = float(np.median(entry["cost"]))
    else:
        # Rounded like Team.get_median_token_count.
        objective2 = float(int(round(float(np.median(entry["token_count"])))))
    return TeamPoint(id=int(entry["id"]), performance=performance, objective2=objective2,
                     failed=performance <= 0.0)
