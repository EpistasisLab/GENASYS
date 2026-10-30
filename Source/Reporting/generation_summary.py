"""The end-of-generation summary: one record, printed to stdout (the SLURM log) and appended
as one row of generation_summary.csv.

The EA computes every number in the record; this module only formats and writes it, so
the console and the CSV always show the same values. Errors are not caught here; the EA
catches them at its call site.
"""

import csv
import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from typeguard import typechecked

from ..Optimizer.objectives import OBJECTIVE_COST_USD, PERFORMANCE_LABEL, format_objective_2
from ..Utils.log import format_duration

# Cost in the console carries more places than on the plots: a team's median cost can
# be a fraction of a cent on a cheap model.
_COST_DECIMALS = 4


@dataclass(frozen=True)
class Quartiles:
    """Five-number summary of one quantity over a group of teams."""
    min: float
    q1: float
    median: float
    q3: float
    max: float


@dataclass(frozen=True)
class GenerationSummary:
    """One generation's numbers, as reported.

    Candidates are the distinct teams that competed this generation: the initial
    population, or the population plus offspring before truncation. Each one was
    newly evaluated, carried over (a survivor of the previous generation), or
    replayed from this run's archive or an external one.

    The quartiles are over each team's median across its evaluations, split into
    successful and failed candidates; ``None`` when the group is empty. ROC-AUC is
    not summarized for failed teams: their median is 0.0 by definition.

    The totals add up every evaluation of every candidate, whether or not it was
    paid for this generation: an estimate of what evaluating all candidates would
    cost, for following the trend across generations, not this generation's spend.
    Comparable across generations as long as every team carries the same number
    of evaluations (the EA warns when a replayed team does not).
    """
    generation: int
    n_candidates: int
    n_new: int
    n_carried: int
    n_replayed_run: int
    n_replayed_external: int
    n_successful: int
    n_failed: int
    n_survivors: int
    successful_roc_auc: Optional[Quartiles]
    successful_tokens: Optional[Quartiles]
    successful_cost_usd: Optional[Quartiles]
    failed_tokens: Optional[Quartiles]
    failed_cost_usd: Optional[Quartiles]
    total_tokens: int
    total_cost_usd: float
    failed_total_tokens: int
    failed_total_cost_usd: float
    failed_tokens_pct: float
    failed_cost_pct: float
    generation_seconds: float
    evaluation_seconds: float
    team_seconds_mean: float
    team_seconds_min: float
    team_seconds_max: float
    # Named sub-steps of the generation (e.g. selection, offspring) and their seconds.
    stage_times: Dict[str, float]
    # The best successful candidate's metric suite from its latest evaluation; {} if none.
    best_metrics: Dict[str, Any]
    # The survivor plot written for this generation, if any.
    plot_path: Optional[str]


_STATS = ("min", "q1", "median", "q3", "max")
# Quartile fields of GenerationSummary and the decimals each is written with.
_QUARTILE_FIELDS = (("successful_roc_auc", 6), ("successful_tokens", 1),
                    ("successful_cost_usd", 6), ("failed_tokens", 1), ("failed_cost_usd", 6))
_CSV_COLUMNS = (
    "generation", "n_candidates", "n_new", "n_carried", "n_replayed_run",
    "n_replayed_external", "n_successful", "n_failed", "n_survivors",
    *[f"{field}_{stat}" for field, _ in _QUARTILE_FIELDS for stat in _STATS],
    "total_tokens", "total_cost_usd", "failed_total_tokens", "failed_total_cost_usd",
    "failed_tokens_pct", "failed_cost_pct",
    "generation_seconds", "evaluation_seconds", "team_seconds_mean", "team_seconds_min",
    "team_seconds_max")


def _quartile_cells(quartiles: Optional[Quartiles], decimals: int) -> List[str]:
    # An empty group is left blank, not 0: there was nothing to measure.
    if quartiles is None:
        return [""] * len(_STATS)
    return [f"{getattr(quartiles, stat):.{decimals}f}" for stat in _STATS]


@typechecked
def append_generation_summary(summary: GenerationSummary, csv_path: str) -> None:
    """Append ``summary`` as one row; the header is written when the file is created."""
    s = summary
    write_header = not os.path.exists(csv_path)
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(_CSV_COLUMNS)
        writer.writerow([
            s.generation, s.n_candidates, s.n_new, s.n_carried, s.n_replayed_run,
            s.n_replayed_external, s.n_successful, s.n_failed, s.n_survivors,
            *[cell for field, decimals in _QUARTILE_FIELDS
              for cell in _quartile_cells(getattr(s, field), decimals)],
            s.total_tokens, f"{s.total_cost_usd:.6f}",
            s.failed_total_tokens, f"{s.failed_total_cost_usd:.6f}",
            f"{s.failed_tokens_pct:.2f}", f"{s.failed_cost_pct:.2f}",
            f"{s.generation_seconds:.2f}", f"{s.evaluation_seconds:.2f}",
            f"{s.team_seconds_mean:.2f}", f"{s.team_seconds_min:.2f}",
            f"{s.team_seconds_max:.2f}"])
    return


# Where a candidate came from: an initial team (generation -1 only), a survivor of the
# previous generation, or the variation that bred it this generation ('m' mutation,
# 'c' crossover only, 'cm' crossover then mutation).
MEMBER_ORIGINS = ("init", "carried", "m", "c", "cm")
_BRED_ORIGINS = ("m", "c", "cm")


@dataclass(frozen=True)
class GenerationMember:
    """One candidate of one generation, as recorded in generation_members.csv.

    - origin: one of MEMBER_ORIGINS.
    - parent_ids: the archive ids of the parents that bred it this generation; a
      mutation stores its single parent in both slots. None for 'init' and 'carried'.
    - front: its NSGA-II rank among the generation's successful candidates (0 is
      front zero). None for a failed team, which is never ranked.
    - successful: its median performance is above 0.0.
    - survived: it is in the population this generation hands on.
    """
    id: int
    origin: str
    parent_ids: Optional[Tuple[int, int]]
    front: Optional[int]
    successful: bool
    survived: bool


MEMBER_COLUMNS = ("generation", "id", "origin", "parent_1", "parent_2", "front",
                   "successful", "survived")


def _check_member(generation: int, m: GenerationMember) -> None:
    where = f"Generation {generation}, team {m.id}"
    assert m.origin in MEMBER_ORIGINS, f"{where}: unknown origin {m.origin!r}."
    assert (m.origin == "init") == (generation == -1), (
        f"{where}: origin {m.origin!r}; 'init' is the origin of every generation -1 "
        f"team and of no other.")
    assert (m.parent_ids is None) == (m.origin not in _BRED_ORIGINS), (
        f"{where}: origin {m.origin!r} with parent_ids {m.parent_ids}; only bred "
        f"teams have parents.")
    if m.parent_ids is not None:
        assert all(p >= 0 for p in m.parent_ids), f"{where}: parent_ids {m.parent_ids}."
        assert m.origin != "m" or m.parent_ids[0] == m.parent_ids[1], (
            f"{where}: a mutation stores its single parent in both slots: {m.parent_ids}.")
    assert (m.front is None) == (not m.successful), (
        f"{where}: front {m.front} but successful={m.successful}; exactly the "
        f"successful teams are ranked.")
    assert m.front is None or m.front >= 0, f"{where}: front {m.front}."
    assert m.successful or not m.survived, f"{where}: a failed team survived."
    return


@typechecked
def append_generation_members(generation: int, members: List[GenerationMember],
                              csv_path: str) -> None:
    """Append one row per candidate: its archive id, origin, parents, front, and
    whether it succeeded and survived. Parents and front are blank when None.

    The candidates plot of a generation is its rows; the survivors plot is the rows
    with survived = 1. Scores and genomes are looked up in archive.json by id.
    """
    assert members, f"Generation {generation} has no candidates."
    ids = [m.id for m in members]
    assert len(set(ids)) == len(ids), "candidate ids must be distinct."
    for m in members:
        _check_member(generation, m)
    fronts = [m.front for m in members if m.front is not None]
    assert not fronts or min(fronts) == 0, (
        f"Generation {generation}: successful teams are ranked but none is on front 0.")
    write_header = not os.path.exists(csv_path)
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(MEMBER_COLUMNS)
        writer.writerows([generation, m.id, m.origin, *(m.parent_ids or ("", "")),
                          "" if m.front is None else m.front,
                          int(m.successful), int(m.survived)] for m in members)
    return


def _cost(value: float) -> str:
    return format_objective_2(value, OBJECTIVE_COST_USD, _COST_DECIMALS)


def _tokens(value: float) -> str:
    return f"{value:,.0f}"


def _quartile_table(title: str, n: int,
                    rows: Sequence[Tuple[str, Optional[Quartiles], Callable[[float], str]]]) -> List[str]:
    """A titled five-number table, or a single "none" line for an empty group."""
    heading = f"{title} (n={n})"
    if n == 0:
        return [f"  {heading:<20}: none"]
    lines = [f"  {heading:<20}" + "".join(f"{stat:>12}" for stat in
                                          ("min", "Q1", "median", "Q3", "max"))]
    for label, quartiles, fmt in rows:
        lines.append(f"    {label:<18}" + "".join(f"{fmt(getattr(quartiles, stat)):>12}"
                                               for stat in _STATS))
    return lines


@typechecked
def print_generation_summary(summary: GenerationSummary) -> None:
    """Print the end-of-generation block."""
    s = summary
    lines = [
        "=" * 80,
        f"  GENERATION {s.generation} SUMMARY",
        "=" * 80,
        f"  candidates          : {s.n_candidates} ({s.n_new} newly evaluated, "
        f"{s.n_carried} carried over, {s.n_replayed_run + s.n_replayed_external} replayed: "
        f"{s.n_replayed_run} this run, {s.n_replayed_external} external)",
        f"  survivors           : {s.n_survivors}",
        *_quartile_table("successful", s.n_successful, [
            (PERFORMANCE_LABEL, s.successful_roc_auc, lambda v: f"{v:.4f}"),
            ("tokens", s.successful_tokens, _tokens),
            ("cost", s.successful_cost_usd, _cost)]),
        *_quartile_table("failed", s.n_failed, [
            ("tokens", s.failed_tokens, _tokens),
            ("cost", s.failed_cost_usd, _cost)]),
        *_best_metrics_lines(s.best_metrics),
        f"  est. cost of evaluating all {s.n_candidates} candidates (every evaluation):",
        f"    tokens  {s.total_tokens:,}  (failed {s.failed_total_tokens:,} = "
        f"{s.failed_tokens_pct:.1f}%)",
        f"    cost    {_cost(s.total_cost_usd)}  (failed {_cost(s.failed_total_cost_usd)} = "
        f"{s.failed_cost_pct:.1f}%)",
        f"  generation time     : {format_duration(s.generation_seconds)}",
        *[f"      {name:<16}: {format_duration(secs)}" for name, secs in s.stage_times.items()],
        f"  evaluation time     : {format_duration(s.evaluation_seconds)}",
        f"  per evaluated team  : mean {format_duration(s.team_seconds_mean)} | "
        f"min {format_duration(s.team_seconds_min)} | max {format_duration(s.team_seconds_max)}",
    ]
    if s.plot_path:
        lines.append(f"  front plot          : {s.plot_path}")
    lines.append("=" * 80)
    print("\n" + "\n".join(lines) + "\n", flush=True)
    return


def _best_metrics_lines(metrics: Dict[str, Any]) -> List[str]:
    """The best team's reported metrics, one line per family.

    The two threshold families are on separate lines because they are not
    comparable: at a low positive rate a 0.5 cutoff can leave the model predicting
    almost no positives, so its precision/recall/F1 describe a handful of rows. The
    ``@prev`` line cuts at the base rate and is the one to read.
    """
    if not metrics:
        return []

    def fmt(*keys: str) -> str:
        return "  ".join(f"{k.rsplit('_at_', 1)[0]} {float(metrics[k]):.4f}"
                         for k in keys if metrics.get(k) is not None)

    lines = []
    ranking = fmt("roc_auc", "pr_auc", "positive_rate")
    if ranking:
        # The fold-to-fold spread belongs next to the mean: a std larger than the
        # gap between front members means their ordering is not yet real.
        std, folds = metrics.get("roc_auc_std"), metrics.get("n_folds")
        spread = ""
        if std is not None:
            spread = f"  (roc_auc +/-{float(std):.4f}"
            spread += f" across {int(folds)} folds)" if folds else ")"
        lines.append(f"  best team (ranking) : {ranking}{spread}")
    for suffix, note in (("_at_50", "@0.50"), ("_at_prev", "@prev")):
        line = fmt(*(f"{m}{suffix}" for m in
                     ("accuracy", "balanced_accuracy", "precision", "recall", "f1")))
        if line:
            lines.append(f"  best team ({note})   : {line}")
    return lines
