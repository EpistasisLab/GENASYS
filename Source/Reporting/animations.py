"""Animated fronts: one GIF frame per generation, for slides and web pages.

  population.gif  the survivors of each generation, front zero in red, the rest in blue.
  candidates.gif  everyone who competed (population + offspring), colored by whether it
                  survived, under the candidates' front zero.
  offspring.gif   where each generation's offspring came from: shape and color give the
                  variation that bred a child, lines join it to its parents.
  front_zero.gif  front zero of each generation's candidates (population + offspring),
                  with the earlier fronts fading behind it.

The last three draw the same front zero: the candidates', with the teams crowding
distance cut marked. Every marker carries its archive id, with its performance and
objective 2 printed below it, as in all_front.png. Each frame is also saved as a PNG
(Population_PNG/gen03.png, ...) at twice the resolution, for slides. Everything goes
under {run_dir}/Visuals: the GIFs in GIFs/, each GIF's frames in its *_PNG folder,
and the EA's all_front.png in All_Front_PNG/.

Two entry points:
  save_generation_frames  during the run: one generation's four PNGs, scaled to the
                          run so far, so a long run can be watched.
  make_run_gifs           at the end: every frame redrawn on one scale, and the GIFs.

Drawing only: the fronts, origins and outcomes are the EA's, read back from the run
folder by run_history.py. In make_run_gifs every frame of every GIF shares one figure
size, one marker size and one set of axis limits (taken from the whole run), so a
point that moves on screen really moved in objective space. Cost is plotted on a
linear axis, token count on a log axis.

Errors are not caught here; the caller decides whether a broken GIF matters.
"""

import os
from dataclasses import dataclass
from typing import Callable, List, Tuple

import matplotlib
matplotlib.use("Agg")  # must precede the pyplot import
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter  # noqa: E402
from PIL import Image  # noqa: E402
from typeguard import typechecked  # noqa: E402

from ..Optimizer.objectives import (OBJECTIVE_2_LABELS, OBJECTIVE_COST_USD,  # noqa: E402
                                    PERFORMANCE_LABEL, PERFORMANCE_NO_SKILL,
                                    format_objective_2)
from .run_history import GenerationFrame, RunHistory, load_run_history  # noqa: E402

# Every plot of a run, under the run folder.
VISUALS_DIR = "Visuals"
GIF_DIR = "GIFs"
ALL_FRONT_DIR = "All_Front_PNG"
# Each GIF, in the order written -> the folder its frames are saved to.
PNG_DIRS = {"population": "Population_PNG", "candidates": "Candidate_PNG",
            "offspring": "Offspring_PNG", "front_zero": "Front_Zero_PNG"}

# 1100 x 700 pixels: sharp on a slide, light enough for a web page. The PNG frames are
# the same figure at twice the resolution.
_FIGSIZE = (11.0, 7.0)
_DPI = 100
_PNG_DPI = 200
# Milliseconds every frame is held.
_FRAME_MS = 3000
_COST_DECIMALS = 2

_FRONT_EDGE, _FRONT_FACE = "#c0392b", "#f5b7b1"
_DOMINATED_EDGE, _DOMINATED_FACE = "#5d6d7e", "#d6eaf8"
_SURVIVED_EDGE, _SURVIVED_FACE = "#1e8449", "#abebc6"
_ELIMINATED_EDGE, _ELIMINATED_FACE = "#7f8c8d", "#f2f3f4"
_FAILED_EDGE, _FAILED_FACE = "#7f8c8d", "#eaecee"
_PARENT_EDGE, _PARENT_FACE = "#5d6d7e", "#d5d8dc"
_LINEAGE_ALPHA = 0.35
# Id text: dark red on front zero, grey on a failure, slate otherwise.
_FRONT_ID, _FAILED_ID, _ID = "#7b241c", "#7f8c8d", "#2c3e50"

# The variation that bred an offspring (generation_members.csv's origin) -> marker and
# colors. The faces are light so the id inside stays readable.
_VARIATION_MARKERS = {"m": "^", "c": "s", "cm": "D"}
_VARIATION_EDGES = {"m": "#2874a6", "c": "#d68910", "cm": "#7d3c98"}
_VARIATION_FACES = {"m": "#aed6f1", "c": "#fad7a0", "cm": "#d2b4de"}
_VARIATION_LABELS = {"m": "mutation", "c": "crossover", "cm": "crossover + mutation"}

Axes = Tuple[Tuple[float, float], Tuple[float, float]]


@dataclass(frozen=True)
class _Style:
    """Marker area and label sizes (points), one set for the whole run."""
    marker_s: float
    id_fs: float
    label_fs: float
    label_dy: float


@typechecked
def gen_tag(generation: int, gen_width: int) -> str:
    """A frame's file name stem: ``gen-1`` for the initial population, else zero-padded
    to ``gen_width`` digits (``gen07``) so the files sort in generation order."""
    assert generation >= -1 and gen_width >= 1, (generation, gen_width)
    if generation < 0:
        return f"gen{generation}"
    assert len(str(generation)) <= gen_width, (
        f"Generation {generation} does not fit in {gen_width} digits.")
    return f"gen{generation:0{gen_width}d}"


@typechecked
def frame_path(run_dir: str, gif: str, generation: int, gen_width: int) -> str:
    """Where ``make_run_gifs`` saves ``generation``'s frame of ``gif`` (a key of
    ``PNG_DIRS``)."""
    assert gif in PNG_DIRS, f"Unknown GIF {gif!r}; expected one of {list(PNG_DIRS)}."
    return os.path.join(run_dir, VISUALS_DIR, PNG_DIRS[gif], f"{gen_tag(generation, gen_width)}.png")


@typechecked
def save_generation_frames(run_dir: str, objective_2: str, gen_width: int,
                           generation: int) -> List[str]:
    """Read the run in ``run_dir`` so far and save ``generation``'s frame of each GIF as a
    PNG, scaled to every generation up to it. ``make_run_gifs`` redraws them all on the
    whole run's scale. Returns the PNG paths written, in ``PNG_DIRS`` order."""
    history = load_run_history(run_dir, objective_2)
    ks = [k for k, frame in enumerate(history.generations) if frame.generation == generation]
    assert ks, f"{run_dir} has no generation {generation} in generation_members.csv."
    # Only the generations up to this one: a rerun of an earlier generation's frames
    # must not be scaled to later results.
    history = RunHistory(objective_2=history.objective_2, teams=history.teams,
                         generations=history.generations[:ks[0] + 1])
    fig, ax = plt.subplots(figsize=_FIGSIZE, dpi=_DPI)
    paths = []
    try:
        for name, (draw, limits) in _draws(history).items():
            path = frame_path(run_dir, name, generation, gen_width)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            _render_frame(fig, ax, history, draw, ks[0], limits)
            fig.savefig(path, dpi=_PNG_DPI)
            paths.append(path)
    finally:
        plt.close(fig)
    return paths


@typechecked
def make_run_gifs(run_dir: str, objective_2: str, gen_width: int) -> List[str]:
    """Read the run in ``run_dir`` and write its four GIFs, and every frame of each as a
    PNG, under ``{run_dir}/Visuals``. ``gen_width`` is the EA's (``EA.gen_width``),
    so the frame names stay the same across rebuilds. Returns the GIF paths written."""
    return save_run_gifs(load_run_history(run_dir, objective_2),
                         os.path.join(run_dir, VISUALS_DIR), gen_width)


@typechecked
def save_run_gifs(history: RunHistory, out_dir: str, gen_width: int) -> List[str]:
    """Write population.gif, candidates.gif, offspring.gif and front_zero.gif to
    ``{out_dir}/GIFs``, and their frames to ``out_dir``'s ``PNG_DIRS`` folders."""
    paths = []
    for name, (draw, limits) in _draws(history).items():
        path = os.path.join(out_dir, GIF_DIR, f"{name}.gif")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        png_dir = os.path.join(out_dir, PNG_DIRS[name])
        os.makedirs(png_dir, exist_ok=True)
        _write_gif(history, draw, limits, path, png_dir, gen_width)
        paths.append(path)
    return paths


def _draws(history: RunHistory):
    """Each GIF's draw function and axis limits, in ``PNG_DIRS`` order."""
    teams = history.teams
    candidate_ids = sorted({i for frame in history.generations for i in frame.candidates})
    survivor_ids = sorted({i for frame in history.generations for i in frame.survivors})
    front_ids = sorted({i for frame in history.generations for i in frame.front})
    # One x range for all four GIFs; y spans the survivors and the fronts (a front-zero
    # team can be cut), and reaches down to 0.0 on the candidates and offspring GIFs
    # only if some candidate failed.
    x_limits = _x_limits([teams[i].objective2 for i in candidate_ids], history.objective_2)
    survivor_y = _y_limits([teams[i].performance for i in sorted(set(survivor_ids) | set(front_ids))])
    candidate_y = _y_limits([teams[i].performance for i in candidate_ids])
    draws = {"population": (_draw_population, (x_limits, survivor_y)),
             "candidates": (_draw_candidates, (x_limits, candidate_y)),
             "offspring": (_draw_offspring, (x_limits, candidate_y)),
             "front_zero": (_draw_front_zero, (x_limits, survivor_y))}
    return {name: draws[name] for name in PNG_DIRS}


def _write_gif(history: RunHistory, draw: Callable, limits: Axes, path: str, png_dir: str,
               gen_width: int) -> None:
    """Render one frame per generation with ``draw``, saving each to ``png_dir`` and all
    of them as a looping GIF."""
    fig, ax = plt.subplots(figsize=_FIGSIZE, dpi=_DPI)
    images = []
    try:
        for k, frame in enumerate(history.generations):
            _render_frame(fig, ax, history, draw, k, limits)
            images.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())).convert("RGB"))
            fig.savefig(os.path.join(png_dir, f"{gen_tag(frame.generation, gen_width)}.png"),
                        dpi=_PNG_DPI)
    finally:
        # Closed even when drawing fails, so a caller looping over runs does not leak.
        plt.close(fig)
    images[0].save(path, save_all=True, append_images=images[1:], duration=_FRAME_MS,
                   loop=0)
    return


def _render_frame(fig, ax, history: RunHistory, draw: Callable, k: int, limits: Axes) -> None:
    """Draw generation ``k`` of ``history`` onto ``ax``, ready to save."""
    frame = history.generations[k]
    ax.clear()
    title, subtitle = draw(ax, history, k, frame, limits)
    _style_axes(ax, history.objective_2, limits)
    last = history.generations[-1].generation
    ax.set_title(f"{title}\n{subtitle}", fontsize=13)
    progress = ("initial population" if frame.generation < 0
                else f"generation {frame.generation} of {last}")
    ax.text(0.01, 0.98, progress,
            transform=ax.transAxes, ha="left", va="top", fontsize=11, color="0.3")
    fig.tight_layout()
    fig.canvas.draw()
    return


def _draw_population(ax, history: RunHistory, k: int, frame: GenerationFrame, limits: Axes):
    style = _style(history)
    _scatter(ax, history, frame.dominated, "o", _DOMINATED_FACE, _DOMINATED_EDGE, style,
             f"dominated (n={len(frame.dominated)})")
    _front_line(ax, history, frame.survivor_front, style,
                f"front zero (n={len(frame.survivor_front)})", markers=True)
    _label_points(ax, history, frame.dominated, limits, style, _ID)
    _label_points(ax, history, frame.survivor_front, limits, style, _FRONT_ID)
    title = ("Initial population" if frame.generation < 0
             else f"Population after generation {frame.generation}")
    return title, f"{len(frame.survivors)} survivors, {len(frame.survivor_front)} on front zero"


def _draw_candidates(ax, history: RunHistory, k: int, frame: GenerationFrame, limits: Axes):
    teams, style = history.teams, _style(history)
    survivors, carried = set(frame.survivors), set(frame.carried)
    failed = [i for i in frame.candidates if teams[i].failed]
    eliminated = [i for i in frame.candidates if not teams[i].failed and i not in survivors]
    kept = [i for i in frame.candidates if not teams[i].failed and i in survivors]
    _scatter(ax, history, eliminated, "o", _ELIMINATED_FACE, _ELIMINATED_EDGE, style,
             f"eliminated (n={len(eliminated)})")
    _scatter(ax, history, kept, "o", _SURVIVED_FACE, _SURVIVED_EDGE, style,
             f"survived (n={len(kept)})")
    _failures(ax, history, failed, limits, style)
    _candidate_front(ax, history, frame, style, markers=False)
    _label_candidates(ax, history, frame, limits, style)

    if frame.generation < 0:
        return "Initial teams", f"{len(frame.candidates)} teams evaluated"
    bred = len(frame.candidates) - len(carried)
    return (f"Generation {frame.generation}: population + offspring",
            f"{len(carried)} carried over + {bred} offspring, {len(survivors)} survive")


def _draw_offspring(ax, history: RunHistory, k: int, frame: GenerationFrame, limits: Axes):
    teams, style = history.teams, _style(history)
    bred = [i for i in frame.candidates if frame.parents[i] is not None]
    parent_ids = {p for i in bred for p in frame.parents[i]}
    if frame.generation < 0:
        # Nothing bred yet: the initial teams stand in as the first parents.
        _scatter(ax, history, [i for i in frame.candidates if not teams[i].failed], "o",
                 _PARENT_FACE, _PARENT_EDGE, style, f"initial team (n={len(frame.candidates)})")
        _failures(ax, history, [i for i in frame.candidates if teams[i].failed], limits, style)
        _candidate_front(ax, history, frame, style, markers=False)
        _label_candidates(ax, history, frame, limits, style)
        return "Offspring: initial population", "no offspring yet"

    parents = [i for i in frame.carried if i in parent_ids]
    idle = [i for i in frame.carried if i not in parent_ids]
    _scatter(ax, history, parents, "o", _PARENT_FACE, _PARENT_EDGE, style,
             f"parent (n={len(parents)})")
    if idle:
        ax.scatter(*_pinned_xy(history, idle, limits), s=style.marker_s, marker="o",
                   c=_PARENT_FACE, edgecolors=_PARENT_EDGE, linewidths=1.2, alpha=0.35,
                   zorder=3, label=f"not a parent (n={len(idle)})")

    # One line per distinct parent: a mutation's parent_1 == parent_2 draws one.
    for i in bred:
        child_x, child_y = _pinned_xy(history, [i], limits)
        for p in sorted(set(frame.parents[i])):
            parent_x, parent_y = _pinned_xy(history, [p], limits)
            ax.plot([parent_x[0], child_x[0]], [parent_y[0], child_y[0]],
                    color=_VARIATION_EDGES[frame.origins[i]], linewidth=1.0,
                    alpha=_LINEAGE_ALPHA, zorder=1, gid="lineage")

    for origin, marker in _VARIATION_MARKERS.items():
        ids = [i for i in bred if frame.origins[i] == origin and not teams[i].failed]
        _scatter(ax, history, ids, marker, _VARIATION_FACES[origin], _VARIATION_EDGES[origin],
                 style, None)
    _failures(ax, history, [i for i in bred if teams[i].failed], limits, style)
    _candidate_front(ax, history, frame, style, markers=False)
    _label_candidates(ax, history, frame, limits, style)

    # Legend proxies (no data): one entry per variation present, failures in its count.
    for origin, marker in _VARIATION_MARKERS.items():
        n = sum(1 for i in bred if frame.origins[i] == origin)
        if n:
            ax.plot([], [], linestyle="", marker=marker, markersize=9,
                    markerfacecolor=_VARIATION_FACES[origin],
                    markeredgecolor=_VARIATION_EDGES[origin],
                    label=f"{_VARIATION_LABELS[origin]} (n={n})")
    if bred:
        ax.plot([], [], color="0.4", linewidth=1.0, alpha=_LINEAGE_ALPHA * 2,
                label="parent → offspring")
    return (f"Generation {frame.generation}: offspring and their parents",
            f"{len(bred)} offspring from {len(parents)} of {len(frame.carried)} carried teams")


def _draw_front_zero(ax, history: RunHistory, k: int, frame: GenerationFrame, limits: Axes):
    style = _style(history)
    # Earlier fronts fade with age; the one just before this generation is dashed.
    earlier = history.generations[:k]
    for age, old in enumerate(reversed(earlier)):
        if not old.front:
            continue
        x, y = _xy(history, old.front)
        if age == 0:
            ax.plot(x, y, drawstyle="steps-post", color="0.45", linestyle="--",
                    linewidth=1.4, zorder=2, label="previous front zero")
        else:
            ax.plot(x, y, drawstyle="steps-post", color="0.6", linewidth=1.0, zorder=1,
                    alpha=max(0.12, 0.5 * 0.8 ** age))
    _candidate_front(ax, history, frame, style, markers=True)
    _label_points(ax, history, frame.front, limits, style, _FRONT_ID)
    title = ("Front zero: initial population" if frame.generation < 0
             else f"Front zero of generation {frame.generation}: population + offspring")
    return title, f"{len(frame.front)} non-dominated teams"


# -------------------------------- shared pieces ------------------------------------------

def _style(history: RunHistory) -> _Style:
    """Marks scaled to the run's most crowded generation, as plots.py scales them to
    one plot: ten teams can afford big numbered markers, forty-five cannot."""
    n = max(len(frame.candidates) for frame in history.generations)
    if n <= 12:
        return _Style(marker_s=300, id_fs=7.0, label_fs=6.5, label_dy=-3)
    if n <= 25:
        return _Style(marker_s=200, id_fs=6.0, label_fs=5.5, label_dy=-2)
    return _Style(marker_s=130, id_fs=5.0, label_fs=4.5, label_dy=-1)


def _xy(history: RunHistory, ids: List[int]):
    return (np.array([history.teams[i].objective2 for i in ids], dtype=np.float64),
            np.array([history.teams[i].performance for i in ids], dtype=np.float64))


def _pinned_xy(history: RunHistory, ids: List[int], limits: Axes):
    """``_xy``, with x held inside the left edge: a failure can report 0 tokens, which a
    log axis cannot place, so it is pinned there rather than lost."""
    x, y = _xy(history, ids)
    return np.maximum(x, limits[0][0]), y


def _scatter(ax, history: RunHistory, ids: List[int], marker: str, face: str, edge: str,
             style: _Style, label) -> None:
    if ids:
        ax.scatter(*_xy(history, ids), s=style.marker_s, marker=marker, c=face,
                   edgecolors=edge, linewidths=1.2, zorder=3, label=label)
    return


def _failures(ax, history: RunHistory, ids: List[int], limits: Axes, style: _Style) -> None:
    if ids:
        ax.scatter(*_pinned_xy(history, ids, limits), s=style.marker_s, marker="X",
                   c=_FAILED_FACE, edgecolors=_FAILED_EDGE, linewidths=0.8, zorder=3,
                   label=f"failed (n={len(ids)})")
    return


def _front_line(ax, history: RunHistory, front: List[int], style: _Style, label: str,
                markers: bool) -> None:
    """Front zero joined left to right (``front`` is ordered by objective 2); without
    ``markers`` only the line is drawn, so the teams keep their own markers."""
    if not front:
        return
    x, y = _xy(history, front)
    ax.plot(x, y, drawstyle="steps-post", color=_FRONT_EDGE, linewidth=1.8, zorder=2,
            label=None if markers else label)
    if markers:
        ax.scatter(x, y, s=style.marker_s, c=_FRONT_FACE, edgecolors=_FRONT_EDGE,
                   linewidths=1.5, zorder=4, label=label)
    return


def _candidate_front(ax, history: RunHistory, frame: GenerationFrame, style: _Style,
                     markers: bool) -> None:
    """The candidates' front zero, as candidates.gif, offspring.gif and front_zero.gif
    all draw it."""
    _front_line(ax, history, frame.front, style, f"front zero (n={len(frame.front)})", markers)
    # Front zero larger than the population is split by crowding distance; the teams
    # that lost are still non-dominated, so they stay on the line, marked: drawn hollow
    # in place of the front marker, or ringed around their own marker.
    cut = [i for i in frame.front if i not in set(frame.survivors)]
    if not cut:
        return
    label = f"cut by crowding (n={len(cut)})"
    if markers:
        ax.scatter(*_xy(history, cut), s=style.marker_s, c="white", edgecolors=_FRONT_EDGE,
                   linewidths=1.5, zorder=5, label=label)
    else:
        ax.scatter(*_xy(history, cut), s=2.2 * style.marker_s, facecolors="none",
                   edgecolors=_FRONT_EDGE, linewidths=1.5, zorder=5, label=label)
    return


def _label_candidates(ax, history: RunHistory, frame: GenerationFrame, limits: Axes,
                      style: _Style) -> None:
    """Labels for every candidate, the id colored by front zero, failed or neither."""
    front = set(frame.front)
    failed = [i for i in frame.candidates if history.teams[i].failed]
    _label_points(ax, history, [i for i in frame.candidates if i in front], limits, style,
                  _FRONT_ID)
    _label_points(ax, history, failed, limits, style, _FAILED_ID)
    _label_points(ax, history, [i for i in frame.candidates
                                if i not in front and i not in set(failed)],
                  limits, style, _ID)
    return


def _label_points(ax, history: RunHistory, ids: List[int], limits: Axes, style: _Style,
                  id_color: str) -> None:
    """As in all_front.png: the archive id inside each marker, performance and
    objective 2 just below it (a failure's at its pinned point)."""
    if not ids:
        return
    xs, ys = _pinned_xy(history, ids, limits)
    # Start below the marker's edge: its radius in points is sqrt(area / pi).
    dy = style.label_dy - float(np.sqrt(style.marker_s / np.pi))
    for i, x, y in zip(ids, xs, ys):
        team = history.teams[i]
        ax.annotate(str(i), (x, y), ha="center", va="center", fontsize=style.id_fs,
                    fontweight="bold", color=id_color, zorder=6)
        ax.annotate(f"{team.performance:.4f}\n"
                    f"{format_objective_2(team.objective2, history.objective_2, _COST_DECIMALS)}",
                    (x, y), textcoords="offset points", xytext=(0, dy), ha="center",
                    va="top", fontsize=style.label_fs, color="0.35", zorder=6)
    return


def _x_limits(values: List[float], objective_2: str) -> Tuple[float, float]:
    """Padded range of ``values``: linear for cost, logarithmic for token count."""
    assert values, "Nothing to plot."
    if objective_2 == OBJECTIVE_COST_USD:
        lo, hi = min(values), max(values)
        pad = 0.06 * (hi - lo) if hi > lo else max(0.05 * abs(hi), 0.01)
        return lo - pad, hi + pad
    positive = [v for v in values if v > 0]
    assert positive, "A log axis needs at least one positive token count."
    lo, hi = min(positive), max(positive)
    # Same 6% padding, measured in decades; a single value gets a quarter decade.
    pad = 0.06 * np.log10(hi / lo) if hi > lo else 0.25
    return float(lo / 10 ** pad), float(hi * 10 ** pad)


def _y_limits(values: List[float]) -> Tuple[float, float]:
    """Performance range rounded out to the nearest 0.05, with room below the lowest
    marker for its label; the top stays at or under 1.03."""
    assert values, "Nothing to plot."
    lo = np.floor(round(min(values) / 0.05, 6)) * 0.05
    hi = np.ceil(round(max(values) / 0.05, 6)) * 0.05
    if hi - lo < 0.1:
        lo, hi = lo - 0.05, hi + 0.05
    return float(lo - 0.12 * (hi - lo)), float(min(1.03, hi + 0.03))


def _style_axes(ax, objective_2: str, limits: Axes) -> None:
    (x_lo, x_hi), (y_lo, y_hi) = limits
    if objective_2 == OBJECTIVE_COST_USD:
        ax.set_xscale("linear")
    else:
        ax.set_xscale("log")
        ax.xaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 3.0, 5.0)))
        ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)
    ax.xaxis.set_major_formatter(FuncFormatter(
        lambda v, _: format_objective_2(v, objective_2, _COST_DECIMALS)))
    if y_lo < PERFORMANCE_NO_SKILL < y_hi:
        # Below the no-skill line the ranking is worse than a coin flip.
        ax.axhline(PERFORMANCE_NO_SKILL, color="0.75", linestyle="-.", linewidth=1.0,
                   zorder=0, label=f"no skill ({PERFORMANCE_NO_SKILL:.2f})")
    ax.set_xlabel(f"{OBJECTIVE_2_LABELS[objective_2]}  (minimize)", fontsize=11)
    ax.set_ylabel(f"{PERFORMANCE_LABEL}  (maximize)", fontsize=11)
    ax.grid(alpha=0.3, linestyle=":")
    if ax.get_legend() is None:
        ax.legend(loc="lower right", fontsize=9, framealpha=0.9)
    return
