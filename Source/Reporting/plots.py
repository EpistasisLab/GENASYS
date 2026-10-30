"""Front plots: one labelled marker per team in objective space.

Drawing only. The EA decides which teams are on the front, which failed, which ids they
carry and where the file goes; this module receives that ready to draw and computes
nothing about the teams itself. The only rounding here is for display: axis ticks and
the labels under each marker. Points are plotted at their raw values.

Matplotlib runs on the headless Agg backend because compute nodes have no display.
Errors are not caught here; the EA catches them at its call sites, so a broken plot
never ends a run.
"""

import os
from typing import List, Optional

import matplotlib
matplotlib.use("Agg")  # must precede the pyplot import
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import numpy.typing as npt  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402
from typeguard import typechecked  # noqa: E402

from ..Optimizer.objectives import (OBJECTIVE_2_LABELS, PERFORMANCE_LABEL,  # noqa: E402
                                    PERFORMANCE_NO_SKILL, format_objective_2)

# Decimal places for cost on the axis ticks and the marker labels.
_COST_DECIMALS = 2


@typechecked
def plot_front(objective2: npt.NDArray[np.float64],
               performances: npt.NDArray[np.float64],
               ids: List[int],
               front: npt.NDArray[np.int64],
               dominated: npt.NDArray[np.int64],
               failed: npt.NDArray[np.int64],
               objective_2: str,
               title: str,
               subtitle: str,
               out_path: str,
               previous_front: Optional[npt.NDArray[np.float64]] = None,
               previous_front_label: str = "") -> None:
    """Write one front plot to ``out_path``.

    Args:
        objective2, performances: each team's raw values (x and y), by position.
        ids: each team's archive id, drawn inside its marker.
        front: positions on the non-dominated front, ordered by objective 2 so
            the step line joining them runs left to right.
        dominated, failed: the remaining positions. ``front``, ``dominated`` and
            ``failed`` together cover every position exactly once.
        objective_2: which objective 2 is plotted; sets the x label and how
            values are printed.
        previous_front: optional ``(k, 2)`` array of ``[objective2, performance]``
            drawn as a dashed step line, to show how the front moved.
        previous_front_label: legend entry for ``previous_front``.
    """
    n = performances.size
    assert n > 0, "Nothing to plot."
    assert objective2.size == n and len(ids) == n, (
        f"objective2 ({objective2.size}), performances ({n}) and ids ({len(ids)}) "
        f"must be the same length.")
    covered = np.sort(np.concatenate([front, dominated, failed]))
    assert covered.size == n and (covered == np.arange(n)).all(), (
        "front, dominated and failed must cover every position exactly once.")

    # Scale the canvas and the marks to the crowd: a 10-point plot can afford big
    # numbered markers, a 45-point one cannot without turning into a blob.
    figsize = (min(20.0, 9.0 + 0.14 * n), min(13.0, 7.0 + 0.07 * n))
    if n <= 12:
        marker_s, id_fs, lab_fs, lab_dy = 300, 7.0, 6.5, -22
    elif n <= 25:
        marker_s, id_fs, lab_fs, lab_dy = 170, 6.0, 5.5, -17
    else:
        marker_s, id_fs, lab_fs, lab_dy = 110, 5.0, 4.5, -13

    fig, ax = plt.subplots(figsize=figsize)
    try:
        _draw(ax, objective2, performances, ids, front, dominated, failed, objective_2,
              marker_s, id_fs, lab_fs, lab_dy, previous_front, previous_front_label)
        ax.set_title(f"{title}\n{subtitle}" if subtitle else title, fontsize=11)
        fig.tight_layout()
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        fig.savefig(out_path, dpi=150)
    finally:
        # Closed even when drawing fails, so a long run does not leak figures.
        plt.close(fig)
    return


def _draw(ax, objective2: npt.NDArray[np.float64], performances: npt.NDArray[np.float64],
          ids: List[int], front: npt.NDArray[np.int64], dominated: npt.NDArray[np.int64],
          failed: npt.NDArray[np.int64], objective_2: str, marker_s: float, id_fs: float,
          lab_fs: float, lab_dy: float, previous_front: Optional[npt.NDArray[np.float64]],
          previous_front_label: str) -> None:
    """Markers, labels and axes for ``plot_front``."""
    n = performances.size

    if previous_front is not None and len(previous_front):
        assert previous_front_label, "previous_front needs a legend label."
        ax.plot(previous_front[:, 0], previous_front[:, 1], drawstyle="steps-post",
                color="0.70", linestyle="--", linewidth=1.3, zorder=1,
                label=previous_front_label)

    # Front joined left to right, drawn under the markers.
    if front.size:
        ax.plot(objective2[front], performances[front], drawstyle="steps-post",
                color="#c0392b", linewidth=1.6, zorder=2)
        ax.scatter(objective2[front], performances[front], s=marker_s, c="#f5b7b1",
                   edgecolors="#c0392b", linewidths=1.4, zorder=4,
                   label=f"front (n={front.size})")
    if dominated.size:
        ax.scatter(objective2[dominated], performances[dominated], s=marker_s,
                   c="#d6eaf8", edgecolors="#5d6d7e", linewidths=1.0, zorder=3,
                   label=f"dominated (n={dominated.size})")
    if failed.size:
        ax.scatter(objective2[failed], performances[failed], s=marker_s, c="#eaecee",
                   marker="X", edgecolors="#7f8c8d", linewidths=0.8, zorder=3,
                   label=f"failed, off front (n={failed.size})")

    # Id inside every marker; performance and objective 2 printed just below it.
    id_colors = {**{int(i): "#2c3e50" for i in dominated},
                 **{int(i): "#7b241c" for i in front},
                 **{int(i): "#7f8c8d" for i in failed}}
    for i in range(n):
        x, y = float(objective2[i]), float(performances[i])
        ax.annotate(str(ids[i]), (x, y), ha="center", va="center", fontsize=id_fs,
                    fontweight="bold", color=id_colors[i], zorder=6)
        ax.annotate(f"{y:.4f}\n{format_objective_2(x, objective_2, _COST_DECIMALS)}",
                    (x, y), textcoords="offset points", xytext=(0, lab_dy),
                    ha="center", va="top", fontsize=lab_fs, color="0.35", zorder=6)

    # Below the no-skill line the ranking is worse than a coin flip.
    ax.axhline(PERFORMANCE_NO_SKILL, color="0.75", linestyle="-.", linewidth=1.0,
               zorder=0, label=f"no skill ({PERFORMANCE_NO_SKILL:.2f})")

    ax.set_xlabel(f"{OBJECTIVE_2_LABELS[objective_2]}  (minimize)")
    ax.set_ylabel(f"{PERFORMANCE_LABEL}  (maximize)")
    ax.xaxis.set_major_formatter(FuncFormatter(
        lambda v, _: format_objective_2(v, objective_2, _COST_DECIMALS)))

    # The y-axis tops out at the next 0.1 above the best score in this plot, so the
    # points fill the frame (medical-data ROC-AUCs sit well below 1.0). At least 0.1,
    # so an all-failed plot still has a positive top; round() before ceil() clears
    # float noise (0.8 / 0.1 = 7.999...).
    y_top = max(0.1, round(float(np.ceil(round(float(performances.max()) / 0.1, 6))) * 0.1, 1))
    # Headroom below 0 keeps the labels under failed markers (at 0.0) inside the frame.
    ax.set_ylim(-0.12, y_top)
    ax.set_yticks(np.round(np.arange(0.0, y_top + 1e-9, 0.1), 1))

    ax.grid(alpha=0.3, linestyle=":")
    ax.legend(loc="lower right", fontsize=9, framealpha=0.9)
    return
