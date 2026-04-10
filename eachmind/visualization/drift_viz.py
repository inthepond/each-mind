"""Drift visualization — matplotlib-based charts for perspective divergence.

Requires matplotlib: pip install eachmind[viz]
"""
from __future__ import annotations

from itertools import combinations

try:
    import matplotlib
    matplotlib.use("Agg")  # Non-interactive backend for saving
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
except ImportError as e:
    raise ImportError(
        "Visualization requires matplotlib. "
        "Install with: pip install eachmind[viz]"
    ) from e

from eachmind.primitives.drift import Drift
from eachmind.primitives.perspective import Perspective

# ---------------------------------------------------------------------------
# Style helper
# ---------------------------------------------------------------------------

_STYLE = "seaborn-v0_8-whitegrid"


def _pair_label(a: str, b: str) -> str:
    """Consistent label for an agent pair (alphabetical order)."""
    return f"{min(a, b)} vs {max(a, b)}"


# ---------------------------------------------------------------------------
# 1. Heatmap
# ---------------------------------------------------------------------------


def plot_drift_heatmap(perspectives: dict[str, Perspective]) -> Figure:
    """Pairwise drift matrix rendered as a colour heatmap.

    Args:
        perspectives: Mapping of agent name to Perspective.

    Returns:
        A matplotlib Figure (caller is responsible for saving/closing).
    """
    names = sorted(perspectives.keys())
    n = len(names)

    # Build the matrix
    matrix = [[0.0] * n for _ in range(n)]
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            if i == j:
                matrix[i][j] = 0.0
            else:
                matrix[i][j] = perspectives[a].drift_from(perspectives[b])

    with plt.style.context(_STYLE):
        fig, ax = plt.subplots(figsize=(max(6, n + 2), max(5, n + 1)))

        cmap = plt.get_cmap("RdYlGn_r")
        norm = mcolors.Normalize(vmin=0.0, vmax=1.0)

        im = ax.imshow(matrix, cmap=cmap, norm=norm, aspect="equal")

        # Ticks
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(names, fontsize=11, rotation=45, ha="right")
        ax.set_yticklabels(names, fontsize=11)

        # Annotate cells
        for i in range(n):
            for j in range(n):
                val = matrix[i][j]
                text_color = "white" if val > 0.55 else "black"
                ax.text(
                    j, i, f"{val:.2f}",
                    ha="center", va="center",
                    fontsize=10, fontweight="bold",
                    color=text_color,
                )

        # Colorbar
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Drift (0 = identical, 1 = max divergence)", fontsize=10)

        ax.set_title("Perspective Drift Heatmap", fontsize=14, fontweight="bold", pad=12)
        fig.tight_layout()

    return fig


# ---------------------------------------------------------------------------
# 2. Timeline
# ---------------------------------------------------------------------------


def plot_drift_timeline(drift_tracker: Drift) -> Figure:
    """Line chart of drift measurements over time.

    Each unique agent pair gets its own line with a distinct colour.

    Args:
        drift_tracker: A Drift instance containing measurement history.

    Returns:
        A matplotlib Figure.
    """
    if not drift_tracker.measurements:
        with plt.style.context(_STYLE):
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.set_title("Perspective Drift Over Time", fontsize=14, fontweight="bold")
            ax.text(0.5, 0.5, "No measurements yet", transform=ax.transAxes,
                    ha="center", va="center", fontsize=13, color="grey")
            fig.tight_layout()
        return fig

    # Group measurements by pair
    pair_series: dict[str, list[float]] = {}
    for m in drift_tracker.measurements:
        label = _pair_label(m.agent_a, m.agent_b)
        pair_series.setdefault(label, []).append(m.drift_value)

    with plt.style.context(_STYLE):
        fig, ax = plt.subplots(figsize=(10, 5))

        colors = plt.get_cmap("tab10")
        for idx, (label, values) in enumerate(sorted(pair_series.items())):
            color = colors(idx % 10)
            xs = list(range(1, len(values) + 1))
            ax.plot(
                xs, values,
                marker="o", markersize=5,
                linewidth=2, label=label,
                color=color, alpha=0.85,
            )

        ax.set_xlabel("Measurement #", fontsize=11)
        ax.set_ylabel("Drift Value", fontsize=11)
        ax.set_ylim(-0.02, 1.02)
        ax.set_title("Perspective Drift Over Time", fontsize=14, fontweight="bold", pad=12)
        ax.legend(loc="best", fontsize=9, framealpha=0.9)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

    return fig


# ---------------------------------------------------------------------------
# 3. Team Diversity
# ---------------------------------------------------------------------------


def plot_team_diversity(perspectives: dict[str, Perspective]) -> Figure:
    """Comprehensive team cognitive diversity overview.

    Top subplot: horizontal bar chart of pairwise drift.
    Bottom subplot: summary stats with a gauge-like diversity indicator.

    Args:
        perspectives: Mapping of agent name to Perspective.

    Returns:
        A matplotlib Figure.
    """
    names = sorted(perspectives.keys())

    # Compute pairwise drift
    pairs: list[tuple[str, float]] = []
    for a, b in combinations(names, 2):
        drift_val = perspectives[a].drift_from(perspectives[b])
        pairs.append((_pair_label(a, b), drift_val))
    pairs.sort(key=lambda p: p[1], reverse=True)

    if not pairs:
        with plt.style.context(_STYLE):
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.text(0.5, 0.5, "Need at least 2 agents", transform=ax.transAxes,
                    ha="center", va="center", fontsize=13, color="grey")
            fig.tight_layout()
        return fig

    labels = [p[0] for p in pairs]
    values = [p[1] for p in pairs]
    avg_diversity = sum(values) / len(values)
    min_pair = pairs[-1]
    max_pair = pairs[0]

    # Color mapping: low drift = cool (blue), high drift = warm (red/orange)
    cmap = plt.get_cmap("RdYlGn_r")
    bar_colors = [cmap(v) for v in values]

    with plt.style.context(_STYLE):
        fig, (ax_bars, ax_gauge) = plt.subplots(
            2, 1, figsize=(10, max(6, len(pairs) * 0.8 + 4)),
            gridspec_kw={"height_ratios": [3, 2]},
        )

        # --- Top: horizontal bar chart ---
        y_pos = list(range(len(labels)))
        ax_bars.barh(y_pos, values, color=bar_colors, edgecolor="white", height=0.65)
        ax_bars.set_yticks(y_pos)
        ax_bars.set_yticklabels(labels, fontsize=10)
        ax_bars.set_xlim(0, 1.0)
        ax_bars.set_xlabel("Drift Value", fontsize=10)
        ax_bars.set_title("Pairwise Perspective Drift", fontsize=12, fontweight="bold", pad=8)
        ax_bars.invert_yaxis()

        # Annotate bar values
        for i, v in enumerate(values):
            ax_bars.text(v + 0.015, i, f"{v:.3f}", va="center", fontsize=9, fontweight="bold")

        # --- Bottom: summary stats + gauge ---
        ax_gauge.set_xlim(0, 10)
        ax_gauge.set_ylim(0, 5)
        ax_gauge.axis("off")

        # Diversity gauge bar
        gauge_y = 3.2
        gauge_height = 0.7
        # Background bar (grey)
        ax_gauge.barh(
            gauge_y, 10, left=0, height=gauge_height,
            color="#e0e0e0", edgecolor="none",
        )
        # Filled portion — colour from green (low) to red (high)
        gauge_color = cmap(avg_diversity)
        ax_gauge.barh(
            gauge_y, avg_diversity * 10, left=0, height=gauge_height,
            color=gauge_color, edgecolor="none",
        )
        # Gauge label
        ax_gauge.text(
            avg_diversity * 10, gauge_y, f" {avg_diversity:.3f}",
            va="center", ha="left", fontsize=12, fontweight="bold",
        )

        # Gauge end labels
        ax_gauge.text(0, gauge_y - 0.6, "Groupthink\n(0.0)", ha="center", fontsize=8, color="grey")
        ax_gauge.text(10, gauge_y - 0.6, "Fragmented\n(1.0)", ha="center", fontsize=8, color="grey")
        ax_gauge.text(5, gauge_y + 0.7, "Team Cognitive Diversity Score", ha="center",
                      fontsize=12, fontweight="bold")

        # Summary text
        summary_lines = [
            f"Average diversity:  {avg_diversity:.3f}",
            f"Most similar pair:  {min_pair[0]}  (drift = {min_pair[1]:.3f})",
            f"Most divergent pair:  {max_pair[0]}  (drift = {max_pair[1]:.3f})",
            f"Number of agents:  {len(names)}   |   Pairs measured:  {len(pairs)}",
        ]
        for i, line in enumerate(summary_lines):
            ax_gauge.text(
                5, 1.6 - i * 0.45, line,
                ha="center", va="center", fontsize=10,
                fontfamily="monospace",
            )

        fig.suptitle("Team Cognitive Diversity", fontsize=15, fontweight="bold", y=0.98)
        fig.tight_layout(rect=[0, 0, 1, 0.96])

    return fig
