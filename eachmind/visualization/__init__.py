"""Visualization tools for eachmind.

Requires matplotlib: pip install eachmind[viz]
"""
from eachmind.visualization.drift_viz import (
    plot_drift_heatmap,
    plot_drift_timeline,
    plot_team_diversity,
)

__all__ = [
    "plot_drift_heatmap",
    "plot_drift_timeline",
    "plot_team_diversity",
]
