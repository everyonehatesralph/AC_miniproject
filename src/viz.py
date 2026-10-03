"""
viz.py -- one plotting style for every figure in the project.

Colours are the validated reference palette of the dataviz guidelines: categorical
slots in fixed order (the first three pass colour-vision-deficiency checks for all
pairs, so class-coloured charts use at most three hues), a single-hue blue ramp for
magnitudes, and a blue<->red diverging map with a neutral grey midpoint.
Rules kept here: one y-axis per panel (no twin axes), hairline solid grids,
text in ink colours (never the series colour), a legend whenever >= 2 series.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
SEQUENTIAL = LinearSegmentedColormap.from_list("seq_blue", ["#f4f8fd"] + BLUE_RAMP)
DIVERGING = LinearSegmentedColormap.from_list("div_blue_red", ["#104281", "#3987e5", "#9ec5f4", "#f0efec",
                                                               "#f3a3a2", "#e34948", "#a32a29"])

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "sans-serif", "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"],
    "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "semibold", "axes.titlelocation": "left",
    "axes.labelsize": 9, "axes.labelcolor": INK_2, "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "axes.axisbelow": True,
    "grid.color": GRID, "grid.linewidth": 0.8, "grid.linestyle": "-",
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK_2, "ytick.labelcolor": INK_2,
    "text.color": INK, "legend.frameon": False, "legend.fontsize": 8, "lines.linewidth": 2,
    "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round", "figure.dpi": 110,
})


def save(fig, path: Path, dpi: int = 150) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", metadata={"Software": None})
    plt.close(fig)


def bar_labels(ax, bars, fmt="{:.0f}", horizontal=False) -> None:
    """Value at the bar tip, in secondary ink."""
    for b in bars:
        v = b.get_width() if horizontal else b.get_height()
        if horizontal:
            ax.text(b.get_x() + v, b.get_y() + b.get_height() / 2, " " + fmt.format(v), va="center",
                    ha="left", color=INK_2, fontsize=7.5)
        else:
            ax.text(b.get_x() + b.get_width() / 2, v, fmt.format(v), va="bottom", ha="center",
                    color=INK_2, fontsize=7.5)


def ink_for(rgba) -> str:
    """White or ink text inside a filled cell, chosen by the fill's luminance."""
    r, g, b = rgba[:3]
    return "white" if (0.2126 * r + 0.7152 * g + 0.0722 * b) < 0.5 else INK
