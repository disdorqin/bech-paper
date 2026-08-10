"""Shared publication style for Route-E paper figures.

Usage:
    from paper_plot_style import *
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ...
    save_fig(fig, "fig2_main_mae")
"""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

try:
    import scienceplots  # noqa: F401
    plt.style.use(["science", "no-latex", "grid"])
except Exception:
    pass

FIG_DIR = Path(__file__).resolve().parent
DPI = 300
FORMAT = "pdf"
FONT_SIZE = 10

mpl.rcParams.update(
    {
        "font.size": FONT_SIZE,
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "axes.labelsize": FONT_SIZE,
        "axes.titlesize": FONT_SIZE + 1,
        "xtick.labelsize": FONT_SIZE - 1,
        "ytick.labelsize": FONT_SIZE - 1,
        "legend.fontsize": FONT_SIZE - 1,
        "figure.dpi": DPI,
        "savefig.dpi": DPI,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "text.usetex": False,
        "mathtext.fontset": "stix",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

# colorblind-friendly (Wong 2011 + tab10 fallback)
COLORS = [
    "#0072B2",  # blue
    "#E69F00",  # orange
    "#009E73",  # green
    "#CC79A7",  # purple
    "#D55E00",  # vermillion
    "#56B4E9",  # sky
    "#F0E442",  # yellow
    "#000000",  # black
]


def save_fig(fig: plt.Figure, name: str, fmt: str = FORMAT) -> Path:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / f"{name}.{fmt}"
    fig.savefig(out)
    # also png preview for quick look
    if fmt != "png":
        fig.savefig(FIG_DIR / f"{name}.png")
    print(f"Saved: {out}")
    return out


def bar_with_labels(ax, labels, values, ylabel: str, colors=None):
    colors = colors or [COLORS[i % len(COLORS)] for i in range(len(labels))]
    x = np.arange(len(labels))
    bars = ax.bar(x, values, color=colors, edgecolor="black", linewidth=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel(ylabel)
    for b, v in zip(bars, values):
        ax.text(
            b.get_x() + b.get_width() / 2,
            b.get_height(),
            f"{v:.2f}",
            ha="center",
            va="bottom",
            fontsize=FONT_SIZE - 1,
        )
    return bars
