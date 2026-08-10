"""Plot all comparative experiment results: Exp1 Shandong, Exp2 LTSF, Exp3 NEM-SA1.

Reads JSON results and generates publication-quality PNG figures.
"""
from __future__ import annotations

import json, os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.patches as mpatches

HERE = Path(__file__).resolve().parent          # src/
ROOT = HERE.parent                               # 01-comparative/
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

# ── style ──
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
    "axes.unicode_minus": False,
    "figure.dpi": 150,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# ── color palette ──
COLORS = {
    "Base": "#95a5a6",
    "HCH": "#e74c3c",
    "QuantileCorrection": "#3498db",
    "VahediStyle": "#2ecc71",
    "CRC": "#f39c12",
    "PIR": "#9b59b6",
    "SpikeRegularization": "#e67e22",
}
LABEL_MAP = {
    "Base": "Base",
    "HCH": "HCH (Ours)",
    "QuantileCorrection": "Quantile",
    "VahediStyle": "Vahedi",
    "CRC": "CRC",
    "PIR": "PIR",
    "SpikeRegularization": "SpikeReg",
}
BACKBONE_COLORS = {
    "Linear": "#3498db",
    "MLP": "#e67e22",
    "LSTM": "#2ecc71",
    "Transformer": "#9b59b6",
    "GBDT": "#e74c3c",
}

METHOD_ORDER = ["Base", "HCH", "QuantileCorrection", "VahediStyle", "CRC", "PIR", "SpikeRegularization"]
METHOD_ORDER_EXP2 = ["Base", "HCH", "QuantileCorrection", "CRC", "PIR"]
METHOD_ORDER_EXP3 = ["Base", "HCH", "VahediStyle", "SpikeRegularization"]


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_method_order(experiment: str):
    if experiment == "exp1":
        return METHOD_ORDER
    elif experiment == "exp2":
        return METHOD_ORDER_EXP2
    else:
        return METHOD_ORDER_EXP3


# ═══════════════════════════════════════════════════════════
# Fig 1: Exp1 Shandong — 5 backbone comparison (MAE, NegMiss, EpRecall)
# ═══════════════════════════════════════════════════════════
def plot_exp1_shandong():
    data = load_json(RESULTS / "exp1_shandong.json")
    backbones = list(data.keys())
    methods = METHOD_ORDER

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Exp1: Shandong — 5 Backbone × 7 Method Comparison", fontsize=14, fontweight="bold", y=1.02)

    for ax_idx, (metric, title, fmt) in enumerate([
        ("mae", "MAE ↓", "{:.1f}"),
        ("neg_miss_rate", "Negative Price Miss Rate ↓", "{:.1%}"),
        ("ep_our_episode_recall", "Episode Recall ↑", "{:.1%}"),
    ]):
        ax = axes[ax_idx]
        x = np.arange(len(backbones))
        width = 0.12
        n_methods = len(methods)
        offset_start = -(n_methods - 1) * width / 2

        for mi, method in enumerate(methods):
            vals = []
            for bb in backbones:
                m = data[bb].get(method, {})
                val = m.get(metric, None)
                if val is None:
                    val = 0
                vals.append(val)
            positions = x + offset_start + mi * width
            bars = ax.bar(positions, vals, width,
                         label=LABEL_MAP.get(method, method),
                         color=COLORS.get(method, "#cccccc"),
                         alpha=0.85,
                         edgecolor="white",
                         linewidth=0.5)

            # Highlight HCH bars
            if method == "HCH":
                for bar, val in zip(bars, vals):
                    bar.set_edgecolor("black")
                    bar.set_linewidth(2)

        ax.set_xticks(x)
        ax.set_xticklabels(backbones, fontsize=10)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_ylabel(metric, fontsize=9)

    # Single legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=7, fontsize=8,
              bbox_to_anchor=(0.5, -0.06), frameon=True)

    plt.tight_layout()
    fig.savefig(FIGURES / "exp1_shandong_comparison.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("  ✓ exp1_shandong_comparison.png")


# ═══════════════════════════════════════════════════════════
# Fig 2: Exp1 — HCH vs Base improvement per backbone
# ═══════════════════════════════════════════════════════════
def plot_exp1_hch_improvement():
    data = load_json(RESULTS / "exp1_shandong.json")
    backbones = list(data.keys())

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle("Exp1: HCH Improvement over Base (Shandong)", fontsize=14, fontweight="bold")

    metrics_config = [
        ("mae", "MAE Improvement (%)", True),
        ("neg_miss_rate", "Neg Miss Rate Reduction (pp)", True),
        ("ep_our_episode_recall", "Episode Recall Gain (pp)", False),
    ]

    for ax_idx, (metric, title, is_reduction) in enumerate(metrics_config):
        ax = axes[0, ax_idx]
        hch_vals = []
        base_vals = []
        for bb in backbones:
            hch = data[bb]["HCH"].get(metric, 0)
            base = data[bb]["Base"].get(metric, 0)
            hch_vals.append(hch)
            base_vals.append(base)

        x = np.arange(len(backbones))
        width = 0.3
        bars1 = ax.bar(x - width/2, base_vals, width, label="Base", color="#95a5a6", edgecolor="white")
        bars2 = ax.bar(x + width/2, hch_vals, width, label="HCH", color="#e74c3c", edgecolor="white", linewidth=1.5)

        ax.set_xticks(x)
        ax.set_xticklabels(backbones, fontsize=9)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.legend(fontsize=8)

        # Add value labels on HCH bars
        for bar in bars2:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., h + max(hch_vals)*0.01,
                   f"{h:.2f}" if metric == "mae" else f"{h:.1%}" if "rate" in metric or "recall" in metric else f"{h:.2f}",
                   ha="center", va="bottom", fontsize=7, color="#e74c3c", fontweight="bold")

    # Bottom row: improvement percentage
    for ax_idx, (metric, title, is_reduction) in enumerate(metrics_config):
        ax = axes[1, ax_idx]
        improvements = []
        for bb in backbones:
            hch = data[bb]["HCH"].get(metric, 0)
            base = data[bb]["Base"].get(metric, 0)
            if base == 0:
                imp = 0
            elif metric in ("mae", "normal_harm"):
                imp = (base - hch) / base * 100
            elif metric == "neg_miss_rate":
                imp = (base - hch) * 100  # percentage points
            elif metric == "ep_our_episode_recall":
                imp = (hch - base) * 100  # percentage points (higher is better)
            else:
                imp = (base - hch) / base * 100
            improvements.append(imp)

        colors = ["#2ecc71" if v > 0 else "#e74c3c" for v in improvements]
        bars = ax.barh(backbones, improvements, color=colors, edgecolor="white", height=0.6)
        ax.axvline(x=0, color="black", linewidth=0.8)
        ax.set_xlabel("Improvement" if ax_idx == 0 else "", fontsize=9)
        title_suffix = "(pp)" if "Rate" in title or "Recall" in title else "(%)"
        ax.set_title(title + " " + title_suffix, fontsize=10, fontweight="bold")

        for bar, val in zip(bars, improvements):
            ax.text(val + (1 if val >= 0 else -1), bar.get_y() + bar.get_height()/2.,
                   f"{val:+.1f}", va="center", fontsize=8, fontweight="bold")

    plt.tight_layout()
    fig.savefig(FIGURES / "exp1_hch_improvement.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("  ✓ exp1_hch_improvement.png")


# ═══════════════════════════════════════════════════════════
# Fig 3: Exp2 LTSF — Cross-dataset comparison
# ═══════════════════════════════════════════════════════════
def plot_exp2_ltsf():
    data = load_json(RESULTS / "exp2_ltsf.json")
    datasets = list(data.keys())
    backbones = ["Linear", "GBDT"]
    methods = METHOD_ORDER_EXP2

    fig, axes = plt.subplots(len(datasets), len(backbones), figsize=(12, 8),
                             sharey="row")
    fig.suptitle("Exp2: LTSF — Low/Zero Negative Price Markets", fontsize=14, fontweight="bold")

    for row, ds in enumerate(datasets):
        for col, bb in enumerate(backbones):
            ax = axes[row][col] if len(datasets) > 1 else axes[col]
            vals = []
            colors = []
            for m in methods:
                v = data[ds][bb].get(m, {}).get("mae", 0)
                vals.append(v)
                colors.append(COLORS.get(m, "#cccccc"))

            x = np.arange(len(methods))
            bars = ax.bar(x, vals, color=colors, edgecolor="white", linewidth=0.8)

            # Highlight HCH
            hch_idx = methods.index("HCH")
            bars[hch_idx].set_edgecolor("black")
            bars[hch_idx].set_linewidth(2)

            ax.set_xticks(x)
            ax.set_xticklabels([LABEL_MAP.get(m, m) for m in methods],
                              fontsize=8, rotation=25, ha="right")
            ax.set_ylabel("MAE" if col == 0 else "", fontsize=9)
            neg_info = f" ({data[ds][bb]['Base'].get('neg_n', 0)} neg pts)"
            ax.set_title(f"{ds} — {bb}{neg_info}", fontsize=10, fontweight="bold")

            # Add value labels
            for bar, val in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + max(vals)*0.01,
                       f"{val:.2f}", ha="center", va="bottom", fontsize=7)

    plt.tight_layout()
    fig.savefig(FIGURES / "exp2_ltsf_comparison.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("  ✓ exp2_ltsf_comparison.png")


# ═══════════════════════════════════════════════════════════
# Fig 4: Exp3 NEM-SA1 — Negative/Spike comparison
# ═══════════════════════════════════════════════════════════
def plot_exp3_nem():
    data = load_json(RESULTS / "exp3_nem_spike.json")
    backbones = list(data.keys())
    methods = METHOD_ORDER_EXP3

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Exp3: NEM SA1 — High Negative Price Market (26% neg)", fontsize=14, fontweight="bold")

    metrics_config = [
        ("mae", "MAE ↓", "{:.1f}"),
        ("neg_miss_rate", "Neg Miss Rate ↓", "{:.1%}"),
        ("ep_our_episode_recall", "Episode Recall ↑", "{:.1%}"),
        ("normal_harm", "Normal Harm (lower=safer)", "{:.1f}"),
    ]

    for ax_idx, (metric, title, fmt) in enumerate(metrics_config):
        ax = axes[ax_idx // 2][ax_idx % 2]
        x = np.arange(len(backbones))
        width = 0.18
        n = len(methods)
        offset_start = -(n - 1) * width / 2

        for mi, method in enumerate(methods):
            vals = []
            for bb in backbones:
                vals.append(data[bb][method].get(metric, 0))
            positions = x + offset_start + mi * width
            bars = ax.bar(positions, vals, width,
                         label=LABEL_MAP.get(method, method),
                         color=COLORS.get(method, "#cccccc"),
                         edgecolor="white", alpha=0.85)

            if method == "HCH":
                for bar in bars:
                    bar.set_edgecolor("black")
                    bar.set_linewidth(2)

        ax.set_xticks(x)
        ax.set_xticklabels(backbones, fontsize=10)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.axhline(y=0, color="black", linewidth=0.8, alpha=0.5)

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, fontsize=9,
              bbox_to_anchor=(0.5, -0.02), frameon=True)

    plt.tight_layout()
    fig.savefig(FIGURES / "exp3_nem_comparison.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("  ✓ exp3_nem_comparison.png")


# ═══════════════════════════════════════════════════════════
# Fig 5: Cross-experiment — HCH safety vs effectiveness
# ═══════════════════════════════════════════════════════════
def plot_cross_safety_effectiveness():
    """Pareto plot: NDR (safety) vs MAE improvement (effectiveness) for all experiments."""
    exp1 = load_json(RESULTS / "exp1_shandong.json")
    exp2 = load_json(RESULTS / "exp2_ltsf.json")
    exp3 = load_json(RESULTS / "exp3_nem_spike.json")

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    fig.suptitle("Safety (NDR) vs Effectiveness (MAE Reduction) — All Experiments",
                fontsize=13, fontweight="bold")

    for ax_idx, (exp_name, exp_data, ds_list) in enumerate([
        ("Exp1 Shandong", exp1, list(exp1.keys())),
        ("Exp2 LTSF", exp2, list(exp2.keys())),
        ("Exp3 NEM-SA1", exp3, list(exp3.keys())),
    ]):
        ax = axes[ax_idx]
        methods = METHOD_ORDER if ax_idx == 0 else (METHOD_ORDER_EXP2 if ax_idx == 1 else METHOD_ORDER_EXP3)

        for bb_or_ds in ds_list:
            if exp_name.startswith("Exp2"):
                # Exp2: key is dataset, sub-key is backbone
                for bb in exp_data[bb_or_ds]:
                    for method in methods:
                        m = exp_data[bb_or_ds][bb].get(method, {})
                        if not m or method == "Base":
                            continue
                        ndr = m.get("ndr", 0)
                        base_mae = exp_data[bb_or_ds][bb]["Base"].get("mae", 1)
                        hch_mae = m.get("mae", base_mae)
                        improvement = (base_mae - hch_mae) / base_mae * 100
                        color = COLORS.get(method, "#cccccc")
                        marker = "s" if bb == "GBDT" else "o"
                        ax.scatter(ndr, improvement, c=color, marker=marker, s=80, alpha=0.7,
                                  edgecolors="white", linewidth=0.5, zorder=5)
            else:
                # Exp1/Exp3: key is backbone
                for method in methods:
                    m = exp_data[bb_or_ds].get(method, {})
                    if not m or method == "Base":
                        continue
                    ndr = m.get("ndr", 0)
                    base_mae = exp_data[bb_or_ds]["Base"].get("mae", 1)
                    hch_mae = m.get("mae", base_mae)
                    improvement = (base_mae - hch_mae) / base_mae * 100
                    color = COLORS.get(method, "#cccccc")
                    ax.scatter(ndr, improvement, c=color, marker="o", s=80, alpha=0.7,
                              edgecolors="white", linewidth=0.5, zorder=5)

        # HCH stars
        if exp_name == "Exp2":
            for ds in ds_list:
                for bb in exp_data[ds]:
                    hch_m = exp_data[ds][bb].get("HCH", {})
                    if hch_m:
                        ndr = hch_m.get("ndr", 0)
                        base_mae = exp_data[ds][bb]["Base"].get("mae", 1)
                        hch_mae_val = hch_m.get("mae", base_mae)
                        improvement = (base_mae - hch_mae_val) / base_mae * 100
                        ax.scatter(ndr, improvement, c=COLORS["HCH"], marker="*", s=250,
                                  edgecolors="black", linewidth=1.5, zorder=10)
        else:
            for bb in ds_list:
                hch_m = exp_data[bb].get("HCH", {})
                if hch_m:
                    ndr = hch_m.get("ndr", 0)
                    base_mae = exp_data[bb]["Base"].get("mae", 1)
                    hch_mae_val = hch_m.get("mae", base_mae)
                    improvement = (base_mae - hch_mae_val) / base_mae * 100
                    ax.scatter(ndr, improvement, c=COLORS["HCH"], marker="*", s=250,
                              edgecolors="black", linewidth=1.5, zorder=10)

        ax.axhline(y=0, color="gray", linewidth=0.8, linestyle="--", alpha=0.5)
        ax.axvline(x=80, color="gray", linewidth=0.8, linestyle="--", alpha=0.3)
        ax.set_xlabel("NDR (Non-Degradation Rate %)", fontsize=10)
        ax.set_title(exp_name, fontsize=11, fontweight="bold")

    axes[0].set_ylabel("MAE Improvement (%)", fontsize=10)

    # Legend
    legend_elements = []
    for m in METHOD_ORDER:
        legend_elements.append(
            mpatches.Patch(facecolor=COLORS[m], label=LABEL_MAP.get(m, m), alpha=0.85)
        )
    legend_elements.append(plt.scatter([], [], marker="*", c=COLORS["HCH"], s=250,
                                       edgecolors="black", linewidth=1.5, label="HCH (highlighted)"))
    fig.legend(handles=legend_elements, loc="upper center", ncol=4, fontsize=8,
              bbox_to_anchor=(0.5, -0.06), frameon=True)

    plt.tight_layout()
    fig.savefig(FIGURES / "cross_safety_effectiveness.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("  ✓ cross_safety_effectiveness.png")


# ═══════════════════════════════════════════════════════════
# Fig 6: Exp1 — Neg Miss Rate heatmap (backbone × method)
# ═══════════════════════════════════════════════════════════
def plot_heatmap_neg_miss():
    data = load_json(RESULTS / "exp1_shandong.json")
    backbones = list(data.keys())
    methods = METHOD_ORDER

    matrix = np.zeros((len(methods), len(backbones)))
    for mi, method in enumerate(methods):
        for bi, bb in enumerate(backbones):
            m = data[bb].get(method, {})
            matrix[mi, bi] = m.get("neg_miss_rate", 0) * 100 if m.get("neg_miss_rate") else np.nan

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(matrix, cmap="RdYlGn_r", aspect="auto", vmin=0, vmax=35)

    ax.set_xticks(range(len(backbones)))
    ax.set_xticklabels(backbones, fontsize=10)
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels([LABEL_MAP.get(m, m) for m in methods], fontsize=10)

    # Annotate cells
    for i in range(len(methods)):
        for j in range(len(backbones)):
            val = matrix[i, j]
            if not np.isnan(val):
                text_color = "white" if val > 20 else "black"
                # Bold for HCH row
                weight = "bold" if methods[i] == "HCH" else "normal"
                ax.text(j, i, f"{val:.1f}%", ha="center", va="center",
                       fontsize=9, color=text_color, fontweight=weight)

    # Highlight HCH row
    hch_idx = methods.index("HCH")
    for spine in ["left", "right"]:
        ax.spines[spine].set_visible(False)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Negative Miss Rate (%)", fontsize=9)

    ax.set_title("Exp1: Negative Price Miss Rate — Backbone × Method", fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig.savefig(FIGURES / "heatmap_neg_miss_rate.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("  ✓ heatmap_neg_miss_rate.png")


# ═══════════════════════════════════════════════════════════
# Fig 7: Exp1 — Normal Harm heatmap (safety indicator)
# ═══════════════════════════════════════════════════════════
def plot_heatmap_normal_harm():
    data = load_json(RESULTS / "exp1_shandong.json")
    backbones = list(data.keys())
    methods = METHOD_ORDER

    matrix = np.zeros((len(methods), len(backbones)))
    for mi, method in enumerate(methods):
        for bi, bb in enumerate(backbones):
            m = data[bb].get(method, {})
            val = m.get("normal_harm", 0)
            matrix[mi, bi] = val if val is not None else np.nan

    fig, ax = plt.subplots(figsize=(10, 6))
    # Diverging colormap: red (harm) -> white (neutral) -> green (improvement)
    im = ax.imshow(matrix, cmap="RdYlGn", aspect="auto",
                  vmin=-50, vmax=5)

    ax.set_xticks(range(len(backbones)))
    ax.set_xticklabels(backbones, fontsize=10)
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels([LABEL_MAP.get(m, m) for m in methods], fontsize=10)

    for i in range(len(methods)):
        for j in range(len(backbones)):
            val = matrix[i, j]
            if not np.isnan(val):
                text_color = "white" if abs(val) > 30 else "black"
                weight = "bold" if methods[i] == "HCH" else "normal"
                ax.text(j, i, f"{val:.1f}", ha="center", va="center",
                       fontsize=9, color=text_color, fontweight=weight)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Normal Harm (negative=damage to normal hours)", fontsize=9)
    ax.set_title("Exp1: Normal Regime Harm — Safety Indicator\n(All values should be ≥ 0 for safe methods)",
                fontsize=11, fontweight="bold")
    plt.tight_layout()
    fig.savefig(FIGURES / "heatmap_normal_harm.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("  ✓ heatmap_normal_harm.png")


# ═══════════════════════════════════════════════════════════
# Fig 8: Exp1 — Episode recall + complete miss dual-axis
# ═══════════════════════════════════════════════════════════
def plot_episode_metrics():
    data = load_json(RESULTS / "exp1_shandong.json")
    backbones = list(data.keys())
    methods = ["HCH", "VahediStyle", "SpikeRegularization", "QuantileCorrection", "CRC", "PIR"]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Exp1: Episode-Level Metrics (Shandong)", fontsize=14, fontweight="bold")

    for ax_idx, (metric, title) in enumerate([
        ("ep_our_episode_recall", "Episode Recall ↑ (higher=better)"),
        ("ep_our_complete_miss", "Complete Miss Rate ↓ (lower=better)"),
    ]):
        ax = axes[ax_idx]
        x = np.arange(len(backbones))
        width = 0.15
        n = len(methods)
        offset_start = -(n - 1) * width / 2

        for mi, method in enumerate(methods):
            vals = []
            for bb in backbones:
                m = data[bb].get(method, {})
                vals.append(m.get(metric, 0) * 100 if m.get(metric) is not None else 0)
            positions = x + offset_start + mi * width
            bars = ax.bar(positions, vals, width,
                         label=LABEL_MAP.get(method, method),
                         color=COLORS.get(method, "#cccccc"),
                         edgecolor="white", alpha=0.85)

            if method == "HCH":
                for bar in bars:
                    bar.set_edgecolor("black")
                    bar.set_linewidth(2)

        ax.set_xticks(x)
        ax.set_xticklabels(backbones, fontsize=10)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_ylabel("%", fontsize=9)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, fontsize=9,
              bbox_to_anchor=(0.5, -0.06), frameon=True)

    plt.tight_layout()
    fig.savefig(FIGURES / "episode_metrics_comparison.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("  ✓ episode_metrics_comparison.png")


# ═══════════════════════════════════════════════════════════
# Fig 9: Exp3 — Negative price detail (MAEonNeg + NegMiss)
# ═══════════════════════════════════════════════════════════
def plot_exp3_neg_detail():
    data = load_json(RESULTS / "exp3_nem_spike.json")
    backbones = list(data.keys())
    methods = METHOD_ORDER_EXP3

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Exp3: NEM SA1 — Negative Price Detail", fontsize=14, fontweight="bold")

    for ax_idx, (metric, title) in enumerate([
        ("mae_on_neg", "MAE on Negative Hours ↓"),
        ("neg_miss_rate", "Negative Miss Rate ↓"),
    ]):
        ax = axes[ax_idx]
        x = np.arange(len(backbones))
        width = 0.18
        n = len(methods)
        offset_start = -(n - 1) * width / 2

        for mi, method in enumerate(methods):
            vals = []
            for bb in backbones:
                m = data[bb].get(method, {})
                v = m.get(metric, 0)
                if v is None:
                    v = 0
                if metric == "neg_miss_rate":
                    v *= 100
                vals.append(v)
            positions = x + offset_start + mi * width
            bars = ax.bar(positions, vals, width,
                         label=LABEL_MAP.get(method, method),
                         color=COLORS.get(method, "#cccccc"),
                         edgecolor="white", alpha=0.85)

            if method == "HCH":
                for bar in bars:
                    bar.set_edgecolor("black")
                    bar.set_linewidth(2)

        ax.set_xticks(x)
        ax.set_xticklabels(backbones, fontsize=10)
        ax.set_title(title, fontsize=11, fontweight="bold")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, fontsize=9,
              bbox_to_anchor=(0.5, -0.06), frameon=True)

    plt.tight_layout()
    fig.savefig(FIGURES / "exp3_neg_detail.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("  ✓ exp3_neg_detail.png")


# ═══════════════════════════════════════════════════════════
# Fig 10: Europe 7 Markets — Multi-market comparison
# ═══════════════════════════════════════════════════════════
def plot_europe_7mkts():
    """Plot comparison across 7 European markets for all methods."""
    data = load_json(RESULTS / "europe_7mkts.json")
    markets = list(data.keys())
    
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle("Europe 7 Markets — Method Comparison", fontsize=14, fontweight="bold")
    
    plot_idx = 0
    for metric, title, fmt in [
        ("mae", "MAE ↓", "{:.1f}"),
        ("neg_miss_rate", "Negative Miss Rate ↓", "{:.1%}"),
        ("ep_our_episode_recall", "Episode Recall ↑", "{:.1%}"),
        ("ndr", "NDR (Safety) ↑", "{:.1%}"),
        ("normal_harm", "Normal Harm ↓", "{:.1f}"),
        ("fire", "Fire Rate ↑", "{:.1%}"),
    ]:
        ax = axes[plot_idx // 3][plot_idx % 3]
        plot_idx += 1
        
        x = np.arange(len(markets))
        width = 0.12
        methods = ["HCH", "Quantile", "Vahedi", "CRC", "PIR", "SpikeReg"]
        method_keys = ["HCH", "Quantile", "Vahedi", "CRC", "PIR", "SpikeReg"]
        colors_list = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6", "#e67e22"]
        
        for mi, (method, mkey) in enumerate(zip(methods, method_keys)):
            vals = []
            for market in markets:
                bb_list = list(data[market].keys())
                # Aggregate: average across backbones for this market+method
                bb_vals = []
                for bb in bb_list:
                    m = data[market][bb].get(mkey, {})
                    val = m.get(metric, None)
                    if val is not None:
                        bb_vals.append(val)
                if bb_vals:
                    vals.append(np.mean(bb_vals))
                else:
                    vals.append(0)
            
            positions = x + (mi - len(methods)/2 + 0.5) * width
            bars = ax.bar(positions, vals, width,
                         label=method, color=colors_list[mi],
                         edgecolor="white", alpha=0.85)
            
            if method == "HCH":
                for bar in bars:
                    bar.set_edgecolor("black")
                    bar.set_linewidth(2)
        
        ax.set_xticks(x)
        ax.set_xticklabels(markets, fontsize=8, rotation=30, ha="right")
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.axhline(y=0, color="gray", linewidth=0.5)
    
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=6, fontsize=8,
              bbox_to_anchor=(0.5, -0.04), frameon=True)
    
    plt.tight_layout()
    fig.savefig(FIGURES / "europe_7mkts_comparison.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("  ✓ europe_7mkts_comparison.png")


# ═══════════════════════════════════════════════════════════
# Fig 11: Cross-market HCH safety vs effectiveness bubble chart
# ═══════════════════════════════════════════════════════════
def plot_cross_market_summary():
    """Bubble chart: each market is a bubble, size=neg_n, x=NDR, y=MAE improvement."""
    # Load all data sources
    euro_data = load_json(RESULTS / "europe_7mkts.json")
    exp2b_data = load_json(RESULTS / "exp2b_epex_pjm.json")
    
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_title("HCH Safety (NDR) vs Effectiveness (MAE Reduction) — All Markets", 
                fontsize=13, fontweight="bold")
    
    all_markets = {}
    
    # Process europe_7mkts
    for market in euro_data:
        hch_ndrs = []
        hch_improvements = []
        neg_counts = []
        for bb in euro_data[market]:
            hch = euro_data[market][bb].get("HCH", {})
            base = euro_data[market][bb].get("Base", {})
            if hch and base:
                ndr = hch.get("ndr", 1.0)
                hch_mae = hch.get("mae", 0)
                base_mae = base.get("mae", 1)
                improvement = (base_mae - hch_mae) / base_mae * 100 if base_mae > 0 else 0
                hch_ndrs.append(ndr)
                hch_improvements.append(improvement)
                neg_counts.append(base.get("neg_n", 0))
        
        if hch_ndrs:
            all_markets[market] = {
                "ndr": np.mean(hch_ndrs),
                "improvement": np.mean(hch_improvements),
                "neg_n": np.mean(neg_counts),
                "color": "#3498db"
            }
    
    # Process exp2b (DE_EPEX and PJM_2020)
    for market in exp2b_data:
        hch_ndrs = []
        hch_improvements = []
        neg_counts = []
        for bb in exp2b_data[market]:
            hch = exp2b_data[market][bb].get("HCH", {})
            base = exp2b_data[market][bb].get("Base", {})
            if hch and base:
                ndr = hch.get("ndr", 1.0)
                hch_mae = hch.get("mae", 0)
                base_mae = base.get("mae", 1)
                improvement = (base_mae - hch_mae) / base_mae * 100 if base_mae > 0 else 0
                hch_ndrs.append(ndr)
                hch_improvements.append(improvement)
                neg_counts.append(base.get("neg_n", 0))
        
        if hch_ndrs:
            color = "#e74c3c" if "DE" in market else "#95a5a6"
            all_markets[market] = {
                "ndr": np.mean(hch_ndrs),
                "improvement": np.mean(hch_improvements),
                "neg_n": np.mean(neg_counts),
                "color": color
            }
    
    # Plot bubbles
    for market, info in all_markets.items():
        size = max(info["neg_n"] * 3, 50)  # scale bubble size
        ax.scatter(info["ndr"] * 100, info["improvement"], 
                  s=size, c=info["color"], alpha=0.7,
                  edgecolors="white", linewidth=1, zorder=5)
        ax.annotate(market, (info["ndr"] * 100, info["improvement"]),
                   fontsize=8, ha="center", va="bottom",
                   xytext=(0, 8), textcoords="offset points")
    
    ax.axhline(y=0, color="gray", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.axvline(x=80, color="gray", linewidth=0.8, linestyle="--", alpha=0.3)
    ax.set_xlabel("NDR (Non-Degradation Rate %)", fontsize=11)
    ax.set_ylabel("MAE Improvement (%)", fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Legend categories
    legend_elements = [
        plt.scatter([], [], c="#3498db", s=100, label="European Markets (7)", alpha=0.7),
        plt.scatter([], [], c="#e74c3c", s=100, label="DE_EPEX", alpha=0.7),
        plt.scatter([], [], c="#95a5a6", s=100, label="PJM_2020 (0% neg)", alpha=0.7),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=9)
    
    plt.tight_layout()
    fig.savefig(FIGURES / "cross_market_hch_summary.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("  ✓ cross_market_hch_summary.png")


# ═══════════════════════════════════════════════════════════
# Fig 12: New experiments — DE_EPEX and PJM_2020 comparison
# ═══════════════════════════════════════════════════════════
def plot_new_exp_de_pjm():
    """Plot DE_EPEX and PJM_2020 backbone × method comparison."""
    data = load_json(RESULTS / "exp2b_epex_pjm.json")
    markets = list(data.keys())
    
    fig, axes = plt.subplots(len(markets), 1, figsize=(16, 6 * len(markets)), sharey=False)
    if len(markets) == 1:
        axes = [axes]
    
    for ax_idx, market in enumerate(markets):
        ax = axes[ax_idx]
        backbones = list(data[market].keys())
        methods = ["Base", "HCH", "QuantileCorrection", "VahediStyle", "CRC", "PIR", "SpikeRegularization"]
        method_labels = ["Base", "HCH", "Quantile", "Vahedi", "CRC", "PIR", "SpikeReg"]
        
        x = np.arange(len(backbones))
        width = 0.12
        n_methods = len(methods)
        offset_start = -(n_methods - 1) * width / 2
        
        for mi, (method, label) in enumerate(zip(methods, method_labels)):
            vals = []
            for bb in backbones:
                m = data[market][bb].get(method, {})
                val = m.get("mae", 0)
                vals.append(val)
            
            positions = x + offset_start + mi * width
            bars = ax.bar(positions, vals, width,
                         label=label, color=COLORS.get(method, "#cccccc"),
                         edgecolor="white", alpha=0.85)
            
            if method == "HCH":
                for bar in bars:
                    bar.set_edgecolor("black")
                    bar.set_linewidth(2)
        
        neg_info = ""
        if backbones:
            base_neg = data[market][backbones[0]]["Base"].get("neg_n", 0)
            neg_info = f" ({base_neg} neg pts)"
        
        ax.set_xticks(x)
        ax.set_xticklabels(backbones, fontsize=10)
        ax.set_title(f"{market}{neg_info}", fontsize=12, fontweight="bold")
        ax.set_ylabel("MAE", fontsize=10)
        ax.grid(True, alpha=0.3, axis="y")
    
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=7, fontsize=8,
              bbox_to_anchor=(0.5, 1.02), frameon=True)
    
    plt.tight_layout()
    fig.savefig(FIGURES / "new_exp_de_pjm_comparison.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("  ✓ new_exp_de_pjm_comparison.png")


# ═══════════════════════════════════════════════════════════
# Fig 13: Europe markets heatmap — NegMiss and EpRecall
# ═══════════════════════════════════════════════════════════
def plot_europe_heatmaps():
    """Two heatmaps: NegMiss rate and Episode Recall across Europe markets."""
    data = load_json(RESULTS / "europe_7mkts.json")
    markets = list(data.keys())
    methods = ["HCH", "Quantile", "Vahedi", "CRC", "PIR", "SpikeReg"]
    method_keys = ["HCH", "Quantile", "Vahedi", "CRC", "PIR", "SpikeReg"]
    
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    fig.suptitle("Europe 7 Markets — Method Performance Heatmaps", 
                fontsize=14, fontweight="bold")
    
    for ax_idx, (metric, title, cmap) in enumerate([
        ("neg_miss_rate", "Negative Miss Rate (%)", "RdYlGn_r"),
        ("ep_our_episode_recall", "Episode Recall (%)", "RdYlGn"),
    ]):
        ax = axes[ax_idx]
        matrix = np.zeros((len(methods), len(markets)))
        
        for mi, mkey in enumerate(method_keys):
            for ma, market in enumerate(markets):
                bb_vals = []
                for bb in data[market]:
                    m = data[market][bb].get(mkey, {})
                    val = m.get(metric, None)
                    if val is not None:
                        bb_vals.append(val * 100)
                matrix[mi, ma] = np.mean(bb_vals) if bb_vals else np.nan
        
        im = ax.imshow(matrix, cmap=cmap, aspect="auto")
        
        ax.set_xticks(range(len(markets)))
        ax.set_xticklabels(markets, fontsize=9, rotation=30, ha="right")
        ax.set_yticks(range(len(methods)))
        ax.set_yticklabels(methods, fontsize=10)
        
        for i in range(len(methods)):
            for j in range(len(markets)):
                val = matrix[i, j]
                if not np.isnan(val):
                    text_color = "white" if (val > 50 if "Miss" in title else val < 30) else "black"
                    weight = "bold" if methods[i] == "HCH" else "normal"
                    ax.text(j, i, f"{val:.1f}", ha="center", va="center",
                           fontsize=8, color=text_color, fontweight=weight)
        
        cbar = fig.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label(title, fontsize=9)
        ax.set_title(title, fontsize=11, fontweight="bold")
    
    plt.tight_layout()
    fig.savefig(FIGURES / "europe_heatmaps.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("  ✓ europe_heatmaps.png")


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 60)
    print("Plotting Comparative Experiment Results")
    print(f"Results dir: {RESULTS}")
    print(f"Figures dir: {FIGURES}")
    print("=" * 60)

    # Original experiments
    plot_exp1_shandong()
    plot_exp1_hch_improvement()
    plot_heatmap_neg_miss()
    plot_heatmap_normal_harm()
    plot_episode_metrics()
    plot_exp2_ltsf()
    plot_exp3_nem()
    plot_exp3_neg_detail()
    plot_cross_safety_effectiveness()
    
    # New experiments
    plot_europe_7mkts()
    plot_cross_market_summary()
    plot_new_exp_de_pjm()
    plot_europe_heatmaps()

    print("\n" + "=" * 60)
    print("Done! All figures saved to:", FIGURES)
    print("=" * 60)
