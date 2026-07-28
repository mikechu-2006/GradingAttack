"""Generate mechanistic analysis figure for the defense speech."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

DOCS_DIR = Path(__file__).resolve().parents[1] / "docs"
OUTPUT = DOCS_DIR / "defense_mechanistic_analysis.svg"

DEFENSES = {
    "SR": {"color": "#3b82f6", "text": "#1d4ed8", "edge": "#2563eb"},
    "PD": {"color": "#10b981", "text": "#047857", "edge": "#059669"},
    "HS": {"color": "#f59e0b", "text": "#b45309", "edge": "#d97706"},
    "AS": {"color": "#8b5cf6", "text": "#6d28d9", "edge": "#7c3aed"},
}

ATTACKS = {
    "GCG": {
        "label": "GCG Suffix Bank",
        "attention": 14,
        "mechanism": "Attention hijacking",
        "bar_color": "#f59e0b",
        "panel_bg": "#fffbeb",
        "panel_edge": "#fcd34d",
        "aligned": ["PD", "HS"],
        "mismatched": ["SR", "AS"],
    },
    "RolePlay": {
        "label": "RolePlay",
        "attention": 4,
        "mechanism": "Semantic persuasion",
        "bar_color": "#3b82f6",
        "panel_bg": "#eff6ff",
        "panel_edge": "#93c5fd",
        "aligned": ["SR", "PD"],
        "mismatched": ["HS", "AS"],
    },
}


def _draw_badge(ax, abbr: str, x: float, y: float, *, alpha: float = 1.0, strike: bool = False) -> None:
    info = DEFENSES[abbr]
    ax.add_patch(
        FancyBboxPatch(
            (x - 0.038, y - 0.022),
            0.076,
            0.044,
            boxstyle="round,pad=0.01,rounding_size=0.015",
            transform=ax.transAxes,
            facecolor="white",
            edgecolor=info["edge"],
            linewidth=1.6,
            alpha=alpha,
            zorder=4,
        )
    )
    ax.text(
        x, y, abbr,
        transform=ax.transAxes,
        ha="center", va="center",
        fontsize=11, fontweight="bold",
        color=info["text"], alpha=alpha, zorder=5,
    )
    if strike:
        ax.plot(
            [x - 0.032, x + 0.032], [y, y],
            transform=ax.transAxes,
            color="#b91c1c", linewidth=1.8, zorder=6,
        )


def _draw_attention_bar(ax, cx: float, cy: float, width: float, height: float, pct: int, color: str) -> None:
    rest = 100 - pct
    bar_left = cx - width / 2
    ax.add_patch(
        mpatches.FancyBboxPatch(
            (bar_left, cy - height / 2),
            width, height,
            boxstyle="round,pad=0.004,rounding_size=0.012",
            transform=ax.transAxes,
            facecolor="#e5e7eb", edgecolor="#d1d5db", linewidth=1.0, zorder=2,
        )
    )
    attack_w = width * pct / 100
    ax.add_patch(
        mpatches.FancyBboxPatch(
            (bar_left, cy - height / 2),
            attack_w, height,
            boxstyle="round,pad=0.004,rounding_size=0.012",
            transform=ax.transAxes,
            facecolor=color, edgecolor="none", alpha=0.85, zorder=3,
        )
    )
    pct_x = bar_left + max(attack_w / 2, 0.025)
    pct_color = "#ffffff" if pct >= 10 else color
    ax.text(
        pct_x, cy,
        f"{pct}%",
        transform=ax.transAxes,
        ha="center", va="center",
        fontsize=12, fontweight="bold",
        color=pct_color, zorder=4,
    )
    ax.text(
        bar_left + attack_w + (width - attack_w) / 2, cy,
        "grading context",
        transform=ax.transAxes,
        ha="center", va="center",
        fontsize=8, color="#6b7280", zorder=4,
    )
    ax.text(
        cx, cy + height / 2 + 0.035,
        "Adversarial attention share",
        transform=ax.transAxes,
        ha="center", va="bottom",
        fontsize=9, fontweight="bold", color="#374151", zorder=4,
    )


def _draw_panel(ax, attack_key: str, panel_x: float, panel_w: float) -> None:
    data = ATTACKS[attack_key]
    px = panel_x
    pw = panel_w

    ax.add_patch(
        FancyBboxPatch(
            (px, 0.12), pw, 0.72,
            boxstyle="round,pad=0.012,rounding_size=0.02",
            transform=ax.transAxes,
            facecolor=data["panel_bg"],
            edgecolor=data["panel_edge"],
            linewidth=2.0,
            zorder=1,
        )
    )

    cx = px + pw / 2
    ax.text(cx, 0.78, data["label"], transform=ax.transAxes,
            ha="center", va="center", fontsize=13, fontweight="bold", color="#111827", zorder=3)
    ax.text(cx, 0.73, data["mechanism"], transform=ax.transAxes,
            ha="center", va="center", fontsize=10, color="#4b5563", style="italic", zorder=3)

    _draw_attention_bar(ax, cx, 0.58, pw * 0.78, 0.07, data["attention"], data["bar_color"])

    # Aligned defenses
    ax.text(px + 0.04, 0.44, "Aligned defenses ✓", transform=ax.transAxes,
            ha="left", va="center", fontsize=9.5, fontweight="bold", color="#047857", zorder=3)
    n_aligned = len(data["aligned"])
    start_x = cx - (n_aligned - 1) * 0.05
    for i, abbr in enumerate(data["aligned"]):
        _draw_badge(ax, abbr, start_x + i * 0.10, 0.36)

    # Mismatched defenses
    ax.text(px + 0.04, 0.26, "Mismatched defenses ✗", transform=ax.transAxes,
            ha="left", va="center", fontsize=9.5, fontweight="bold", color="#b91c1c", zorder=3)
    n_mis = len(data["mismatched"])
    start_x = cx - (n_mis - 1) * 0.05
    for i, abbr in enumerate(data["mismatched"]):
        _draw_badge(ax, abbr, start_x + i * 0.10, 0.18, alpha=0.55, strike=True)


def plot_mechanistic_analysis(output: Path) -> None:
    fig, ax = plt.subplots(figsize=(10.0, 5.5), dpi=160)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.5, 0.96, "Mechanistic Analysis", transform=ax.transAxes,
            ha="center", va="top", fontsize=16, fontweight="bold", color="#111827")
    ax.text(
        0.5, 0.905,
        "Why defense effectiveness depends on the attack mechanism",
        transform=ax.transAxes,
        ha="center", va="top", fontsize=10, color="#6b7280",
    )

    gap = 0.04
    panel_w = (0.94 - gap) / 2
    _draw_panel(ax, "GCG", 0.03, panel_w)
    _draw_panel(ax, "RolePlay", 0.03 + panel_w + gap, panel_w)

    # Center arrow / lesson
    ax.annotate(
        "",
        xy=(0.72, 0.06), xytext=(0.28, 0.06),
        xycoords="axes fraction",
        arrowprops=dict(arrowstyle="<->", color="#9ca3af", lw=1.5),
    )
    ax.text(
        0.5, 0.06,
        "Defenses aligned with the attack mechanism work; mismatched ones fail",
        transform=ax.transAxes,
        ha="center", va="center",
        fontsize=10, fontweight="bold", color="#111827",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#f9fafb", edgecolor="#e5e7eb"),
    )

    fig.savefig(output, bbox_inches="tight", facecolor="white", pad_inches=0.12)
    plt.close(fig)


def main() -> None:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    plot_mechanistic_analysis(OUTPUT)
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
