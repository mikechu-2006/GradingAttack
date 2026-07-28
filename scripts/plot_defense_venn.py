"""Generate three Venn diagrams comparing four defense methods (SR, PD, HS, AS)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch

DOCS_DIR = Path(__file__).resolve().parents[1] / "docs"

DEFENSES = {
    "SR": {"color": "#3b82f6", "label_color": "#1d4ed8", "edge": "#2563eb"},
    "PD": {"color": "#10b981", "label_color": "#047857", "edge": "#059669"},
    "HS": {"color": "#f59e0b", "label_color": "#b45309", "edge": "#d97706"},
    "AS": {"color": "#8b5cf6", "label_color": "#6d28d9", "edge": "#7c3aed"},
}

FIG_W, FIG_H, DPI = 9.0, 5.8, 160
TITLE_KW = dict(fontsize=15, fontweight="bold", color="#111827", pad=14)
SET_LABEL_KW = dict(fontsize=11.5, fontweight="bold", ha="center", va="bottom")
CAPTION_KW = dict(fontsize=8.5, ha="center", va="center")
ABBR_KW = dict(fontsize=12, fontweight="bold", ha="center", va="center")
NOTE_KW = dict(fontsize=8.5, ha="center", va="top")


def _new_figure(title: str, footnote: str) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H), dpi=DPI)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_xlim(-1.55, 1.55)
    ax.set_ylim(-1.02, 1.12)
    ax.set_title(title, **TITLE_KW)
    ax.text(
        0.5,
        -0.06,
        footnote,
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=9.5,
        color="#6b7280",
        style="italic",
        zorder=10,
    )
    return fig, ax


def _draw_region(
    ax: plt.Axes,
    cx: float,
    cy: float,
    r: float,
    color: str,
    *,
    alpha: float = 0.32,
    zorder: int = 1,
) -> None:
    ax.add_patch(
        Circle(
            (cx, cy),
            r,
            facecolor=color,
            edgecolor="#ffffff",
            linewidth=2.8,
            alpha=alpha,
            zorder=zorder,
        )
    )


def _draw_set_label(
    ax: plt.Axes,
    cx: float,
    cy: float,
    r: float,
    label: str,
    color: str,
) -> None:
    lines = label.count("\n") + 1
    y_offset = 0.06 + (lines - 1) * 0.05
    ax.text(
        cx,
        cy + r + y_offset,
        label,
        color=color,
        zorder=6,
        linespacing=1.2,
        **SET_LABEL_KW,
    )


def _draw_badge(ax: plt.Axes, abbr: str, x: float, y: float) -> None:
    info = DEFENSES[abbr]
    ax.add_patch(
        FancyBboxPatch(
            (x - 0.11, y - 0.07),
            0.22,
            0.14,
            boxstyle="round,pad=0.02,rounding_size=0.04",
            facecolor="white",
            edgecolor=info["edge"],
            linewidth=1.8,
            alpha=0.95,
            zorder=4,
        )
    )
    ax.text(x, y, abbr, color=info["label_color"], **ABBR_KW, zorder=5)


def _place_defense(
    ax: plt.Axes,
    abbr: str,
    x: float,
    y: float,
    note: str | None = None,
    note_offset: float = -0.16,
) -> None:
    _draw_badge(ax, abbr, x, y)
    if note:
        va = "bottom" if note_offset > 0 else "top"
        ax.text(
            x,
            y + note_offset,
            note,
            color=DEFENSES[abbr]["label_color"],
            fontsize=8.5,
            ha="center",
            va=va,
            zorder=5,
        )


def _save(fig: plt.Figure, output: Path) -> None:
    fig.subplots_adjust(left=0.01, right=0.99, top=0.88, bottom=0.12)
    fig.savefig(output, bbox_inches="tight", facecolor="white", pad_inches=0.12)
    plt.close(fig)


def plot_intervention_level(output: Path) -> None:
    """Prompt-level (SR, PD) vs mechanism-level (HS, AS)."""
    fig, ax = _new_figure(
        "Intervention Level",
        "SR & PD modify prompts only; HS & AS hook into attention computation",
    )

    r, cy = 0.52, 0.02
    gap = 0.22
    left_cx = -(r + gap / 2)
    right_cx = r + gap / 2

    _draw_region(ax, left_cx, cy, r, "#60a5fa")
    _draw_region(ax, right_cx, cy, r, "#fbbf24")
    _draw_set_label(ax, left_cx, cy, r, "Prompt-level", "#1e40af")
    _draw_set_label(ax, right_cx, cy, r, "Mechanism-level", "#92400e")

    _place_defense(ax, "SR", left_cx, cy + 0.14)
    _place_defense(ax, "PD", left_cx, cy - 0.14)
    _place_defense(ax, "HS", right_cx, cy + 0.14)
    _place_defense(ax, "AS", right_cx, cy - 0.14)

    ax.text(left_cx, cy - r - 0.08, "No model hooks", color="#1e40af", zorder=6, **NOTE_KW)
    ax.text(right_cx, cy - r - 0.08, "Attention hooks at inference", color="#92400e", zorder=6, **NOTE_KW)

    _save(fig, output)


def plot_deployment_cost(output: Path) -> None:
    """Deployment cost tiers with overlapping requirements."""
    fig, ax = _new_figure(
        "Deployment Cost",
        "Cost increases left → right: text-only guards vs. hook-based defenses",
    )

    r = 0.48
    left_cx, right_cx, cy = -0.62, 0.40, 0.0
    inner_r = 0.27
    inner_cx = 0.76

    _draw_region(ax, left_cx, cy, r, "#93c5fd")
    _draw_region(ax, right_cx, cy, r, "#fcd34d")
    _draw_region(ax, inner_cx, cy, inner_r, "#f59e0b", alpha=0.45, zorder=2)

    _draw_set_label(ax, left_cx, cy, r, "No model modification", "#1e40af")
    _draw_set_label(ax, right_cx, cy, r, "Inference hooks\n+ eager attention", "#92400e")

    _place_defense(ax, "SR", left_cx, cy + 0.20, "1 forward pass", note_offset=0.15)
    _place_defense(ax, "PD", left_cx, cy - 0.20, "+ longer prompt", note_offset=-0.15)
    _place_defense(ax, "AS", right_cx - 0.22, cy + 0.14)
    _place_defense(ax, "HS", inner_cx, cy - 0.02, "+ suffix length", note_offset=-0.15)

    _save(fig, output)


def plot_attack_target(output: Path) -> None:
    """Each defense targets a distinct attack mechanism."""
    fig, ax = _new_figure(
        "Target Attack Mechanism",
        "Each defense assumes a different explanation of why attacks succeed",
    )

    r = 0.40
    gap_x, gap_y = 0.28, 0.22
    positions = {
        "SR": (-(r + gap_x / 2), r + gap_y / 2),
        "PD": (r + gap_x / 2, r + gap_y / 2),
        "HS": (-(r + gap_x / 2), -(r + gap_y / 2)),
        "AS": (r + gap_x / 2, -(r + gap_y / 2)),
    }
    descriptions = {
        "SR": "Semantic injection",
        "PD": "Meaningless suffixes",
        "HS": "Attention hijacking",
        "AS": "Attention slipping",
    }

    for abbr, (cx, cy) in positions.items():
        info = DEFENSES[abbr]
        _draw_region(ax, cx, cy, r, info["color"])
        _draw_badge(ax, abbr, cx, cy + 0.05)
        ax.text(
            cx,
            cy - 0.12,
            descriptions[abbr],
            color=info["label_color"],
            **CAPTION_KW,
            zorder=5,
        )

    _save(fig, output)


def main() -> None:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    outputs = {
        "intervention_level": DOCS_DIR / "defense_venn_intervention_level.svg",
        "deployment_cost": DOCS_DIR / "defense_venn_deployment_cost.svg",
        "attack_target": DOCS_DIR / "defense_venn_attack_target.svg",
    }
    plot_intervention_level(outputs["intervention_level"])
    plot_deployment_cost(outputs["deployment_cost"])
    plot_attack_target(outputs["attack_target"])
    for path in outputs.values():
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
