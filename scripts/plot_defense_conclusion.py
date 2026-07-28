"""Generate conclusion recommendation figure for the defense speech."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

DOCS_DIR = Path(__file__).resolve().parents[1] / "docs"
OUTPUT = DOCS_DIR / "defense_conclusion_recommendations.svg"

DEFENSES = {
    "SR": {"bg": "#eff6ff", "text": "#1d4ed8", "edge": "#3b82f6"},
    "PD": {"bg": "#ecfdf5", "text": "#047857", "edge": "#10b981"},
    "HS": {"bg": "#fffbeb", "text": "#b45309", "edge": "#f59e0b"},
    "AS": {"bg": "#f5f3ff", "text": "#6d28d9", "edge": "#8b5cf6"},
}

RECOMMENDATIONS = [
    {
        "abbr": "SR",
        "headline": "Best low-cost choice",
        "lines": ["Semantic attacks (RolePlay)", "Preserves QWK"],
        "tag": "Low cost",
        "tag_color": "#047857",
        "muted": False,
    },
    {
        "abbr": "PD",
        "headline": "Go-to vs GCG",
        "lines": ["Strongest GCG ASR cut (3.6%)", "⚠ Clean-QWK side effects"],
        "tag": "Best vs GCG",
        "tag_color": "#047857",
        "muted": False,
    },
    {
        "abbr": "HS",
        "headline": "Strongest inference-time",
        "lines": ["Attention hijacking threat", "Best mechanistic on GCG", "⚠ High deploy cost"],
        "tag": "Mech. best",
        "tag_color": "#b45309",
        "muted": False,
    },
    {
        "abbr": "AS",
        "headline": "Limited practical value",
        "lines": ["Underperforms HS", "No clear win"],
        "tag": "Not recommended",
        "tag_color": "#9ca3af",
        "muted": True,
    },
]


def _draw_card(ax, item: dict, x: float, y: float, w: float, h: float) -> None:
    info = DEFENSES[item["abbr"]]
    alpha = 0.55 if item["muted"] else 1.0
    face = "#f9fafb" if item["muted"] else info["bg"]
    edge = "#d1d5db" if item["muted"] else info["edge"]
    cx = x + w / 2

    ax.add_patch(
        FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.012,rounding_size=0.018",
            transform=ax.transAxes,
            facecolor=face,
            edgecolor=edge,
            linewidth=2.0 if not item["muted"] else 1.2,
            alpha=alpha,
            zorder=2,
        )
    )

    # Vertical stack from top — fixed spacing, no horizontal sharing
    gap = 0.038
    cur = y + h - 0.028

    # 1) Badge
    cur -= 0.018
    ax.add_patch(
        FancyBboxPatch(
            (cx - 0.032, cur - 0.018), 0.064, 0.036,
            boxstyle="round,pad=0.006,rounding_size=0.01",
            transform=ax.transAxes,
            facecolor="white", edgecolor=info["edge"],
            linewidth=1.8, alpha=alpha, zorder=3,
        )
    )
    ax.text(cx, cur, item["abbr"], transform=ax.transAxes,
            ha="center", va="center", fontsize=12, fontweight="bold",
            color=info["text"], alpha=alpha, zorder=4)
    cur -= 0.018 + gap

    # 2) Headline
    ax.text(cx, cur, item["headline"], transform=ax.transAxes,
            ha="center", va="top", fontsize=10.5, fontweight="bold",
            color="#6b7280" if item["muted"] else "#111827",
            alpha=alpha, zorder=4)
    cur -= 0.022 + gap

    # 3) Tag pill
    tag_w = 0.12 if len(item["tag"]) < 14 else 0.145
    tag_h = 0.028
    ax.add_patch(
        FancyBboxPatch(
            (cx - tag_w / 2, cur - tag_h), tag_w, tag_h,
            boxstyle="round,pad=0.005,rounding_size=0.008",
            transform=ax.transAxes,
            facecolor="white", edgecolor=item["tag_color"],
            linewidth=1.2, alpha=alpha, zorder=3,
        )
    )
    ax.text(cx, cur - tag_h / 2, item["tag"], transform=ax.transAxes,
            ha="center", va="center", fontsize=7.8, fontweight="bold",
            color=item["tag_color"], alpha=alpha, zorder=4)
    cur -= tag_h + gap + 0.01

    # 4) Body lines — distribute remaining space
    floor = y + 0.028
    n = len(item["lines"])
    slot = (cur - floor) / n
    for i, line in enumerate(item["lines"]):
        ly = cur - (i + 0.5) * slot
        color = "#b91c1c" if line.startswith("⚠") else ("#6b7280" if item["muted"] else "#374151")
        ax.text(cx, ly, line, transform=ax.transAxes,
                ha="center", va="center", fontsize=8.8,
                color=color, alpha=alpha, zorder=4)


def plot_conclusion(output: Path) -> None:
    fig, ax = plt.subplots(figsize=(10.0, 7.5), dpi=160)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.5, 0.97, "Conclusion: No Single Defense Wins Everywhere",
            transform=ax.transAxes, ha="center", va="top",
            fontsize=16, fontweight="bold", color="#111827")
    ax.text(0.5, 0.935, "Choose the defense that matches your threat model and budget",
            transform=ax.transAxes, ha="center", va="top",
            fontsize=10, color="#6b7280")

    gap_x, gap_y = 0.04, 0.05
    card_w = (0.94 - gap_x) / 2
    card_h = (0.78 - gap_y) / 2
    left, top = 0.03, 0.88

    positions = [
        (left, top - card_h),
        (left + card_w + gap_x, top - card_h),
        (left, top - 2 * card_h - gap_y),
        (left + card_w + gap_x, top - 2 * card_h - gap_y),
    ]

    for item, pos in zip(RECOMMENDATIONS, positions):
        _draw_card(ax, item, *pos, card_w, card_h)

    fig.savefig(output, bbox_inches="tight", facecolor="white", pad_inches=0.12)
    plt.close(fig)


def main() -> None:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    plot_conclusion(OUTPUT)
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
