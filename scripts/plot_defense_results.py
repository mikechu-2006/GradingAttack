"""Generate result figures from defense_mechanisms_speech.md Table 1 & Table 2."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

DOCS_DIR = Path(__file__).resolve().parents[1] / "docs"

DEFENSE_COLORS = {
    "None": {"bg": "#f9fafb", "text": "#374151", "edge": "#d1d5db"},
    "SR": {"bg": "#eff6ff", "text": "#1d4ed8", "edge": "#3b82f6"},
    "PD": {"bg": "#ecfdf5", "text": "#047857", "edge": "#10b981"},
    "HS": {"bg": "#fffbeb", "text": "#b45309", "edge": "#f59e0b"},
    "AS": {"bg": "#f5f3ff", "text": "#6d28d9", "edge": "#8b5cf6"},
}

CELL_STYLE = {
    "best": {"face": "#059669", "text": "#ffffff", "weight": "bold"},
    "good": {"face": "#d1fae5", "text": "#047857", "weight": "bold"},
    "warn": {"face": "#fef3c7", "text": "#b45309", "weight": "bold"},
    "bad": {"face": "#fee2e2", "text": "#b91c1c", "weight": "bold"},
    "caution": {"face": "#ffedd5", "text": "#c2410c", "weight": "bold"},
    "na": {"face": "#f3f4f6", "text": "#9ca3af", "weight": "normal"},
    "base": {"face": "#ffffff", "text": "#374151", "weight": "normal"},
    "alt": {"face": "#f9fafb", "text": "#374151", "weight": "normal"},
}

# ── Table 1 · Detailed results ──────────────────────────────────────────────

DETAILED_COLUMNS = [
    ("Def", 0.06),
    ("Atk", 0.07),
    ("Setting", 0.10),
    ("N", 0.05),
    ("ASR→Def", 0.12),
    ("ΔASR", 0.07),
    ("QWK", 0.07),
    ("Assess.", 0.12),
]

DETAILED_ROWS: list[dict[str, str]] = [
    {
        "def": "SR", "attack": "RP", "setting": "bin 2c", "n": "300",
        "asr": "0.6→0%", "red": "100%", "qwk_ret": "107%", "assess": "✓ QWK ok",
    },
    {
        "def": "SR", "attack": "RP", "setting": "3-class", "n": "300",
        "asr": "17.8→5.5%", "red": "69%", "qwk_ret": "103%", "assess": "✓ Moderate",
    },
    {
        "def": "SR", "attack": "GCG", "setting": "held 500", "n": "500",
        "asr": "99.6→95.5%", "red": "4%", "qwk_ret": "102%", "assess": "✗ Minimal",
    },
    {
        "def": "PD", "attack": "RP", "setting": "bin 2c", "n": "300",
        "asr": "0.6→0%", "red": "100%", "qwk_ret": "42%", "assess": "△ QWK↓",
    },
    {
        "def": "PD", "attack": "GCG", "setting": "pipe.500†", "n": "500",
        "asr": "100→3.6%", "red": "96%", "qwk_ret": "68%", "assess": "✓ Strong",
    },
    {
        "def": "PD", "attack": "GCG", "setting": "abl.500†", "n": "500",
        "asr": "99.6→36.8%", "red": "63%", "qwk_ret": "137%", "assess": "✓ Strong",
    },
    {
        "def": "HS", "attack": "GCG", "setting": "bin 2c", "n": "100",
        "asr": "100→75.4%", "red": "25%", "qwk_ret": "92%", "assess": "✓ Mech.",
    },
    {
        "def": "HS", "attack": "Inj.DC", "setting": "bin 2c", "n": "50",
        "asr": "74.1→53.6%", "red": "28%", "qwk_ret": "98%", "assess": "✓ Effective",
    },
    {
        "def": "HS", "attack": "Inj.AO/IM", "setting": "bin 2c", "n": "50",
        "asr": "100→100%", "red": "0%", "qwk_ret": "98%", "assess": "✗ None",
    },
    {
        "def": "HS", "attack": "RP", "setting": "expl.3c", "n": "300",
        "asr": "0.6→1.7%", "red": "−193%", "qwk_ret": "88%", "assess": "✗ No gain",
    },
    {
        "def": "AS", "attack": "GCG", "setting": "bin 2c", "n": "100",
        "asr": "100→98.1%", "red": "2%", "qwk_ret": "96%", "assess": "✗ Minimal",
    },
    {
        "def": "AS", "attack": "Inj.DC", "setting": "bin 2c", "n": "50",
        "asr": "74.1→77.8%", "red": "−5%", "qwk_ret": "109%", "assess": "✗ Backfire",
    },
    {
        "def": "AS", "attack": "Inj.AO/IM", "setting": "bin 2c", "n": "50",
        "asr": "100→100%", "red": "0%", "qwk_ret": "109%", "assess": "✗ None",
    },
    {
        "def": "AS", "attack": "RP", "setting": "expl.3c", "n": "300",
        "asr": "16.6→21.6%", "red": "−30%", "qwk_ret": "105%", "assess": "✗ ASR↑",
    },
]

# (row_index, column_key) → highlight style
DETAILED_HIGHLIGHTS: dict[tuple[int, str], str] = {
    (0, "asr"): "good", (0, "red"): "good", (0, "qwk_ret"): "good",
    (1, "asr"): "good", (1, "red"): "good",
    (2, "asr"): "bad", (2, "red"): "bad", (2, "assess"): "bad",
    (3, "asr"): "good", (3, "qwk_ret"): "bad", (3, "assess"): "caution",
    (4, "asr"): "best", (4, "red"): "best",
    (5, "asr"): "good", (5, "red"): "good", (5, "qwk_ret"): "good",
    (6, "asr"): "warn", (6, "assess"): "good",
    (7, "asr"): "good", (7, "red"): "good",
    (8, "asr"): "bad", (8, "red"): "bad", (8, "assess"): "bad",
    (9, "asr"): "bad", (9, "red"): "bad", (9, "assess"): "bad",
    (10, "asr"): "bad", (10, "red"): "bad", (10, "assess"): "bad",
    (11, "asr"): "bad", (11, "red"): "bad", (11, "assess"): "bad",
    (12, "asr"): "bad", (12, "red"): "bad", (12, "assess"): "bad",
    (13, "asr"): "bad", (13, "red"): "bad", (13, "assess"): "bad",
}

# ── Table 2 · Defended ASR matrix ───────────────────────────────────────────

MATRIX_COLUMNS = ["RP (2c)", "RP (3c)", "GCG", "Inj. DC", "Inj. AO/IM"]

MATRIX_ROWS: list[tuple[str, list[float | None]]] = [
    ("None", [0.6, 17.8, 100.0, 74.1, 100.0]),
    ("SR", [0.0, 5.5, 95.5, None, None]),
    ("PD", [0.0, None, 3.6, None, None]),
    ("HS", [1.7, None, 75.4, 53.6, 100.0]),
    ("AS", [None, 21.6, 98.1, 77.8, 100.0]),
]

MATRIX_HIGHLIGHTS: dict[tuple[str, int], str] = {
    ("PD", 2): "best",
    ("SR", 0): "good", ("PD", 0): "good", ("SR", 1): "good", ("HS", 3): "good",
    ("HS", 2): "warn",
    ("SR", 2): "bad", ("HS", 0): "bad", ("AS", 1): "bad",
    ("AS", 2): "bad", ("AS", 3): "bad", ("HS", 4): "bad", ("AS", 4): "bad",
}


def _style_for(key: tuple, highlights: dict, default: str = "base") -> dict:
    return CELL_STYLE[highlights.get(key, default)]


def _draw_legend(ax: plt.Axes, y_base: float = 0.04) -> None:
    legend_items = [
        ("best", "Best result"),
        ("good", "Effective"),
        ("warn", "Mech. best (ASR still high)"),
        ("bad", "Ineffective / backfires"),
        ("caution", "Trade-off (e.g. QWK drop)"),
    ]
    for idx, (key, label) in enumerate(legend_items):
        row, col = divmod(idx, 3)
        lx = 0.04 + col * 0.32
        ly = y_base - row * 0.028
        style = CELL_STYLE[key]
        ax.add_patch(
            mpatches.Rectangle(
                (lx, ly - 0.009),
                0.018,
                0.018,
                transform=ax.transAxes,
                facecolor=style["face"],
                edgecolor="#d1d5db",
                linewidth=0.8,
                clip_on=False,
            )
        )
        ax.text(lx + 0.024, ly, label, ha="left", va="center", fontsize=8.5,
                color="#4b5563", transform=ax.transAxes)


def _draw_cell(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    text: str,
    style: dict,
    *,
    fontsize: float = 9,
    face_override: str | None = None,
    text_override: str | None = None,
) -> None:
    ax.add_patch(
        mpatches.FancyBboxPatch(
            (x + 0.002, y - h / 2 + 0.001),
            w - 0.004,
            h - 0.002,
            boxstyle="round,pad=0.003,rounding_size=0.008",
            transform=ax.transAxes,
            facecolor=face_override or style["face"],
            edgecolor="#e5e7eb",
            linewidth=0.8,
            zorder=2,
        )
    )
    ax.text(
        x + w / 2,
        y,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight=style["weight"],
        color=text_override or style["text"],
        transform=ax.transAxes,
        zorder=3,
    )


def plot_detailed_results(output: Path) -> None:
    col_keys_lower = ["def", "attack", "setting", "n", "asr", "red", "qwk_ret", "assess"]
    col_weights = [c[1] for c in DETAILED_COLUMNS]
    total_w = sum(col_weights)

    n_rows = len(DETAILED_ROWS)
    fig, ax = plt.subplots(figsize=(9.8, 8.0), dpi=160)
    ax.axis("off")

    left, top = 0.02, 0.93
    table_w = 0.96
    header_h = 0.045
    cell_h = 0.048
    bottom_margin = 0.12

    ax.text(0.5, 0.975, "Detailed Defense Results", ha="center", va="top",
            fontsize=16, fontweight="bold", color="#111827", transform=ax.transAxes)
    ax.text(0.5, 0.945, "Table 1  ·  Llama-3.1-8B-Instruct  ·  SciEntsBank",
            ha="center", va="top", fontsize=10, color="#6b7280", transform=ax.transAxes)

    # Column headers
    cx = left
    for (header, weight), key in zip(DETAILED_COLUMNS, col_keys_lower):
        w = table_w * weight / total_w
        _draw_cell(ax, cx, top - header_h / 2, w, header_h, header,
                   {"face": "#e5e7eb", "text": "#111827", "weight": "bold"}, fontsize=8.5)
        cx += w

    # Data rows
    for i, row in enumerate(DETAILED_ROWS):
        row_y = top - header_h - i * cell_h - cell_h / 2
        cx = left
        def_info = DEFENSE_COLORS[row["def"]]
        for j, (key, weight) in enumerate(zip(col_keys_lower, col_weights)):
            w = table_w * weight / total_w
            hi_key = (i, key)
            style = _style_for(hi_key, DETAILED_HIGHLIGHTS, "alt" if i % 2 else "base")
            face = def_info["bg"] if key == "def" else style["face"]
            text_color = def_info["text"] if key == "def" else style["text"]
            weight_font = "bold" if key == "def" else style["weight"]
            ax.add_patch(
                mpatches.FancyBboxPatch(
                    (cx + 0.002, row_y - cell_h / 2 + 0.001),
                    w - 0.004,
                    cell_h - 0.002,
                    boxstyle="round,pad=0.003,rounding_size=0.008",
                    transform=ax.transAxes,
                    facecolor=face,
                    edgecolor=def_info["edge"] if key == "def" else "#e5e7eb",
                    linewidth=1.2 if key == "def" else 0.8,
                    zorder=2,
                )
            )
            fs = 9.5 if key == "def" else (7.8 if key in ("setting", "assess", "attack") else 8.5)
            ax.text(cx + w / 2, row_y, row[key], ha="center", va="center",
                    fontsize=fs, fontweight=weight_font, color=text_color,
                    transform=ax.transAxes, zorder=3)
            cx += w

    _draw_legend(ax, y_base=bottom_margin + 0.06)
    ax.text(0.5, 0.018,
            "† Two GCG+PD pipelines; latest pipeline def. ASR = 3.6%",
            ha="center", va="bottom", fontsize=8.5, color="#6b7280", style="italic",
            transform=ax.transAxes)

    fig.savefig(output, bbox_inches="tight", facecolor="white", pad_inches=0.08)
    plt.close(fig)


def plot_asr_matrix(output: Path) -> None:
    n_rows = len(MATRIX_ROWS)
    n_cols = len(MATRIX_COLUMNS)

    fig, ax = plt.subplots(figsize=(10.5, 5.2), dpi=160)
    ax.axis("off")

    label_w = 0.11
    cell_w = (0.96 - label_w) / n_cols
    header_h = 0.10
    cell_h = (0.78 - header_h) / n_rows
    top, left = 0.88, 0.03

    ax.text(0.5, 0.97, "Defended ASR Matrix (%)", ha="center", va="top",
            fontsize=16, fontweight="bold", color="#111827", transform=ax.transAxes)
    ax.text(0.5, 0.915, "Table 2  ·  Lower is better  ·  — = not evaluated",
            ha="center", va="top", fontsize=10, color="#6b7280", transform=ax.transAxes)

    for j, col in enumerate(MATRIX_COLUMNS):
        x = left + label_w + j * cell_w + cell_w / 2
        ax.text(x, top - header_h / 2, col, ha="center", va="center",
                fontsize=10.5, fontweight="bold", color="#111827", transform=ax.transAxes)

    for i, (abbr, values) in enumerate(MATRIX_ROWS):
        row_y = top - header_h - i * cell_h - cell_h / 2
        info = DEFENSE_COLORS[abbr]
        ax.add_patch(
            mpatches.FancyBboxPatch(
                (left, row_y - cell_h / 2 + 0.002), label_w - 0.008, cell_h - 0.004,
                boxstyle="round,pad=0.004,rounding_size=0.012", transform=ax.transAxes,
                facecolor=info["bg"], edgecolor=info["edge"], linewidth=1.5, zorder=2,
            )
        )
        ax.text(left + (label_w - 0.008) / 2, row_y, abbr, ha="center", va="center",
                fontsize=12, fontweight="bold", color=info["text"],
                transform=ax.transAxes, zorder=3)

        for j, value in enumerate(values):
            cx = left + label_w + j * cell_w
            if value is None:
                style = CELL_STYLE["na"]
                text = "—"
            else:
                style = _style_for((abbr, j), MATRIX_HIGHLIGHTS)
                text = f"{value:.1f}"
                if abbr == "AS" and j == 3:
                    text += " ↑"
            _draw_cell(ax, cx, row_y, cell_w, cell_h, text, style,
                       fontsize=11 if style["weight"] == "bold" else 10.5)

    _draw_legend(ax, y_base=0.075)
    ax.text(0.5, 0.018,
            "PD → GCG 3.6% (best)  ·  SR/PD → RP (2c) 0%  ·  HS → GCG 75.4% (best mechanistic)",
            ha="center", va="bottom", fontsize=9, color="#6b7280", style="italic",
            transform=ax.transAxes)

    fig.savefig(output, bbox_inches="tight", facecolor="white", pad_inches=0.1)
    plt.close(fig)


def main() -> None:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    out1 = DOCS_DIR / "defense_results_detailed.svg"
    out2 = DOCS_DIR / "defense_results_asr_matrix.svg"
    plot_detailed_results(out1)
    plot_asr_matrix(out2)
    print(f"Wrote {out1}")
    print(f"Wrote {out2}")


if __name__ == "__main__":
    main()
