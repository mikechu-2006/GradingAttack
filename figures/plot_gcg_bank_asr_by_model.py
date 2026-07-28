"""
Plot: GCG Suffix Bank vs Injection Attack ASR Across Models by Release Time
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from datetime import datetime
from matplotlib.lines import Line2D

# ── Data ─────────────────────────────────────────────────────────────────
models = [
    # (model,           release,         GCG_ASR,  INJ_ASR, short_label)
    ("Mistral-7B-Instruct-v0.3",    datetime(2024, 5, 1),   70.8,  28.2, "Mistral-7B"),
    ("Llama-3.1-8B-Instruct",       datetime(2024, 7, 23), 100.0,  91.36, "Llama-3.1-8B"),
    ("Qwen2.5-7B-Instruct",         datetime(2024, 9, 19),  23.3,   2.63, "Qwen2.5-7B"),
    ("Qwen3-4B-Instruct-2507",      datetime(2025, 4, 29),   0.0,   0.00, "Qwen3-4B"),
    ("Qwen3.5-4B",                  datetime(2026, 2, 1),    0.0,   4.17, "Qwen3.5-4B"),
]

names  = [m[0] for m in models]
dates  = [m[1] for m in models]
gcg    = [m[2] for m in models]
inj    = [m[3] for m in models]
short  = [m[4] for m in models]

# ── Colours by model family ──────────────────────────────────────────────
family_colour = {"Mistral": "#E8694A", "Llama": "#2563EB", "Qwen": "#7C3AED"}
gcg_colours = []
inj_colours = []
for n in names:
    c = family_colour["Mistral"] if "Mistral" in n else \
        family_colour["Llama"] if "Llama" in n else \
        family_colour["Qwen"]
    gcg_colours.append(c)
    inj_colours.append(c)

# ── Figure ───────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5.5))

# ── GCG line (solid) ─────────────────────────────────────────────────────
ax.plot(dates, gcg, color="#2563EB", linestyle="-", linewidth=2.0,
        zorder=4, label="GCG Suffix Bank Attack")
ax.scatter(dates, gcg, color=gcg_colours, s=140, zorder=5,
           edgecolors="white", linewidth=1.5, marker="o")

for d, v, s in zip(dates, gcg, short):
    ax.annotate(f"{s}\n{v:.1f}%", (d, v), textcoords="offset points",
                xytext=(0, 18), ha="center", fontsize=8.5, fontweight="bold",
                color="#1D4ED8",
                arrowprops=dict(arrowstyle="->", color="#2563EB", lw=0.8))

# ── Injection line (dashed) ──────────────────────────────────────────────
ax.plot(dates, inj, color="#DC2626", linestyle="--", linewidth=2.0,
        zorder=4, label="Injection Attack")
ax.scatter(dates, inj, color=inj_colours, s=140, zorder=5,
           edgecolors="white", linewidth=1.5, marker="s")

for d, v, s in zip(dates, inj, short):
    ax.annotate(f"{s}\n{v:.1f}%", (d, v), textcoords="offset points",
                xytext=(0, -22), ha="center", fontsize=8.5, fontweight="bold",
                color="#991B1B",
                arrowprops=dict(arrowstyle="->", color="#DC2626", lw=0.8))

# ── Axis labels & styling ────────────────────────────────────────────────
ax.set_ylabel("Attack Success Rate (ASR, %)", fontsize=12, fontweight="bold")
ax.set_xlabel("Model Release Date", fontsize=12, fontweight="bold")
ax.set_title(
    "GCG Suffix Bank vs Injection Attack — ASR Across Models",
    fontsize=13, fontweight="bold", pad=12,
)

ax.set_ylim(-10, 115)
ax.yaxis.set_major_locator(plt.MultipleLocator(20))
ax.yaxis.set_minor_locator(plt.MultipleLocator(10))
ax.axhline(y=0, color="#D1D5DB", linewidth=0.8)
ax.axhline(y=100, color="#D1D5DB", linewidth=0.8)

loc = mdates.MonthLocator(bymonth=[1, 5, 9])
ax.xaxis.set_major_locator(loc)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
ax.xaxis.set_minor_locator(mdates.MonthLocator())

ax.grid(True, axis="y", alpha=0.3, linestyle="--")
ax.grid(True, axis="x", which="minor", alpha=0.1)

# ── Legend ────────────────────────────────────────────────────────────────
legend_elements = [
    Line2D([0], [0], color="#2563EB", linewidth=2, marker="o",
           markersize=8, markerfacecolor="#2563EB", label="GCG Suffix Bank"),
    Line2D([0], [0], color="#DC2626", linewidth=2, linestyle="--", marker="s",
           markersize=8, markerfacecolor="#DC2626", label="Injection Attack"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor="#E8694A",
           markersize=9, label="Mistral"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor="#2563EB",
           markersize=9, label="Meta (Llama)"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor="#7C3AED",
           markersize=9, label="Alibaba (Qwen)"),
]
ax.legend(handles=legend_elements, loc="upper right", fontsize=8.5,
          framealpha=0.9, ncol=2)

# ── Footnote ─────────────────────────────────────────────────────────────
fig.text(
    0.5, 0.01,
    "GCG: 5-suffix bank built via GCG optimization on each model individually.  "
    "Injection: average across AO/DC/IM for Llama; RolePlay-only for Qwen;\n"
    "average across AO/DC/IM/RolePlay for Mistral.  "
    "All results are undefended, 2-class unless noted.",
    ha="center", fontsize=7.5, color="#6B7280",
    transform=fig.transFigure,
)

plt.tight_layout(rect=[0, 0.10, 1, 1])
fig.savefig("/home/mikechu/Documents/cources/4313/project/GradingAttack/figures/gcg_bank_asr_by_model.svg",
            dpi=200, bbox_inches="tight")
fig.savefig("/home/mikechu/Documents/cources/4313/project/GradingAttack/figures/gcg_bank_asr_by_model.png",
            dpi=200, bbox_inches="tight")

print("Saved: figures/gcg_bank_asr_by_model.svg + .png")

# ── Print combined table ─────────────────────────────────────────────────
print()
print("=" * 95)
print("TABLE: Attack ASR Across Models (ordered by release date)")
print("=" * 95)
print(f"{'Model':<30} {'Release':<12} {'GCG Bank ASR':<15} {'Injection ASR':<15} {'Family':<10}")
print("-" * 95)
for name, date, g, i, s in models:
    family = "Mistral" if "Mistral" in name else ("Llama" if "Llama" in name else "Qwen")
    date_str = date.strftime("%b %Y")
    print(f"{name:<30} {date_str:<12} {g:<14.1f}% {i:<14.2f}% {family:<10}")
print("=" * 95)
