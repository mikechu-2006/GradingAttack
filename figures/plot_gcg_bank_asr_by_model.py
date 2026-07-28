"""
Plot: GCG Suffix Bank Attack ASR Across Models by Release Time

Correct data (provided by user):
  - Mistral-7B-Instruct-v0.3 (May 2024):  ASR = 70.8%
  - Llama-3.1-8B-Instruct       (Jul 2024):  ASR = 100%
  - Qwen2.5-7B-Instruct        (Sep 2024):  ASR = 23.3%
  - Qwen3-4B-Instruct-2507     (Apr 2025):  ASR = 0%
  - Qwen3.5-4B                 (Feb 2026):  ASR = 0%

Each model had its own 5-suffix bank built via GCG optimization on
that model itself, then evaluated on heldout samples (no defense).
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from datetime import datetime

# ── Data ─────────────────────────────────────────────────────────────────
models = [
    # (model,           release,         ASR,  short_label)
    ("Mistral-7B-Instruct-v0.3",    datetime(2024, 5, 1),   70.8, "Mistral-7B"),
    ("Llama-3.1-8B-Instruct",       datetime(2024, 7, 23), 100.0, "Llama-3.1-8B"),
    ("Qwen2.5-7B-Instruct",         datetime(2024, 9, 19),  23.3, "Qwen2.5-7B"),
    ("Qwen3-4B-Instruct-2507",      datetime(2025, 4, 29),   0.0, "Qwen3-4B"),
    ("Qwen3.5-4B",                  datetime(2026, 2, 1),    0.0, "Qwen3.5-4B"),
]

names       = [m[0] for m in models]
dates       = [m[1] for m in models]
asrs        = [m[2] for m in models]
short_names = [m[3] for m in models]

# ── Colour by model family ───────────────────────────────────────────────
family_colour = {
    "Mistral": "#E8694A",
    "Llama":   "#2563EB",
    "Qwen":    "#7C3AED",
}
colours = []
for n in names:
    if "Mistral" in n:
        colours.append(family_colour["Mistral"])
    elif "Llama" in n:
        colours.append(family_colour["Llama"])
    else:
        colours.append(family_colour["Qwen"])

# ── Figure ───────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5.5))

# Scatter
ax.scatter(dates, asrs, c=colours, s=220, zorder=5,
           edgecolors="white", linewidth=1.5)

# Connecting line
ax.plot(dates, asrs, color="#9CA3AF", linestyle="--",
        linewidth=1.5, alpha=0.7, zorder=3)

# Annotations
for date, asr, short, colour in zip(dates, asrs, short_names, colours):
    label = f"{short}\n{asr:.1f}%"
    if asr == 0:
        label += "  "
    ax.annotate(
        label, (date, asr),
        textcoords="offset points", xytext=(0, 16),
        ha="center", fontsize=9, fontweight="bold",
        color=colour,
        arrowprops=dict(arrowstyle="->", color="#666666", lw=0.8),
    )

# Axis labels & styling
ax.set_ylabel("GCG Suffix Bank ASR (%)", fontsize=12, fontweight="bold")
ax.set_xlabel("Model Release Date", fontsize=12, fontweight="bold")
ax.set_title(
    "GCG Suffix Bank Attack ASR\n"
    "(Per-Model 5-Suffix Bank, No Defense)",
    fontsize=13, fontweight="bold", pad=12,
)

ax.set_ylim(-10, 115)
ax.yaxis.set_major_locator(plt.MultipleLocator(20))
ax.yaxis.set_minor_locator(plt.MultipleLocator(10))
ax.axhline(y=0, color="#D1D5DB", linewidth=0.8)
ax.axhline(y=100, color="#D1D5DB", linewidth=0.8)

# Date axis
loc = mdates.MonthLocator(bymonth=[1, 5, 9])
ax.xaxis.set_major_locator(loc)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
ax.xaxis.set_minor_locator(mdates.MonthLocator())

ax.grid(True, axis="y", alpha=0.3, linestyle="--")
ax.grid(True, axis="x", which="minor", alpha=0.1)

# Gradient fill under the line
ax.fill_between(dates, asrs, 0, alpha=0.08, color="#2563EB")

# Legend
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker="o", color="w", markerfacecolor="#E8694A",
           markersize=10, label="Mistral"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor="#2563EB",
           markersize=10, label="Meta (Llama)"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor="#7C3AED",
           markersize=10, label="Alibaba (Qwen)"),
]
ax.legend(handles=legend_elements, loc="lower left", fontsize=9, framealpha=0.9)

# Footnote
fig.text(
    0.5, 0.01,
    "Each model's ASR was evaluated with a 5-suffix bank built via GCG optimization on that same model.  "
    "Newer models (Qwen3, Qwen3.5) are completely resistant to\n"
    "GCG suffix optimization — no effective adversarial suffix could be found.  "
    "Llama-3.1 is maximally vulnerable (100%).",
    ha="center", fontsize=8, color="#6B7280",
    transform=fig.transFigure,
)

plt.tight_layout(rect=[0, 0.10, 1, 1])
fig.savefig("/home/mikechu/Documents/cources/4313/project/GradingAttack/figures/gcg_bank_asr_by_model.svg",
            dpi=200, bbox_inches="tight")
fig.savefig("/home/mikechu/Documents/cources/4313/project/GradingAttack/figures/gcg_bank_asr_by_model.png",
            dpi=200, bbox_inches="tight")

print("Saved: figures/gcg_bank_asr_by_model.svg + .png")

# ── Print table ──────────────────────────────────────────────────────────
print()
print("=" * 90)
print("TABLE: GCG Suffix Bank ASR Across Models (ordered by release date)")
print("=" * 90)
print(f"{'Model':<30} {'Release':<12} {'ASR':<15} {'Family':<12}")
print("-" * 90)
for name, date, asr, short in models:
    family = "Mistral" if "Mistral" in name else ("Llama" if "Llama" in name else "Qwen")
    date_str = date.strftime("%b %Y")
    asr_str = f"{asr:.1f}%"
    print(f"{name:<30} {date_str:<12} {asr_str:<15} {family:<12}")
print("=" * 90)
