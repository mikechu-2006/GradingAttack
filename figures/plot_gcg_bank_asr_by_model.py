"""
Plot: GCG Suffix Bank Attack ASR Across Models by Release Time

Data sources:
  - Llama-3.1-8B-Instruct: result_from_hpc_gcg_ablation/gcg_suffix_bank_transfer_3c_promotion_heldout500_cm_metrics.json
  - Mistral-7B-Instruct-v0.3: shared_results/.../mistral_bank_and_hs/gcg_suffix_bank_mistral_50_full.jsonl
  - Qwen3.5-4B: shared_results/.../qwen_suffix_banks/gcg_suffix_bank_qwen35_4b_50_full.jsonl

Only models with a no-defense GCG suffix bank evaluation are plotted.
Qwen2.5-7B and Qwen3-4B have only defense-on runs (no undefended baseline).
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from datetime import datetime

# ── Data: model → (release_date, asr, n, note) ──────────────────────────
# Release dates are approximate (month-level precision is sufficient)
models = [
    # (model_name, release_date, asr_pct, n_samples, bank_size, class_count, note)
    ("Mistral-7B-Instruct-v0.3",    datetime(2024, 5, 1),   18.92, 50,   5, "2c", "reconstructed bank"),
    ("Llama-3.1-8B-Instruct",       datetime(2024, 7, 23),  99.63, 500,  5, "3c", "definitive result"),
    ("Qwen2.5-7B-Instruct",         datetime(2024, 9, 19),  None,  None, None, None, "no undefended data"),
    ("Qwen3-4B-Instruct-2507",      datetime(2025, 4, 29),  None,  None, None, None, "no undefended data"),
    ("Qwen3.5-4B",                  datetime(2026, 3, 1),   0.00,  50,   5, "2c", "reconstructed bank"),
]

# ── Build arrays (excluding models without data) ─────────────────────────
has_data = [(name, date, asr, n, bs, cls, note)
            for name, date, asr, n, bs, cls, note in models
            if asr is not None]

names_plot  = [m[0] for m in has_data]
dates_plot  = [m[1] for m in has_data]
asrs_plot   = [m[2] for m in has_data]
ns_plot     = [m[3] for m in has_data]
notes_plot  = [m[5] for m in has_data]  # class count
bank_sizes  = [m[4] for m in has_data]

# ── Figure ───────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5.5))

# Colour by model family
colours = []
for n in names_plot:
    if "Mistral" in n:
        colours.append("#E8694A")   # Mistral orange
    elif "Llama" in n:
        colours.append("#2563EB")   # Meta blue
    elif "Qwen" in n:
        colours.append("#7C3AED")   # Qwen purple
    else:
        colours.append("#6B7280")

# Scatter points
scatter = ax.scatter(dates_plot, asrs_plot, c=colours, s=200, zorder=5, edgecolors="white", linewidth=1.5)

# Line connecting points
ax.plot(dates_plot, asrs_plot, color="#9CA3AF", linestyle="--", linewidth=1.5, alpha=0.7, zorder=3)

# Annotate points
for i, (name, date, asr, n, cls_label, note) in enumerate(zip(names_plot, dates_plot, asrs_plot, ns_plot, notes_plot, notes_plot)):
    short = name.replace("-Instruct", "").replace("-v0.3", "")
    label = f"{short}\n{asr:.1f}% ASR (n={n}, {cls_label})"
    ax.annotate(label, (date, asr),
                textcoords="offset points", xytext=(0, 18),
                ha="center", fontsize=9, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="#666666", lw=0.8))

# Mark models without data with faint "N/A" markers
no_data = [(name, date) for name, date, asr, *_ in models if asr is None]
for name, date in no_data:
    short = name.replace("-Instruct", "").replace("-2507", "")
    ax.scatter([date], [50], marker="x", s=120, color="#D1D5DB", zorder=2, linewidths=2)
    ax.annotate(f"{short}\nN/A", (date, 50),
                textcoords="offset points", xytext=(0, 15),
                ha="center", fontsize=8, color="#9CA3AF", fontstyle="italic")

# Axis labels & styling
ax.set_ylabel("GCG Suffix Bank Attack ASR (%)", fontsize=12, fontweight="bold")
ax.set_xlabel("Model Release Date", fontsize=12, fontweight="bold")
ax.set_title("GCG Suffix Bank Attack Transferability Across Models\n(5-Suffix Bank, No Defense)",
             fontsize=13, fontweight="bold", pad=12)

ax.set_ylim(-5, 110)
ax.yaxis.set_major_locator(plt.MultipleLocator(20))
ax.yaxis.set_minor_locator(plt.MultipleLocator(10))
ax.axhline(y=0, color="#D1D5DB", linewidth=0.8)
ax.axhline(y=100, color="#D1D5DB", linewidth=0.8)

# Date formatting
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 5, 9]))
ax.xaxis.set_minor_locator(mdates.MonthLocator())

ax.grid(True, axis="y", alpha=0.3, linestyle="--")
ax.grid(True, axis="x", which="minor", alpha=0.1)

# Legend for families
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker="o", color="w", markerfacecolor="#E8694A", markersize=10, label="Mistral"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor="#2563EB", markersize=10, label="Meta (Llama)"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor="#7C3AED", markersize=10, label="Alibaba (Qwen)"),
    Line2D([0], [0], marker="x", color="#D1D5DB", markersize=8, markeredgewidth=2, label="No data available"),
]
ax.legend(handles=legend_elements, loc="lower left", fontsize=9, framealpha=0.9)

# Footnote
note_text = (
    "Notes:\n"
    "• Llama-3.1: n=500, 3-class grading, definitive heldout evaluation\n"
    "• Mistral / Qwen3.5: n=50, bank reconstructed per-model; smaller sample\n"
    "• Qwen2.5-7B / Qwen3-4B: no undefended GCG suffix bank evaluation exists\n"
    "• All experiments use a 5-suffix bank; bank_size=1 yields 0% ASR (see ablation)"
)
fig.text(0.5, 0.01, note_text, ha="center", fontsize=8, color="#6B7280",
         transform=fig.transFigure)

plt.tight_layout(rect=[0, 0.12, 1, 1])
fig.savefig("/home/mikechu/Documents/cources/4313/project/GradingAttack/figures/gcg_bank_asr_by_model.svg",
            dpi=200, bbox_inches="tight")
fig.savefig("/home/mikechu/Documents/cources/4313/project/GradingAttack/figures/gcg_bank_asr_by_model.png",
            dpi=200, bbox_inches="tight")

print("Saved: figures/gcg_bank_asr_by_model.svg + .png")

# ── Also print the table ────────────────────────────────────────────────
print("\n" + "=" * 100)
print("TABLE: GCG Suffix Bank Attack ASR Across Models (ordered by release date)")
print("=" * 100)
print(f"{'Model':<32} {'Release':<12} {'ASR':<10} {'n':<8} {'Bank':<8} {'Classes':<10} {'Note'}")
print("-" * 100)
for name, date, asr, n, bs, cls_note, note in models:
    date_str = date.strftime("%b %Y")
    if asr is not None:
        asr_str = f"{asr:.2f}%"
        n_str = str(n)
        bs_str = str(bs)
        cls_str = cls_note
    else:
        asr_str = "—"
        n_str = "—"
        bs_str = "—"
        cls_str = "—"
    short = name[:30]
    print(f"{short:<32} {date_str:<12} {asr_str:<10} {n_str:<8} {bs_str:<8} {cls_str:<10} {note}")
print("=" * 100)
