"""Generate a compact ASR/QWK comparison figure for the current experiments.

The script intentionally uses only the Python standard library so it can run in
the HPC environment without extra plotting dependencies.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FIGURE_DIR = ROOT / "figures"
SUMMARY_PATH = FIGURE_DIR / "roleplay_defense_summary.csv"
SVG_PATH = FIGURE_DIR / "roleplay_defense_comparison.svg"


RESULTS = [
    {
        "setting": "No defense",
        "samples": 300,
        "clean_qwk": 0.3227,
        "defended_clean_qwk": 0.3227,
        "asr": 0.1657,
        "defended_asr": 0.1657,
        "note": "Baseline RolePlay vulnerability",
    },
    {
        "setting": "SelfReminder",
        "samples": 300,
        "clean_qwk": 0.3227,
        "defended_clean_qwk": 0.3319,
        "asr": 0.1657,
        "defended_asr": 0.0497,
        "note": "Best robustness-utility tradeoff",
    },
    {
        "setting": "ParaphraseDefense",
        "samples": 300,
        "clean_qwk": 0.3227,
        "defended_clean_qwk": 0.1837,
        "asr": 0.1657,
        "defended_asr": 0.0221,
        "note": "Strongest ASR reduction, large QWK drop",
    },
    {
        "setting": "SmoothLLM",
        "samples": 50,
        "clean_qwk": 0.4444,
        "defended_clean_qwk": 0.3026,
        "asr": 0.1667,
        "defended_asr": 0.1667,
        "note": "Smoke test only; no ASR improvement",
    },    {
        "setting": "HijackingSuppression",
        "samples": 300,
        "clean_qwk": 0.3136,
        "defended_clean_qwk": 0.2753,
        "asr": 0.0055,
        "defended_asr": 0.0166,
        "note": "Attention-based exploratory run; no ASR improvement",
    },
]


def _load_if_available(path: Path | None, setting: str) -> dict | None:
    if not path or not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return {
        "setting": setting,
        "samples": data.get("total", 0),
        "clean_qwk": data["qwk_clean"],
        "defended_clean_qwk": data["qwk_defense_clean"],
        "asr": data["asr"],
        "defended_asr": data["asr_defended"],
        "note": "Loaded from metrics JSON",
    }


def _resolved_results() -> list[dict]:
    results = list(RESULTS)

    overrides = {
        "No defense": ROOT
        / "result_from_hpc_roleplay"
        / "RolePlay-Llama-3.1-8B-Instruct-nodefense_202607231552_metrics.json",
        "SelfReminder": ROOT
        / "result_from_hpc_self_reminder"
        / "RolePlay-Llama-3.1-8B-Instruct_202607231459_metrics.json",
    }
    for i, row in enumerate(results):
        loaded = _load_if_available(overrides.get(row["setting"]), row["setting"])
        if loaded:
            loaded["note"] = row["note"]
            results[i] = loaded
    return results


def write_csv(rows: list[dict]) -> None:
    FIGURE_DIR.mkdir(exist_ok=True)
    fields = [
        "setting",
        "samples",
        "clean_qwk",
        "defended_clean_qwk",
        "asr",
        "defended_asr",
        "note",
    ]
    with SUMMARY_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _bar(x: float, y: float, width: float, value: float, max_value: float, fill: str) -> str:
    height = 260 * value / max_value if max_value else 0
    return (
        f'<rect x="{x:.1f}" y="{y + 260 - height:.1f}" '
        f'width="{width:.1f}" height="{height:.1f}" fill="{fill}" rx="3" />'
    )


def write_svg(rows: list[dict]) -> None:
    width = 980
    height = 560
    left = 80
    top = 105
    chart_h = 260
    group_w = 200
    bar_w = 42
    max_value = 0.5

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff" />',
        '<text x="80" y="45" font-family="Arial, sans-serif" font-size="24" font-weight="700" fill="#172026">RolePlay Defense Comparison</text>',
        '<text x="80" y="75" font-family="Arial, sans-serif" font-size="14" fill="#52616b">ASR lower is better; clean QWK higher is better.</text>',
        '<line x1="80" y1="365" x2="920" y2="365" stroke="#24343d" stroke-width="1.2" />',
        '<line x1="80" y1="105" x2="80" y2="365" stroke="#24343d" stroke-width="1.2" />',
    ]

    for tick in range(6):
        value = tick / 10
        y = top + chart_h - chart_h * value / max_value
        parts.append(f'<line x1="74" y1="{y:.1f}" x2="920" y2="{y:.1f}" stroke="#e5e9ec" stroke-width="1" />')
        parts.append(f'<text x="38" y="{y + 4:.1f}" font-family="Arial, sans-serif" font-size="12" fill="#5b6970">{value:.1f}</text>')

    for idx, row in enumerate(rows):
        x0 = left + 45 + idx * group_w
        parts.append(_bar(x0, top, bar_w, row["defended_asr"], max_value, "#d95f59"))
        parts.append(_bar(x0 + 54, top, bar_w, row["defended_clean_qwk"], max_value, "#3576b8"))
        parts.append(
            f'<text x="{x0 + 48:.1f}" y="395" text-anchor="middle" '
            'font-family="Arial, sans-serif" font-size="13" fill="#172026">'
            f'{row["setting"]}</text>'
        )
        parts.append(
            f'<text x="{x0 + 21:.1f}" y="{top + chart_h - chart_h * row["defended_asr"] / max_value - 8:.1f}" '
            'text-anchor="middle" font-family="Arial, sans-serif" font-size="12" fill="#172026">'
            f'{row["defended_asr"]:.3f}</text>'
        )
        parts.append(
            f'<text x="{x0 + 75:.1f}" y="{top + chart_h - chart_h * row["defended_clean_qwk"] / max_value - 8:.1f}" '
            'text-anchor="middle" font-family="Arial, sans-serif" font-size="12" fill="#172026">'
            f'{row["defended_clean_qwk"]:.3f}</text>'
        )
        parts.append(
            f'<text x="{x0 + 48:.1f}" y="418" text-anchor="middle" '
            'font-family="Arial, sans-serif" font-size="11" fill="#6b7780">'
            f'n={row["samples"]}</text>'
        )

    parts.extend(
        [
            '<rect x="80" y="462" width="16" height="16" fill="#d95f59" rx="3" />',
            '<text x="104" y="475" font-family="Arial, sans-serif" font-size="13" fill="#172026">Defended ASR</text>',
            '<rect x="220" y="462" width="16" height="16" fill="#3576b8" rx="3" />',
            '<text x="244" y="475" font-family="Arial, sans-serif" font-size="13" fill="#172026">Defended clean QWK</text>',
            '<text x="80" y="515" font-family="Arial, sans-serif" font-size="12" fill="#52616b">Source: RESULTS.md and tracked metrics JSON files. SmoothLLM is a 50-sample smoke test.</text>',
            "</svg>",
        ]
    )
    SVG_PATH.write_text("\n".join(parts), encoding="utf-8")


def main() -> None:
    rows = _resolved_results()
    write_csv(rows)
    write_svg(rows)
    print(f"Wrote {SUMMARY_PATH.relative_to(ROOT)}")
    print(f"Wrote {SVG_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

