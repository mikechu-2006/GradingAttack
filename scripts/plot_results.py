"""Generate the main binary RolePlay defense comparison figure.

The script uses only the Python standard library so it can run locally or on HPC
without additional plotting dependencies.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FIGURE_DIR = ROOT / "figures"
SUMMARY_PATH = FIGURE_DIR / "roleplay_defense_summary.csv"
SVG_PATH = FIGURE_DIR / "roleplay_defense_comparison.svg"


BINARY_RESULTS = [
    {
        "setting": "No defense",
        "samples": 300,
        "clean_qwk": 0.4158,
        "attack_qwk": 0.1251,
        "defended_clean_qwk": 0.4158,
        "defended_attack_qwk": 0.1251,
        "asr": 0.0055,
        "defended_asr": 0.0055,
        "note": "Binary baseline; RolePlay mainly lowers QWK",
    },
    {
        "setting": "SelfReminder",
        "samples": 300,
        "clean_qwk": 0.4158,
        "attack_qwk": 0.1251,
        "defended_clean_qwk": 0.4449,
        "defended_attack_qwk": 0.0701,
        "asr": 0.0055,
        "defended_asr": 0.0,
        "note": "Best clean-utility preservation",
    },
    {
        "setting": "ParaphraseDefense",
        "samples": 300,
        "clean_qwk": 0.4158,
        "attack_qwk": 0.1251,
        "defended_clean_qwk": 0.1734,
        "defended_attack_qwk": 0.1383,
        "asr": 0.0055,
        "defended_asr": 0.0,
        "note": "Removes ASR but large clean-QWK drop",
    },
]


METRIC_OVERRIDES = {
    "No defense": ROOT
    / "result_from_hpc_roleplay_2c"
    / "RolePlay-Llama-3.1-8B-Instruct-nodefense-2c-300_202607241318_metrics.json",
    "SelfReminder": ROOT
    / "result_from_hpc_roleplay_2c"
    / "RolePlay-Llama-3.1-8B-Instruct-self-reminder-2c-300_202607241322_metrics.json",
    "ParaphraseDefense": ROOT
    / "result_from_hpc_roleplay_2c"
    / "RolePlay-Llama-3.1-8B-Instruct-paraphrase-2c-300_202607241330_metrics.json",
}


def _load_if_available(path: Path | None, setting: str) -> dict | None:
    if not path or not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return {
        "setting": setting,
        "samples": data.get("total", 0),
        "clean_qwk": data["qwk_clean"],
        "attack_qwk": data["qwk_attack"],
        "defended_clean_qwk": data["qwk_defense_clean"],
        "defended_attack_qwk": data["qwk_defense_attack"],
        "asr": data["asr"],
        "defended_asr": data["asr_defended"],
        "note": "Loaded from binary metrics JSON",
    }


def _resolved_results() -> list[dict]:
    rows = list(BINARY_RESULTS)
    for i, row in enumerate(rows):
        loaded = _load_if_available(METRIC_OVERRIDES.get(row["setting"]), row["setting"])
        if loaded:
            loaded["note"] = row["note"]
            rows[i] = loaded
    return rows


def write_csv(rows: list[dict]) -> None:
    FIGURE_DIR.mkdir(exist_ok=True)
    fields = [
        "setting",
        "samples",
        "clean_qwk",
        "attack_qwk",
        "defended_clean_qwk",
        "defended_attack_qwk",
        "asr",
        "defended_asr",
        "note",
    ]
    with SUMMARY_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _bar(x: float, y: float, width: float, value: float, max_value: float, fill: str) -> str:
    height = 280 * value / max_value if max_value else 0
    return (
        f'<rect x="{x:.1f}" y="{y + 280 - height:.1f}" '
        f'width="{width:.1f}" height="{height:.1f}" fill="{fill}" rx="3" />'
    )


def write_svg(rows: list[dict]) -> None:
    width = 960
    height = 590
    left = 80
    top = 105
    chart_h = 280
    group_w = 235
    bar_w = 38
    max_value = 0.5

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff" />',
        '<text x="80" y="45" font-family="Arial, sans-serif" font-size="24" font-weight="700" fill="#172026">Binary RolePlay Defense Comparison</text>',
        '<text x="80" y="75" font-family="Arial, sans-serif" font-size="14" fill="#52616b">QWK higher is better; ASR lower is better. Main setting: correct vs. incorrect.</text>',
        '<line x1="80" y1="385" x2="880" y2="385" stroke="#24343d" stroke-width="1.2" />',
        '<line x1="80" y1="105" x2="80" y2="385" stroke="#24343d" stroke-width="1.2" />',
    ]

    for tick in range(6):
        value = tick / 10
        y = top + chart_h - chart_h * value / max_value
        parts.append(f'<line x1="74" y1="{y:.1f}" x2="880" y2="{y:.1f}" stroke="#e5e9ec" stroke-width="1" />')
        parts.append(f'<text x="38" y="{y + 4:.1f}" font-family="Arial, sans-serif" font-size="12" fill="#5b6970">{value:.1f}</text>')

    for idx, row in enumerate(rows):
        x0 = left + 65 + idx * group_w
        parts.append(_bar(x0, top, bar_w, row["attack_qwk"], max_value, "#8c6bb1"))
        parts.append(_bar(x0 + 48, top, bar_w, row["defended_attack_qwk"], max_value, "#3576b8"))
        parts.append(_bar(x0 + 96, top, bar_w, row["defended_asr"], max_value, "#d95f59"))
        parts.append(
            f'<text x="{x0 + 67:.1f}" y="415" text-anchor="middle" '
            'font-family="Arial, sans-serif" font-size="13" fill="#172026">'
            f'{row["setting"]}</text>'
        )
        parts.append(
            f'<text x="{x0 + 67:.1f}" y="438" text-anchor="middle" '
            'font-family="Arial, sans-serif" font-size="11" fill="#6b7780">'
            f'n={row["samples"]}</text>'
        )
        for offset, key in [(19, "attack_qwk"), (67, "defended_attack_qwk"), (115, "defended_asr")]:
            value = row[key]
            label_y = top + chart_h - chart_h * value / max_value - 8
            parts.append(
                f'<text x="{x0 + offset:.1f}" y="{label_y:.1f}" text-anchor="middle" '
                'font-family="Arial, sans-serif" font-size="11" fill="#172026">'
                f'{value:.3f}</text>'
            )

    parts.extend(
        [
            '<rect x="80" y="482" width="16" height="16" fill="#8c6bb1" rx="3" />',
            '<text x="104" y="495" font-family="Arial, sans-serif" font-size="13" fill="#172026">Attack QWK</text>',
            '<rect x="205" y="482" width="16" height="16" fill="#3576b8" rx="3" />',
            '<text x="229" y="495" font-family="Arial, sans-serif" font-size="13" fill="#172026">Defended attack QWK</text>',
            '<rect x="390" y="482" width="16" height="16" fill="#d95f59" rx="3" />',
            '<text x="414" y="495" font-family="Arial, sans-serif" font-size="13" fill="#172026">Defended ASR</text>',
            '<text x="80" y="538" font-family="Arial, sans-serif" font-size="12" fill="#52616b">Source: binary 300-sample RolePlay metrics from result_from_hpc_roleplay_2c.</text>',
            '<text x="80" y="558" font-family="Arial, sans-serif" font-size="12" fill="#52616b">RolePlay mainly lowers QWK in binary grading; ASR is already low.</text>',
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