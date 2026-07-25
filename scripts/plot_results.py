"""Generate project result figures and CSV summaries.

The script uses only the Python standard library so it can run locally or on HPC
without extra plotting dependencies.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FIGURE_DIR = ROOT / "figures"
ROLEPLAY_SUMMARY_PATH = FIGURE_DIR / "roleplay_defense_summary.csv"
ROLEPLAY_SVG_PATH = FIGURE_DIR / "roleplay_defense_comparison.svg"
GCG_SUMMARY_PATH = FIGURE_DIR / "gcg_suffix_bank_ablation_summary.csv"
GCG_MATRIX_PATH = FIGURE_DIR / "gcg_suffix_bank_transition_matrices.csv"
GCG_SVG_PATH = FIGURE_DIR / "gcg_suffix_bank_ablation.svg"


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


GCG_METRICS = {
    "GCG suffix bank": ROOT
    / "result_from_hpc_gcg_ablation"
    / "gcg_suffix_bank_transfer_3c_promotion_heldout500_cm_metrics.json",
    "Fixed RolePlay suffix": ROOT
    / "result_from_hpc_gcg_ablation"
    / "gcg_fixed_roleplay_no_bank_3c_heldout500_cm_metrics.json",
    "ParaphraseDefense": ROOT
    / "result_from_hpc_gcg_ablation"
    / "gcg_suffix_bank_defense_paraphrase_heldout500_metrics.json",
}


def _load_json(path: Path) -> dict | None:
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_roleplay_if_available(path: Path | None, setting: str) -> dict | None:
    if not path:
        return None
    data = _load_json(path)
    if not data:
        return None
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


def _resolved_roleplay_results() -> list[dict]:
    rows = list(BINARY_RESULTS)
    for i, row in enumerate(rows):
        loaded = _load_roleplay_if_available(METRIC_OVERRIDES.get(row["setting"]), row["setting"])
        if loaded:
            loaded["note"] = row["note"]
            rows[i] = loaded
    return rows


def write_roleplay_csv(rows: list[dict]) -> None:
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
    with ROLEPLAY_SUMMARY_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _bar(x: float, y: float, width: float, value: float, max_value: float, fill: str) -> str:
    height = 280 * value / max_value if max_value else 0
    return (
        f'<rect x="{x:.1f}" y="{y + 280 - height:.1f}" '
        f'width="{width:.1f}" height="{height:.1f}" fill="{fill}" rx="3" />'
    )


def write_roleplay_svg(rows: list[dict]) -> None:
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
    ROLEPLAY_SVG_PATH.write_text("\n".join(parts), encoding="utf-8")


def _gcg_rows() -> list[dict]:
    bank = _load_json(GCG_METRICS["GCG suffix bank"]) or {}
    fixed = _load_json(GCG_METRICS["Fixed RolePlay suffix"]) or {}
    defense = _load_json(GCG_METRICS["ParaphraseDefense"]) or {}
    return [
        {
            "setting": "Fixed RolePlay suffix",
            "samples": fixed.get("total_samples", 0),
            "bank_size": fixed.get("bank_size", 1),
            "project_asr": fixed.get("project_asr", 0.0),
            "promotion_asr": fixed.get("promotion_asr", 0.0),
            "clean_qwk_retention": "",
            "note": "No-bank ablation; one fixed RolePlay-style suffix",
        },
        {
            "setting": "GCG suffix bank",
            "samples": bank.get("total_samples", 0),
            "bank_size": bank.get("bank_size", 0),
            "project_asr": bank.get("project_asr", 0.0),
            "promotion_asr": bank.get("promotion_asr", 0.0),
            "clean_qwk_retention": "",
            "note": "Five optimized suffixes tried on the same heldout pool",
        },
        {
            "setting": "GCG bank + ParaphraseDefense",
            "samples": defense.get("total_samples", 0),
            "bank_size": defense.get("bank_size", 0),
            "project_asr": defense.get("defended_project_asr", 0.0),
            "promotion_asr": defense.get("defended_promotion_asr", 0.0),
            "clean_qwk_retention": defense.get("clean_qwk_retention", 0.0),
            "note": "Same GCG bank under ParaphraseDefense",
        },
    ]


def write_gcg_csv(rows: list[dict]) -> None:
    fields = [
        "setting",
        "samples",
        "bank_size",
        "project_asr",
        "promotion_asr",
        "clean_qwk_retention",
        "note",
    ]
    with GCG_SUMMARY_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    matrix_fields = ["experiment", "matrix", "row_label", "correct", "contradictory", "incorrect"]
    with GCG_MATRIX_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=matrix_fields)
        writer.writeheader()
        for experiment, path in GCG_METRICS.items():
            data = _load_json(path) or {}
            for matrix_name in ["cm_clean", "cm_attack_best", "transition_clean_to_attack"]:
                matrix = data.get(matrix_name)
                if not isinstance(matrix, dict):
                    continue
                for row_label, row in matrix.items():
                    writer.writerow({
                        "experiment": experiment,
                        "matrix": matrix_name,
                        "row_label": row_label,
                        "correct": row.get("correct", 0),
                        "contradictory": row.get("contradictory", 0),
                        "incorrect": row.get("incorrect", 0),
                    })


def write_gcg_svg(rows: list[dict]) -> None:
    width = 980
    height = 600
    left = 86
    top = 112
    chart_h = 300
    group_w = 260
    bar_w = 52
    max_value = 1.05

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff" />',
        '<text x="80" y="45" font-family="Arial, sans-serif" font-size="24" font-weight="700" fill="#172026">GCG Suffix-Bank Ablation</text>',
        '<text x="80" y="75" font-family="Arial, sans-serif" font-size="14" fill="#52616b">Three-class SciEntsBank heldout500; source samples used to build the suffix bank are excluded.</text>',
        '<line x1="86" y1="412" x2="900" y2="412" stroke="#24343d" stroke-width="1.2" />',
        '<line x1="86" y1="112" x2="86" y2="412" stroke="#24343d" stroke-width="1.2" />',
    ]

    for tick in range(6):
        value = tick / 5
        y = top + chart_h - chart_h * value / max_value
        parts.append(f'<line x1="80" y1="{y:.1f}" x2="900" y2="{y:.1f}" stroke="#e5e9ec" stroke-width="1" />')
        parts.append(f'<text x="42" y="{y + 4:.1f}" font-family="Arial, sans-serif" font-size="12" fill="#5b6970">{value:.1f}</text>')

    for idx, row in enumerate(rows):
        x0 = left + 80 + idx * group_w
        parts.append(_bar(x0, top, bar_w, row["project_asr"], max_value, "#c7534f"))
        parts.append(_bar(x0 + 66, top, bar_w, row["promotion_asr"], max_value, "#3f7f93"))
        parts.append(
            f'<text x="{x0 + 58:.1f}" y="444" text-anchor="middle" '
            'font-family="Arial, sans-serif" font-size="13" fill="#172026">'
            f'{row["setting"]}</text>'
        )
        parts.append(
            f'<text x="{x0 + 58:.1f}" y="466" text-anchor="middle" '
            'font-family="Arial, sans-serif" font-size="11" fill="#6b7780">'
            f'n={row["samples"]}, suffixes={row["bank_size"]}</text>'
        )
        for offset, key in [(26, "project_asr"), (92, "promotion_asr")]:
            value = row[key]
            label_y = top + chart_h - chart_h * value / max_value - 8
            parts.append(
                f'<text x="{x0 + offset:.1f}" y="{label_y:.1f}" text-anchor="middle" '
                'font-family="Arial, sans-serif" font-size="11" fill="#172026">'
                f'{value * 100:.1f}%</text>'
            )

    parts.extend(
        [
            '<rect x="80" y="510" width="16" height="16" fill="#c7534f" rx="3" />',
            '<text x="104" y="523" font-family="Arial, sans-serif" font-size="13" fill="#172026">Project ASR: eligible non-correct -> correct</text>',
            '<rect x="385" y="510" width="16" height="16" fill="#3f7f93" rx="3" />',
            '<text x="409" y="523" font-family="Arial, sans-serif" font-size="13" fill="#172026">Promotion ASR: any grade relaxation</text>',
            '<text x="80" y="558" font-family="Arial, sans-serif" font-size="12" fill="#52616b">Fixed suffix reaches 0% strict ASR; the optimized suffix bank reaches 99.6%; ParaphraseDefense lowers it to 36.8%.</text>',
            "</svg>",
        ]
    )
    GCG_SVG_PATH.write_text("\n".join(parts), encoding="utf-8")


def main() -> None:
    FIGURE_DIR.mkdir(exist_ok=True)
    roleplay_rows = _resolved_roleplay_results()
    write_roleplay_csv(roleplay_rows)
    write_roleplay_svg(roleplay_rows)

    gcg_rows = _gcg_rows()
    write_gcg_csv(gcg_rows)
    write_gcg_svg(gcg_rows)

    print(f"Wrote {ROLEPLAY_SUMMARY_PATH.relative_to(ROOT)}")
    print(f"Wrote {ROLEPLAY_SVG_PATH.relative_to(ROOT)}")
    print(f"Wrote {GCG_SUMMARY_PATH.relative_to(ROOT)}")
    print(f"Wrote {GCG_MATRIX_PATH.relative_to(ROOT)}")
    print(f"Wrote {GCG_SVG_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()