#!/usr/bin/env python3
"""
Hijacking Suppression 注意力劫持防御管线：小样本调参 → 最优参数全量评估。

阶段:
  tune  — beta × layers 网格搜索，每组合 TUNE_SAMPLES 条样本
  full  — 用 tune 阶段最优参数跑 FULL_SAMPLES 条样本
  all   — 依次执行 tune + full（默认）

调参网格:
  beta ∈ {0.05, 0.1, 0.15, 0.2, 0.3}
  layers ∈ {all, L28-31, L24-31, L16-31}
  top_fraction = 0.01（固定）

评分（越小越好）:
  ASR_defended + 0.5 * max(0, qwk_clean - qwk_defense_clean)

输出:
  configs/generated/grid_search_hs/          各阶段生成的 YAML
  configs/RolePlay-...-hijacking-suppression-best-2c.yaml  最优全量配置
  result/grid_search/hijacking_suppression/summary.json    调参汇总
  result/grid_search/hijacking_suppression/full_result.json 全量结果

用法:
  python scripts/hijacking_suppression_pipeline.py
  python scripts/hijacking_suppression_pipeline.py --phase tune
  python scripts/hijacking_suppression_pipeline.py --phase full
  python scripts/hijacking_suppression_pipeline.py --phase full --summary result/grid_search/hijacking_suppression/summary.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from grid_search_roleplay_defense import (  # noqa: E402
    DEFENSES,
    FULL_SAMPLES,
    TUNE_SAMPLES,
    run_defense_pipeline,
)

HS_SPEC = DEFENSES["hijacking_suppression"]
DEFAULT_CONFIG_DIR = REPO_ROOT / "configs" / "generated" / HS_SPEC.config_subdir
DEFAULT_SUMMARY_DIR = REPO_ROOT / "result" / "grid_search" / HS_SPEC.summary_subdir
BEST_CONFIG = REPO_ROOT / "configs" / f"{HS_SPEC.cluster_stem}-best-2c.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Hijacking Suppression: tune on small data, then full eval with best params"
    )
    parser.add_argument(
        "--phase",
        choices=["tune", "full", "all"],
        default="all",
        help="tune=grid only; full=best params; all=both (default)",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=None,
        help="summary.json from tune phase (for --phase full)",
    )
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=None,
        help="Generated YAML output directory",
    )
    parser.add_argument(
        "--summary-dir",
        type=Path,
        default=None,
        help="Summary / full_result output directory",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_dir = args.config_dir or DEFAULT_CONFIG_DIR
    summary_dir = args.summary_dir or DEFAULT_SUMMARY_DIR

    print("[hs-pipeline] Hijacking Suppression tune → full", flush=True)
    print(f"[hs-pipeline] Repo: {REPO_ROOT}", flush=True)
    print(f"[hs-pipeline] Phase: {args.phase}", flush=True)
    print(
        f"[hs-pipeline] Samples: tune={TUNE_SAMPLES}, full={FULL_SAMPLES}",
        flush=True,
    )
    print(f"[hs-pipeline] Config dir: {config_dir}", flush=True)
    print(f"[hs-pipeline] Summary dir: {summary_dir}", flush=True)

    run_defense_pipeline(
        HS_SPEC,
        args.phase,
        args.summary,
        config_dir,
        summary_dir,
    )

    if args.phase in ("full", "all") and BEST_CONFIG.exists():
        print(f"[hs-pipeline] Best full config: {BEST_CONFIG}", flush=True)
    if args.phase in ("tune", "all"):
        print(f"[hs-pipeline] Tune summary: {summary_dir / 'summary.json'}", flush=True)
    if args.phase in ("full", "all"):
        print(f"[hs-pipeline] Full result: {summary_dir / 'full_result.json'}", flush=True)


if __name__ == "__main__":
    main()
