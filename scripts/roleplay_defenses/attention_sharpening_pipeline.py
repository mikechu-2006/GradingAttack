#!/usr/bin/env python3
"""Attention Sharpening：小样本调参 → 最优参数全量。"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "scripts" / "roleplay_defenses"
sys.path.insert(0, str(SCRIPTS_DIR))

from grid_search import (  # noqa: E402
    DEFENSES,
    FULL_SAMPLES,
    TUNE_SAMPLES,
    run_defense_pipeline,
)

SPEC = DEFENSES["attention_sharpening"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Attention Sharpening tune → full pipeline")
    p.add_argument("--phase", choices=["tune", "full", "all"], default="all")
    p.add_argument("--summary", type=Path, default=None)
    p.add_argument("--config-dir", type=Path, default=None)
    p.add_argument("--summary-dir", type=Path, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    print("[as-pipeline] Attention Sharpening tune → full", flush=True)
    print(f"[as-pipeline] Phase: {args.phase}", flush=True)
    print(f"[as-pipeline] Samples: tune={TUNE_SAMPLES}, full={FULL_SAMPLES}", flush=True)
    run_defense_pipeline(
        SPEC,
        args.phase,
        args.summary,
        args.config_dir,
        args.summary_dir,
    )
    print(f"[as-pipeline] Best config: {SPEC.best_config_path}", flush=True)
    print(f"[as-pipeline] Summary: {SPEC.summary_dir / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
