#!/usr/bin/env python3
"""SciEntsBank 2c 管线本地 smoke test（无需 GPU / main.py）。"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.experiments.scientsbank_2c import grid_search as gs  # noqa: E402


def _write_fake_metrics(path: Path, asr: float = 0.1, asr_def: float = 0.05) -> None:
    payload = {
        "asr": asr,
        "asr_defended": asr_def,
        "qwk_clean": 0.4,
        "qwk_attack": 0.1,
        "qwk_defense_clean": 0.38,
        "qwk_defense_attack": 0.08,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_bank_and_config() -> None:
    bank = gs.ensure_suffix_bank()
    assert bank.is_file(), "suffix bank missing"
    for key in ("gcg_hs", "gcg_as"):
        spec = gs.PIPELINES[key]
        cfg = gs.build_config(spec, "full", 0.1, "L24-31", list(range(24, 32)), 2, True)
        assert cfg["method"] == "gcg_suffix_bank"
        assert cfg["params"]["bank_path"] == gs.resolve_suffix_bank_path()
        assert cfg["pipeline_mode"] is True
    print("[smoke] bank + gcg config OK")


def test_metrics_glob_excludes_bank_metrics() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        gs.REPO_ROOT = root
        result_dir = root / "result" / "experiments" / "scientsbank_2c" / "gcg_hs"
        name = "scientsbank-2c-gcg-hs-tune-B005-L28-31"
        _write_fake_metrics(result_dir / f"{name}_202607010101_metrics.json")
        _write_fake_metrics(result_dir / f"{name}_202607010101_bank_metrics.json", 0.0, 0.0)
        found = gs.try_find_metrics_path(name, "gcg_hs")
        assert found is not None
        assert found.name.endswith("_metrics.json")
        assert not found.name.endswith("_bank_metrics.json")
    print("[smoke] metrics glob excludes bank_metrics OK")


def test_dry_run_flow_with_fake_tune() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        old_root = gs.REPO_ROOT
        gs.REPO_ROOT = root
        try:
            bank_dir = root / "result_from_hpc_gcg_suffix_bank"
            bank_dir.mkdir(parents=True)
            (bank_dir / "gcg_suffix_bank_2c_his.jsonl").write_text(
                '{"best_string": "test-suffix"}\n', encoding="utf-8"
            )

            spec = gs.PIPELINES["gcg_hs"]
            result_dir = root / "result" / "experiments" / "scientsbank_2c" / "gcg_hs"
            for param, layer, _ in gs.grid_combinations(spec):
                name = gs.config_name(spec, "tune", param, layer)
                _write_fake_metrics(result_dir / f"{name}_202607010101_metrics.json")

            assert gs.grid_metrics_complete(spec)
            gs.run_pipeline_flow(
                spec,
                "all",
                resume=True,
                force_rerun=False,
                rebuild_summary=False,
                allow_partial_tune=True,
                dry_run=True,
            )
        finally:
            gs.REPO_ROOT = old_root
    print("[smoke] dry-run phase=all flow OK")


def main() -> None:
    test_bank_and_config()
    test_metrics_glob_excludes_bank_metrics()
    test_dry_run_flow_with_fake_tune()
    print("[smoke] all checks passed")


if __name__ == "__main__":
    main()
