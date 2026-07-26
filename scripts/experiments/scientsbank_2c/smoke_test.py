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
from utils.config_utils import parse_config  # noqa: E402
from utils.experiment_paths import scientsbank_2c_pipeline_id  # noqa: E402

INJECTION_KEYS = ("ao_hs", "ao_as", "dc_hs", "dc_as", "im_hs", "im_as")
GCG_KEYS = ("gcg_hs", "gcg_as")
RUN_WRAPPER_SCRIPTS = {
    "ao_hs": REPO_ROOT / "run_pipeline_injection_ao_hs.sh",
    "ao_as": REPO_ROOT / "run_pipeline_injection_ao_as.sh",
    "dc_hs": REPO_ROOT / "run_pipeline_injection_dc_hs.sh",
    "dc_as": REPO_ROOT / "run_pipeline_injection_dc_as.sh",
    "im_hs": REPO_ROOT / "run_pipeline_injection_im_hs.sh",
    "im_as": REPO_ROOT / "run_pipeline_injection_im_as.sh",
}


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


def _seed_tune_metrics(spec: gs.PipelineSpec, result_dir: Path) -> None:
    for param, layer, _ in gs.grid_combinations(spec):
        name = gs.config_name(spec, "tune", param, layer)
        _write_fake_metrics(result_dir / f"{name}_202607010101_metrics.json")


def test_gcg_samples_unchanged() -> None:
    for key in GCG_KEYS:
        spec = gs.PIPELINES[key]
        assert spec.tune_samples == gs.TUNE_SAMPLES_GCG == 10, key
        assert spec.full_samples == gs.FULL_SAMPLES == 100, key
        assert spec.attack_method == "gcg_suffix_bank", key
        assert spec.injection_short == "", key
    print("[smoke] gcg sample sizes unchanged (tune 10 / full 100)")


def test_bank_and_config() -> None:
    bank = gs.ensure_suffix_bank()
    assert bank.is_file(), "suffix bank missing"
    for key in GCG_KEYS:
        spec = gs.PIPELINES[key]
        cfg = gs.build_config(spec, "full", 0.1, "L24-31", list(range(24, 32)), 100, True)
        assert cfg["method"] == "gcg_suffix_bank"
        assert cfg["params"]["bank_path"] == gs.resolve_suffix_bank_path()
        assert "injection_prompt" not in cfg["params"]
        assert cfg["data"][0]["max_samples"] == 100
        assert cfg["pipeline_mode"] is True
    print("[smoke] bank + gcg config OK")


def test_injection_config() -> None:
    prompts = set()
    for key in INJECTION_KEYS:
        spec = gs.PIPELINES[key]
        assert spec.tune_samples == gs.TUNE_SAMPLES_INJECTION == 5, key
        assert spec.full_samples == gs.FULL_SAMPLES_INJECTION == 50, key
        assert spec.attack_method == "Injection", key
        cfg = gs.build_config(spec, "tune", 0.1, "L24-31", list(range(24, 32)), 5, False)
        assert cfg["method"] == "Injection"
        assert cfg["data"][0]["max_samples"] == 5
        assert cfg["params"]["injection_prompt"]
        assert "bank_path" not in cfg["params"]
        assert cfg["defenses"][0]["type"] in ("hijacking_suppression", "attention_sharpening")
        full_cfg = gs.build_config(spec, "full", 0.1, "L24-31", list(range(24, 32)), 50, True)
        assert full_cfg["data"][0]["max_samples"] == 50
        assert full_cfg["log_attention"] is True
        prompts.add(cfg["params"]["injection_prompt"])
    assert len(prompts) == 3, "ao/dc/im should use three distinct injection prompts"
    print("[smoke] all 6 injection configs OK")


def test_generated_yaml_parseable() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        old_root = gs.REPO_ROOT
        gs.REPO_ROOT = root
        try:
            cases = [
                (gs.PIPELINES["gcg_hs"], "tune", 0.1, "L28-31", [28, 29, 30, 31], 10, False),
                (gs.PIPELINES["gcg_as"], "full", 0.7, "L24-31", list(range(24, 32)), 100, True),
                (gs.PIPELINES["ao_hs"], "tune", 0.1, "L24-31", list(range(24, 32)), 5, False),
                (gs.PIPELINES["im_as"], "full", 0.7, "L28-31", [28, 29, 30, 31], 50, True),
            ]
            for spec, phase, param, layer_label, layers, max_samples, log_attn in cases:
                cfg = gs.build_config(spec, phase, param, layer_label, layers, max_samples, log_attn)
                cfg_path = gs.write_config(cfg, spec.config_dir)
                parsed = parse_config(str(cfg_path))
                assert parsed.pipeline_mode is True
                assert parsed.attack_method.lower() == spec.attack_method.lower()
                assert scientsbank_2c_pipeline_id(parsed) == spec.key
        finally:
            gs.REPO_ROOT = old_root
    print("[smoke] generated YAML parseable + pipeline id OK")


def test_injection_does_not_require_suffix_bank() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        old_root = gs.REPO_ROOT
        gs.REPO_ROOT = root
        try:
            spec = gs.PIPELINES["dc_as"]
            result_dir = root / "result" / "experiments" / "scientsbank_2c" / "dc_as"
            _seed_tune_metrics(spec, result_dir)
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
    print("[smoke] injection flow OK without suffix bank")


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


def _dry_run_all_flow(spec_key: str) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        old_root = gs.REPO_ROOT
        gs.REPO_ROOT = root
        try:
            spec = gs.PIPELINES[spec_key]
            if spec.attack_method == "gcg_suffix_bank":
                bank_dir = root / "result_from_hpc_gcg_suffix_bank"
                bank_dir.mkdir(parents=True)
                (bank_dir / "gcg_suffix_bank_2c_his.jsonl").write_text(
                    '{"best_string": "test-suffix"}\n', encoding="utf-8"
                )
            result_dir = root / "result" / "experiments" / "scientsbank_2c" / spec.key
            _seed_tune_metrics(spec, result_dir)
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


def test_dry_run_flow_with_fake_tune() -> None:
    _dry_run_all_flow("gcg_hs")
    print("[smoke] gcg dry-run phase=all flow OK")


def test_all_injection_dry_run_flows() -> None:
    for key in INJECTION_KEYS:
        _dry_run_all_flow(key)
    print("[smoke] all 6 injection dry-run flows OK")


def test_run_wrapper_scripts() -> None:
    for key, path in RUN_WRAPPER_SCRIPTS.items():
        assert path.is_file(), f"missing wrapper: {path.name}"
        text = path.read_text(encoding="utf-8")
        assert f"PIPELINE={key}" in text, path.name
        assert "scripts/experiments/scientsbank_2c/run.sh" in text, path.name
    print("[smoke] 6 run wrapper scripts OK")


def test_pipeline_registry_isolation() -> None:
    gcg_methods = {gs.PIPELINES[k].attack_method for k in GCG_KEYS}
    inj_methods = {gs.PIPELINES[k].attack_method for k in INJECTION_KEYS}
    assert gcg_methods == {"gcg_suffix_bank"}
    assert inj_methods == {"Injection"}
    assert set(GCG_KEYS).isdisjoint(INJECTION_KEYS)
    print("[smoke] pipeline registry isolation OK")


def main() -> None:
    test_pipeline_registry_isolation()
    test_gcg_samples_unchanged()
    test_bank_and_config()
    test_injection_config()
    test_generated_yaml_parseable()
    test_injection_does_not_require_suffix_bank()
    test_metrics_glob_excludes_bank_metrics()
    test_dry_run_flow_with_fake_tune()
    test_all_injection_dry_run_flows()
    test_run_wrapper_scripts()
    print("[smoke] all checks passed")


if __name__ == "__main__":
    main()
