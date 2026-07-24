"""
RolePlay 注意力防御实验的路径与命名约定。

目录布局:
  configs/roleplay/{defense}/           手写 + grid 最优配置
  configs/roleplay/{defense}/generated/  grid 调参临时 YAML（gitignore）
  logs/roleplay/{defense}/              pipeline .log + attention 文本
  result/roleplay/{defense}/            jsonl / metrics / attention.csv
  result/roleplay/{defense}/grid_search/  grid 汇总 summary.json
  logs/slurm/roleplay/                  SLURM stdout/stderr
  artifacts/hpc/roleplay/{defense}/     从 HPC 归档的结果副本
  scripts/roleplay_defenses/            Python + shell 入口
  slurm/roleplay/                       sbatch 脚本
"""

from __future__ import annotations

import os
from typing import List, Optional

from utils.config_utils import AttackConfig

ROLEPLAY_DEFENSES = frozenset({"attention_sharpening", "hijacking_suppression"})

DEFENSE_SHORT = {
    "attention_sharpening": "as",
    "hijacking_suppression": "hs",
}


def primary_defense_type(config: AttackConfig) -> Optional[str]:
    if not config.defenses:
        return None
    types = [d.type for d in config.defenses if d.type]
    if len(types) == 1 and types[0] in ROLEPLAY_DEFENSES:
        return types[0]
    return None


def is_roleplay_defense_experiment(config: AttackConfig) -> bool:
    return primary_defense_type(config) is not None


def roleplay_defense_dir(config: AttackConfig) -> Optional[str]:
    return primary_defense_type(config)


def scientsbank_2c_pipeline_id(config: AttackConfig) -> Optional[str]:
    """从 run name 解析 pipeline id，如 scientsbank-2c-gcg-hs-full → gcg_hs。"""
    name = config.name or ""
    if not name.startswith("scientsbank-2c-"):
        return None
    rest = name[len("scientsbank-2c-"):]
    parts = rest.split("-")
    if len(parts) >= 2:
        attack = parts[0]
        if attack == "rp":
            attack = "roleplay"
        return f"{attack}_{parts[1]}"
    return None


def is_scientsbank_2c_experiment(config: AttackConfig) -> bool:
    return scientsbank_2c_pipeline_id(config) is not None


def experiment_log_dir(config: AttackConfig) -> str:
    sb_id = scientsbank_2c_pipeline_id(config)
    if sb_id:
        return os.path.join(
            config.log_config.log_dir, "experiments", "scientsbank_2c", sb_id
        )
    defense = roleplay_defense_dir(config)
    if defense:
        return os.path.join(config.log_config.log_dir, "roleplay", defense)
    return os.path.join(
        config.log_config.log_dir,
        config.attack_method,
        config.model_config.name,
    )


def experiment_result_dir(config: AttackConfig) -> str:
    sb_id = scientsbank_2c_pipeline_id(config)
    if sb_id:
        return os.path.join(
            config.log_config.result_dir, "experiments", "scientsbank_2c", sb_id
        )
    defense = roleplay_defense_dir(config)
    if defense:
        return os.path.join(config.log_config.result_dir, "roleplay", defense)
    return os.path.join(
        config.log_config.result_dir,
        config.attack_method,
        config.model_config.name,
    )


def grid_search_summary_dir(defense_key: str, result_root: str = "./result") -> str:
    return os.path.join(result_root, "roleplay", defense_key, "grid_search")


def grid_generated_config_dir(defense_key: str, config_root: str = "./configs") -> str:
    return os.path.join(config_root, "roleplay", defense_key, "generated")


def best_config_path(defense_key: str, config_root: str = "./configs") -> str:
    return os.path.join(config_root, "roleplay", defense_key, "best-2c.yaml")


def hpc_artifact_dir(defense_key: str, artifact_root: str = "./artifacts/hpc") -> str:
    return os.path.join(artifact_root, "roleplay", defense_key)


def metrics_search_dirs(config_name: str, defense_key: Optional[str] = None) -> List[str]:
    """按优先级返回可能存放 metrics 的目录（新 → 旧）。"""
    dirs: List[str] = []
    if defense_key:
        dirs.append(os.path.join("result", "roleplay", defense_key))
    dirs.append(os.path.join("result", "RolePlay", "Llama-3.1-8B-Instruct"))
    return dirs
