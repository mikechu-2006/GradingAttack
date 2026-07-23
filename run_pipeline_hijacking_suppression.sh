#!/bin/bash
#SBATCH -p emergency_gpua40
#SBATCH -o output_hs_pipeline_%j.txt
#SBATCH -e err_hs_pipeline_%j.txt
#SBATCH -n 8
#SBATCH --gres=gpu:1
#SBATCH -t 7-00:00:00

# Hijacking Suppression 完整管线：50 样本调参 → 100 样本全量最优
#
# 默认一次跑完 tune + full。分阶段示例:
#   HS_PIPELINE_PHASE=tune sbatch run_pipeline_hijacking_suppression.sh
#   HS_PIPELINE_PHASE=full  sbatch run_pipeline_hijacking_suppression.sh
#
# 调参网格: beta × layers (20 组合 × 50 样本)
# 全量: 最优 beta/layers × 100 样本
# 输出: result/grid_search/hijacking_suppression/summary.json
#       configs/RolePlay-...-hijacking-suppression-best-2c.yaml

echo "[sbatch-hs-pipeline] job=${SLURM_JOB_ID:-unknown} submit_dir=${SLURM_SUBMIT_DIR:-unknown} $(date -Iseconds)"

bash scripts/run_hijacking_suppression_pipeline.sh
