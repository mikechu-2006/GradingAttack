#!/bin/bash
#SBATCH -p emergency_gpua40
#SBATCH -o output_grid_defense_%j.txt
#SBATCH -e err_grid_defense_%j.txt
#SBATCH -n 8
#SBATCH --gres=gpu:1
#SBATCH -t 7-00:00:00

# 统一网格搜索：50×20 调参 + 100 条全量最优（请在 ~/GradingAttack 下 sbatch）
#
# 环境变量:
#   DEFENSE_TYPE=all|attention_sharpening|hijacking_suppression  (默认 all)
#   GRID_SEARCH_PHASE=all|tune|full                            (默认 all)
#
# 示例:
#   sbatch run_pipeline_roleplay_defense_grid_search.sh
#   DEFENSE_TYPE=hijacking_suppression sbatch run_pipeline_roleplay_defense_grid_search.sh
#   GRID_SEARCH_PHASE=tune DEFENSE_TYPE=attention_sharpening sbatch run_pipeline_roleplay_defense_grid_search.sh

echo "[sbatch] grid job=${SLURM_JOB_ID:-unknown} submit_dir=${SLURM_SUBMIT_DIR:-unknown} $(date -Iseconds)"

bash scripts/run_roleplay_defense_grid_search.sh
