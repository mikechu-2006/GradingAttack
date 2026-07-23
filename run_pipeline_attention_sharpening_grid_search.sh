#!/bin/bash
#SBATCH -p emergency_gpua40
#SBATCH -o output_grid_%j.txt
#SBATCH -e err_grid_%j.txt
#SBATCH -n 8
#SBATCH --gres=gpu:1
#SBATCH -t 7-00:00:00

# 网格搜索(50×20) + 全量最优(100)。请在 ~/GradingAttack 下 sbatch。
# 仅调参: GRID_SEARCH_PHASE=tune sbatch run_pipeline_attention_sharpening_grid_search.sh
# 仅全量: GRID_SEARCH_PHASE=full  sbatch run_pipeline_attention_sharpening_grid_search.sh
echo "[sbatch] grid job=${SLURM_JOB_ID:-unknown} submit_dir=${SLURM_SUBMIT_DIR:-unknown} $(date -Iseconds)"

bash scripts/run_attention_sharpening_grid_search.sh
