#!/bin/bash
# 构建 SciEntsBank 2 分类 GCG 后缀库（供 gcg_hs / gcg_as 管线使用）
#
# 注意：四管线默认读取的是预构建迁移库（非本脚本直接产出）：
#   result_from_hpc_gcg_suffix_bank/gcg_suffix_bank_2c_his.jsonl
# 本脚本构建的是全新优化库，默认写到 gcg_suffix_bank_2c.jsonl。
# 若要用新库跑管线，提交前设置：
#   SUFFIX_BANK_PATH=result_from_hpc_gcg_suffix_bank/gcg_suffix_bank_2c.jsonl
set -eo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"
source "${REPO_ROOT}/scripts/roleplay_defenses/run_env.sh"

OUTPUT_BANK="${OUTPUT_BANK:-result_from_hpc_gcg_suffix_bank/gcg_suffix_bank_2c.jsonl}"
ATTEMPT_LOG="${ATTEMPT_LOG:-result/GCG/gcg_suffix_bank_2c_build_attempts.jsonl}"
TARGET_BANK_SIZE="${TARGET_BANK_SIZE:-10}"
MAX_CANDIDATES="${MAX_CANDIDATES:-50}"

echo "[sb2c-bank] OUTPUT_BANK=${OUTPUT_BANK} TARGET=${TARGET_BANK_SIZE}"

exec "${PYTHON}" scripts/build_gcg_suffix_bank.py \
  --data dataset/scientsbank.jsonl \
  --template configs/grading_template_ci_2c.txt \
  --nclass 2 \
  --success-mode correct \
  --append-eos-tag \
  --target-bank-size "${TARGET_BANK_SIZE}" \
  --max-candidates "${MAX_CANDIDATES}" \
  --output-bank "${OUTPUT_BANK}" \
  --attempt-log "${ATTEMPT_LOG}"
