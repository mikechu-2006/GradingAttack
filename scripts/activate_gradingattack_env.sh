#!/bin/bash
# 在计算节点上初始化 conda 并激活 gradingattack 环境。
# 用法: source scripts/activate_gradingattack_env.sh

_activate_conda() {
    if command -v conda >/dev/null 2>&1; then
        return 0
    fi

    # 非交互 shell（srun --pty bash）通常不会加载 .bashrc
    if [ -f "${HOME}/.bashrc" ]; then
        # shellcheck disable=SC1090
        source "${HOME}/.bashrc"
    fi
    if command -v conda >/dev/null 2>&1; then
        return 0
    fi

    for candidate in \
        "${HOME}/miniconda3/etc/profile.d/conda.sh" \
        "${HOME}/anaconda3/etc/profile.d/conda.sh" \
        "${HOME}/.conda/etc/profile.d/conda.sh" \
        "/opt/conda/etc/profile.d/conda.sh"; do
        if [ -f "${candidate}" ]; then
            # shellcheck disable=SC1090
            source "${candidate}"
            return 0
        fi
    done

    if command -v module >/dev/null 2>&1; then
        for mod in anaconda3 miniconda3 conda Python/anaconda3; do
            if module load "${mod}" 2>/dev/null && command -v conda >/dev/null 2>&1; then
                return 0
            fi
        done
    fi

    return 1
}

if ! _activate_conda; then
    echo "[env] conda not found. On login node, run:" >&2
    echo "      which conda" >&2
    echo "      module avail 2>&1 | grep -i conda" >&2
    echo "Then add the correct init path to scripts/activate_gradingattack_env.sh" >&2
    return 1 2>/dev/null || exit 1
fi

conda activate gradingattack
