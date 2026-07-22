#!/bin/bash
# 在计算节点上初始化 conda 并激活 gradingattack 环境。
# HKUST-GZ HPC II 期: module load anaconda3
# 用法: source scripts/activate_gradingattack_env.sh

_init_conda_shell() {
    if command -v conda >/dev/null 2>&1; then
        eval "$(conda shell.bash hook)"
        return 0
    fi
    return 1
}

_activate_conda() {
    # HKUST HPC 上优先用 module（计算节点常见路径）
    if command -v module >/dev/null 2>&1; then
        module load anaconda3 2>/dev/null || true
    fi
    if _init_conda_shell; then
        return 0
    fi

    if [ -f "${HOME}/.bashrc" ]; then
        # shellcheck disable=SC1090
        source "${HOME}/.bashrc"
    fi
    if _init_conda_shell; then
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
            if _init_conda_shell; then
                return 0
            fi
        fi
    done

    if command -v module >/dev/null 2>&1; then
        for mod in miniconda3 conda Python/anaconda3; do
            module load "${mod}" 2>/dev/null || continue
            if _init_conda_shell; then
                return 0
            fi
        done
    fi

    return 1
}

if ! _activate_conda; then
    echo "[env] conda not found. Try on login/compute node:" >&2
    echo "      module load anaconda3" >&2
    echo "      eval \"\$(conda shell.bash hook)\"" >&2
    echo "      conda env list" >&2
    return 1 2>/dev/null || exit 1
fi

conda activate gradingattack
