#!/bin/bash
# 在计算节点上初始化 conda 并激活 gradingattack 环境。
# HKUST-GZ HPC II 期: module load anaconda3
# 用法: source scripts/activate_gradingattack_env.sh

_init_conda_from_module() {
    if ! command -v module >/dev/null 2>&1; then
        return 1
    fi
    module load anaconda3 2>/dev/null || return 1

    local conda_exe conda_base
    conda_exe="$(command -v conda 2>/dev/null || true)"
    [ -n "${conda_exe}" ] || return 1

    conda_base="$(dirname "$(dirname "${conda_exe}")")"
    if [ -f "${conda_base}/etc/profile.d/conda.sh" ]; then
        # shellcheck disable=SC1091
        source "${conda_base}/etc/profile.d/conda.sh"
        return 0
    fi

    eval "$(conda shell.bash hook)"
}

_activate_conda() {
    if [ "${CONDA_DEFAULT_ENV:-}" = "gradingattack" ] && [ -x "${CONDA_PREFIX:-}/bin/python" ]; then
        return 0
    fi

    if _init_conda_from_module; then
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

    return 1
}

if ! _activate_conda; then
    echo "[env] conda not found. Try:" >&2
    echo "      module load anaconda3" >&2
    echo "      source \"\$(dirname \"\$(dirname \"\$(which conda)\")\")/etc/profile.d/conda.sh\"" >&2
    echo "      conda activate gradingattack" >&2
    return 1 2>/dev/null || exit 1
fi

if [ "${CONDA_DEFAULT_ENV:-}" != "gradingattack" ]; then
    conda activate gradingattack
fi
