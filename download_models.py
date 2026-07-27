#!/usr/bin/env python3
"""Pre-download models from ModelScope to local cache for the GradingAttack demo.

Reads demo_config.json, and for each vllm model downloads it from ModelScope
to the specified model_path directory. Skips models whose path already exists.
"""

import json
import os
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).parent
CONFIG_PATH = PROJECT_DIR / "demo_config.json"


def download_model(model_id: str, model_path: str) -> bool:
    """Download a model from ModelScope to a local directory.

    Uses modelscope.snapshot_download to fetch the full model snapshot.
    Returns True on success, False on failure.
    """
    if os.path.isdir(model_path) and any(
        f.endswith((".safetensors", ".bin", ".pt")) for f in os.listdir(model_path)
    ):
        print(f"  [SKIP] Already exists: {model_path}")
        return True

    try:
        from modelscope import snapshot_download
    except ImportError:
        print("  [FALLBACK] modelscope not installed, trying huggingface_hub...")
        try:
            from huggingface_hub import snapshot_download as hf_snapshot_download
            os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
            hf_snapshot_download(
                model_id, local_dir=model_path,
                local_dir_use_symlinks=False, resume_download=True,
            )
            print(f"  [OK] Downloaded to {model_path}")
            return True
        except ImportError:
            print("  [ERROR] Neither modelscope nor huggingface_hub is installed.")
            print("  Install with: pip install modelscope")
            return False

    print(f"  Downloading {model_id} -> {model_path} ...")
    os.makedirs(model_path, exist_ok=True)
    snapshot_download(
        model_id,
        cache_dir=model_path,
        local_dir=model_path,
    )
    print(f"  [OK] Downloaded to {model_path}")
    return True


def main():
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    models = config.get("models", {})

    vllm_models = [
        (name, cfg)
        for name, cfg in models.items()
        if cfg.get("backend") == "vllm"
    ]

    if not vllm_models:
        print("No vLLM models found in config.")
        return

    print(f"Found {len(vllm_models)} vLLM model(s) to prepare.\n")

    failed = []
    for name, cfg in vllm_models:
        model_id = cfg.get("model_id", "")
        model_path = cfg.get("model_path", "")
        if not model_id or not model_path:
            print(f"[{name}] Missing model_id or model_path — skipping")
            continue

        print(f"[{name}]")
        if not download_model(model_id, model_path):
            failed.append(name)
        print()

    if failed:
        print(f"FAILED: {', '.join(failed)}")
        sys.exit(1)
    else:
        print("All models ready.")


if __name__ == "__main__":
    main()
