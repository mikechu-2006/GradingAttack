#!/usr/bin/env python3
"""Pre-download models from ModelScope to local cache for the GradingAttack demo.

Reads demo_config.json, and for each vllm model downloads it from ModelScope
to ~/.cache/modelscope/hub/<model_id>.  Skips models already cached.
"""

import json
import os
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).parent
CONFIG_PATH = PROJECT_DIR / "demo_config.json"


def download_model(model_id: str) -> bool:
    """Download a model from ModelScope to the local cache.

    Uses modelscope.snapshot_download.  Returns True on success.
    """
    cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "modelscope", "hub")
    cached = os.path.join(cache_dir, model_id.replace("/", "___"))
    os.makedirs(cache_dir, exist_ok=True)

    # Quick check: does it look like model weights are already there?
    if os.path.isdir(cached) and any(
        f.endswith((".safetensors", ".bin", ".pt"))
        for f in os.listdir(cached)
    ):
        print(f"  [SKIP] Already cached: {cached}")
        return True

    try:
        from modelscope import snapshot_download
    except ImportError:
        print("  [ERROR] modelscope not installed. Run: pip install modelscope")
        return False

    print(f"  Downloading {model_id} -> {cached} ...")
    snapshot_download(model_id, cache_dir=cache_dir)
    print(f"  [OK] Cached at {cached}")
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

    print(f"Found {len(vllm_models)} vLLM model(s).\n")

    failed = []
    for name, cfg in vllm_models:
        model_id = cfg.get("model_id", "")
        if not model_id:
            print(f"[{name}] Missing model_id — skipping\n")
            continue

        print(f"[{name}]  model_id={model_id}")
        if not download_model(model_id):
            failed.append(name)
        print()

    if failed:
        print(f"FAILED: {', '.join(failed)}")
        sys.exit(1)
    else:
        print("All models cached.")


if __name__ == "__main__":
    main()
