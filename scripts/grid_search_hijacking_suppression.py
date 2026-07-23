#!/usr/bin/env python3
"""兼容入口：转发至 scripts/hijacking_suppression_pipeline.py"""
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
cmd = [
    sys.executable,
    str(REPO_ROOT / "scripts" / "hijacking_suppression_pipeline.py"),
    *sys.argv[1:],
]
subprocess.run(cmd, cwd=REPO_ROOT, check=True)
