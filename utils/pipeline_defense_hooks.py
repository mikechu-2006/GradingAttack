"""Step-3 hook lifecycle helpers for model-level defenses."""

from typing import Callable, List, TypeVar

T = TypeVar("T")


def run_with_defense_hooks(defenses, model, fn: Callable[[], T]) -> T:
    """Install model hooks for Step 3, run fn, always uninstall in finally."""
    hook_removers: List[Callable[[], None]] = []
    for defense in defenses:
        if defense.requires_model_hooks():
            hook_removers.extend(defense.install_model_hooks(model))
    try:
        return fn()
    finally:
        for remove in hook_removers:
            remove()
