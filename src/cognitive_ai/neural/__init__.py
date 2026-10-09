"""Concept-recognition neural net (Phase 3A). Torch is OPTIONAL.

Rule system modules must never import torch. Everything torch-dependent
goes through `require_torch()` so a missing torch only disables this
package, never the cognitive core.
"""
from __future__ import annotations

try:
    import torch  # noqa: F401
    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - environment dependent
    torch = None  # type: ignore
    TORCH_AVAILABLE = False


def require_torch():
    if not TORCH_AVAILABLE:
        raise RuntimeError(
            "cognitive_ai.neural needs the optional 'neural' extra "
            "(pip install -e \".[neural]\"). The rule-based cognitive "
            "system works without it.")
    import torch as _torch
    return _torch


LEVEL1 = ["entity", "event", "action", "attribute", "abstract"]

LEVEL2 = ["none", "container", "tool", "furniture", "vehicle", "food",
          "electronics", "weather", "social", "accident", "motion",
          "eating", "making", "color", "shape", "temperature", "size",
          "emotion", "science", "time", "society"]
