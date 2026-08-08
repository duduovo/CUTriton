from __future__ import annotations

from math import sqrt
from typing import Any


def layer_norm(x: Any, epsilon: float = 1e-5) -> Any:
    """最后一维 LayerNorm 的参考实现。"""

    try:
        import numpy as np  # type: ignore

        x_arr = np.asarray(x)
        mean = np.mean(x_arr, axis=-1, keepdims=True)
        var = np.mean((x_arr - mean) ** 2, axis=-1, keepdims=True)
        return (x_arr - mean) / np.sqrt(var + epsilon)
    except ImportError:
        values = [float(value) for value in x]
        mean = sum(values) / len(values)
        var = sum((value - mean) ** 2 for value in values) / len(values)
        denom = sqrt(var + epsilon)
        return [(value - mean) / denom for value in values]
