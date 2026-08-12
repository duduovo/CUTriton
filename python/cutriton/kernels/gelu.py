from __future__ import annotations

from math import erf, sqrt, tanh
from typing import Any


def gelu(x: Any, approximate: bool = True) -> Any:
    """用于测试和 Triton Kernel 校验的 GELU 参考实现。"""

    try:
        import numpy as np  # type: ignore

        x_arr = np.asarray(x)
        if approximate:
            return 0.5 * x_arr * (
                1.0 + np.tanh(0.7978845608028654 * (x_arr + 0.044715 * x_arr**3))
            )
        return 0.5 * x_arr * (1.0 + np.vectorize(erf)(x_arr / sqrt(2.0)))
    except ImportError:
        return _map_nested(x, lambda value: _gelu_scalar(float(value), approximate))


def _gelu_scalar(value: float, approximate: bool) -> float:
    if approximate:
        return 0.5 * value * (
            1.0 + tanh(0.7978845608028654 * (value + 0.044715 * value**3))
        )
    return 0.5 * value * (1.0 + erf(value / sqrt(2.0)))


def _map_nested(value: Any, fn):
    if isinstance(value, list):
        return [_map_nested(item, fn) for item in value]
    return fn(value)
