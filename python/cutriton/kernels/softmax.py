from __future__ import annotations

from math import exp
from typing import Any


def softmax(x: Any, axis: int = -1) -> Any:
    """在 Triton 启动链路接入前，用于正确性测试的 Softmax 参考实现。"""

    try:
        import numpy as np  # type: ignore

        x_arr = np.asarray(x)
        x_max = np.max(x_arr, axis=axis, keepdims=True)
        numerator = np.exp(x_arr - x_max)
        return numerator / np.sum(numerator, axis=axis, keepdims=True)
    except ImportError:
        if axis not in (-1, 0):
            raise NotImplementedError("列表回退实现只支持一维 Softmax")
        values = [float(value) for value in x]
        max_value = max(values)
        numerator = [exp(value - max_value) for value in values]
        denom = sum(numerator)
        return [value / denom for value in numerator]
