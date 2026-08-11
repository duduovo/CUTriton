from __future__ import annotations

from typing import Any

import numpy as np


def softmax(x: Any, axis: int = -1) -> Any:
    """Numerically stable Softmax reference used by correctness tests."""
    values = np.asarray(x)
    shifted = values - np.max(values, axis=axis, keepdims=True)
    numerator = np.exp(shifted)
    return numerator / np.sum(numerator, axis=axis, keepdims=True)
