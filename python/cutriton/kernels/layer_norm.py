from __future__ import annotations

from typing import Any

import numpy as np


def layer_norm(x: Any, epsilon: float = 1e-5) -> Any:
    """Final-axis LayerNorm numerical reference."""
    values = np.asarray(x)
    mean = np.mean(values, axis=-1, keepdims=True)
    variance = np.mean((values - mean) ** 2, axis=-1, keepdims=True)
    return (values - mean) / np.sqrt(variance + epsilon)
