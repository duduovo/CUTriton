from __future__ import annotations

from math import erf, sqrt
from typing import Any

import numpy as np


def gelu(x: Any, approximate: bool = False) -> Any:
    """Numerical GELU reference used by kernel correctness tests."""
    values = np.asarray(x)
    if approximate:
        return 0.5 * values * (1.0 + np.tanh(0.7978845608028654 * (values + 0.044715 * values**3)))
    return 0.5 * values * (1.0 + np.vectorize(erf)(values / sqrt(2.0)))
