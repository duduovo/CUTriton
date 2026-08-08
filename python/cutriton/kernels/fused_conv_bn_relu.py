from __future__ import annotations

from typing import Any, Sequence


def fused_conv_bn_relu_reference(
    x: Any,
    weight: Any,
    scale: Any,
    bias: Any,
    mean: Any,
    var: Any,
    strides: Sequence[int] = (1, 1),
    pads: Sequence[int] = (0, 0, 0, 0),
    epsilon: float = 1e-5,
) -> Any:
    """用于小规模正确性用例的朴素 NCHW Conv+BN+ReLU 参考实现。"""

    try:
        import numpy as np  # type: ignore
    except ImportError as exc:
        raise RuntimeError("fused_conv_bn_relu_reference 需要安装 numpy") from exc

    x_arr = np.asarray(x)
    w_arr = np.asarray(weight)
    n, _, h, w = x_arr.shape
    out_c, _, kh, kw = w_arr.shape
    pad_top, pad_left, pad_bottom, pad_right = pads
    stride_h, stride_w = strides
    padded = np.pad(
        x_arr,
        ((0, 0), (0, 0), (pad_top, pad_bottom), (pad_left, pad_right)),
        mode="constant",
    )
    out_h = (h + pad_top + pad_bottom - kh) // stride_h + 1
    out_w = (w + pad_left + pad_right - kw) // stride_w + 1
    conv = np.zeros((n, out_c, out_h, out_w), dtype=x_arr.dtype)
    for batch in range(n):
        for channel in range(out_c):
            for oy in range(out_h):
                for ox in range(out_w):
                    window = padded[
                        batch,
                        :,
                        oy * stride_h : oy * stride_h + kh,
                        ox * stride_w : ox * stride_w + kw,
                    ]
                    conv[batch, channel, oy, ox] = np.sum(window * w_arr[channel])
    normalized = (conv - np.asarray(mean).reshape(1, -1, 1, 1)) / np.sqrt(
        np.asarray(var).reshape(1, -1, 1, 1) + epsilon
    )
    bn = normalized * np.asarray(scale).reshape(1, -1, 1, 1) + np.asarray(
        bias
    ).reshape(1, -1, 1, 1)
    return np.maximum(bn, 0)
