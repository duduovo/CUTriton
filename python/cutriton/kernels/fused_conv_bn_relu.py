from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np


def fused_conv_bn_relu_reference(
    x: Any,
    weight: Any,
    scale: Any,
    bias: Any,
    mean: Any,
    variance: Any,
    strides: Sequence[int] = (1, 1),
    pads: Sequence[int] = (0, 0, 0, 0),
    epsilon: float = 1e-5,
) -> Any:
    """Naive NCHW Conv+BN+ReLU reference for small correctness cases."""
    source, kernel = np.asarray(x), np.asarray(weight)
    batch_count, _, height, width = source.shape
    output_channels, _, kernel_h, kernel_w = kernel.shape
    top, left, bottom, right = pads
    stride_h, stride_w = strides
    padded = np.pad(source, ((0, 0), (0, 0), (top, bottom), (left, right)))
    output_h = (height + top + bottom - kernel_h) // stride_h + 1
    output_w = (width + left + right - kernel_w) // stride_w + 1
    convolution = np.zeros((batch_count, output_channels, output_h, output_w), dtype=source.dtype)
    for batch in range(batch_count):
        for channel in range(output_channels):
            for row in range(output_h):
                for column in range(output_w):
                    window = padded[
                        batch,
                        :,
                        row * stride_h : row * stride_h + kernel_h,
                        column * stride_w : column * stride_w + kernel_w,
                    ]
                    convolution[batch, channel, row, column] = np.sum(window * kernel[channel])
    channel_shape = (1, -1, 1, 1)
    normalized = (convolution - np.asarray(mean).reshape(channel_shape)) / np.sqrt(
        np.asarray(variance).reshape(channel_shape) + epsilon
    )
    result = normalized * np.asarray(scale).reshape(channel_shape)
    result += np.asarray(bias).reshape(channel_shape)
    return np.maximum(result, 0)
