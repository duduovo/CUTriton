from __future__ import annotations

import math
from collections.abc import Mapping
from functools import lru_cache
from typing import Any

from .registry import KernelRegistry, KernelSpec

SUPPORTED_ONNX_TYPES = {1, 10, 16}  # FLOAT, FLOAT16, BFLOAT16


def _attribute(node: Any, name: str, default: Any) -> Any:
    for attribute in node.proto.attribute:
        if attribute.name != name:
            continue
        try:
            import onnx

            return onnx.helper.get_attribute_value(attribute)
        except ImportError:
            return default
    return default


def _input_meta(node: Any, metadata: Mapping[str, Any]) -> list[Any]:
    return [metadata.get(name) for name in node.inputs]


def _floating_contiguous_capability(node: Any, metadata: Mapping[str, Any]) -> tuple[bool, str]:
    values = _input_meta(node, metadata)
    if not values or any(value is None for value in values):
        return False, "missing input metadata"
    if any(value.elem_type not in SUPPORTED_ONNX_TYPES for value in values):
        return False, "only FP32/FP16/BF16 are accelerated"
    return True, "supported"


def _binary_capability(node: Any, metadata: Mapping[str, Any]) -> tuple[bool, str]:
    supported, reason = _floating_contiguous_capability(node, metadata)
    if not supported:
        return supported, reason
    values = _input_meta(node, metadata)
    if len(values) != 2 or max(len(values[0].shape), len(values[1].shape)) > 4:
        return False, "v1 binary broadcast supports rank <= 4"
    if values[0].elem_type != values[1].elem_type:
        return False, "binary input dtypes must match"
    left = (1,) * (4 - len(values[0].shape)) + values[0].shape
    right = (1,) * (4 - len(values[1].shape)) + values[1].shape
    for left_dim, right_dim in zip(left, right, strict=True):
        if left_dim == right_dim or left_dim == 1 or right_dim == 1:
            continue
        return False, "binary input shapes are not statically broadcast-compatible"
    return True, "supported equal-shape or rank<=4 broadcast"


def _softmax_capability(node: Any, metadata: Mapping[str, Any]) -> tuple[bool, str]:
    supported, reason = _floating_contiguous_capability(node, metadata)
    if not supported:
        return supported, reason
    value = _input_meta(node, metadata)[0]
    if not value.shape:
        return False, "softmax requires rank >= 1"
    axis = int(_attribute(node, "axis", -1))
    if axis < 0:
        axis += len(value.shape)
    if axis != len(value.shape) - 1:
        return False, "v1 softmax accelerates the final axis only"
    last = value.shape[-1]
    if isinstance(last, int) and last > 65_536:
        return False, "softmax axis exceeds v1 block limit"
    return True, "supported"


def _matmul_capability(node: Any, metadata: Mapping[str, Any]) -> tuple[bool, str]:
    supported, reason = _binary_capability_for_matmul(node, metadata)
    if not supported:
        return supported, reason
    left, right = _input_meta(node, metadata)
    if len(left.shape) != 2 or len(right.shape) != 2:
        return False, "v1 matmul accelerates rank-2 tensors only"
    if left.shape[1] != right.shape[0]:
        return False, "matmul inner dimensions do not match"
    return True, "supported"


def _layer_norm_capability(node: Any, metadata: Mapping[str, Any]) -> tuple[bool, str]:
    supported, reason = _floating_contiguous_capability(node, metadata)
    if not supported:
        return supported, reason
    values = _input_meta(node, metadata)
    if len(values) not in (2, 3) or len(node.outputs) != 1:
        return False, "v1 LayerNormalization requires scale, optional bias, and one output"
    source, scale = values[0], values[1]
    if not source.shape or len(scale.shape) != 1 or source.shape[-1] != scale.shape[0]:
        return False, "v1 LayerNormalization normalizes the final axis"
    axis = int(_attribute(node, "axis", -1))
    if axis not in (-1, len(source.shape) - 1):
        return False, "v1 LayerNormalization supports axis=-1 only"
    columns = source.shape[-1]
    if not isinstance(columns, int) or columns > 65_536:
        return False, "LayerNormalization width must be static and <= 65536"
    return True, "supported"


def _binary_capability_for_matmul(node: Any, metadata: Mapping[str, Any]) -> tuple[bool, str]:
    values = _input_meta(node, metadata)
    if len(values) != 2 or any(value is None for value in values):
        return False, "matmul requires two typed inputs"
    if any(value.elem_type not in SUPPORTED_ONNX_TYPES for value in values):
        return False, "only FP32/FP16/BF16 are accelerated"
    if values[0].elem_type != values[1].elem_type:
        return False, "matmul input dtypes must match"
    return True, "supported"


def _gemm_capability(node: Any, metadata: Mapping[str, Any]) -> tuple[bool, str]:
    values = _input_meta(node, metadata)
    if len(values) not in (2, 3) or any(value is None for value in values):
        return False, "Gemm requires A, B, and optional C"
    if any(value.elem_type not in SUPPORTED_ONNX_TYPES for value in values):
        return False, "only FP32/FP16/BF16 are accelerated"
    if len({value.elem_type for value in values}) != 1:
        return False, "Gemm input dtypes must match"
    left, right = values[:2]
    if len(left.shape) != 2 or len(right.shape) != 2:
        return False, "v1 Gemm accelerates rank-2 A and B only"
    trans_a = bool(_attribute(node, "transA", 0))
    trans_b = bool(_attribute(node, "transB", 0))
    m, k = (left.shape[1], left.shape[0]) if trans_a else left.shape
    other_k, n = (right.shape[1], right.shape[0]) if trans_b else right.shape
    if k != other_k:
        return False, "Gemm inner dimensions do not match"
    if len(values) == 3:
        bias_shape = values[2].shape
        supported_biases = {(), (1,), (n,), (1, n), (m, n)}
        if bias_shape not in supported_biases:
            return False, "v1 Gemm C supports scalar, [N], [1,N], or [M,N]"
    return True, "supported"


def _global_average_pool_capability(node: Any, metadata: Mapping[str, Any]) -> tuple[bool, str]:
    supported, reason = _floating_contiguous_capability(node, metadata)
    if not supported:
        return supported, reason
    value = _input_meta(node, metadata)[0]
    if len(value.shape) != 4 or len(node.outputs) != 1:
        return False, "v1 GlobalAveragePool requires one NCHW input and output"
    height, width = value.shape[-2:]
    if isinstance(height, int) and isinstance(width, int) and height * width > 65_536:
        return False, "GlobalAveragePool spatial area exceeds the v1 block limit"
    return True, "supported"


def _flatten_capability(node: Any, metadata: Mapping[str, Any]) -> tuple[bool, str]:
    supported, reason = _floating_contiguous_capability(node, metadata)
    if not supported:
        return supported, reason
    value = _input_meta(node, metadata)[0]
    axis = int(_attribute(node, "axis", 1))
    if axis < 0:
        axis += len(value.shape)
    if len(node.inputs) != 1 or len(node.outputs) != 1 or not 0 <= axis <= len(value.shape):
        return False, "Flatten axis or arity is invalid"
    return True, "supported zero-copy view"


@lru_cache(maxsize=1)
def _kernels() -> dict[str, Any]:
    try:
        import triton
        import triton.language as tl
    except ImportError as error:
        raise RuntimeError("Triton kernels require triton>=3.6") from error

    @triton.jit
    def unary_kernel(x, output, count, BLOCK: tl.constexpr, OP: tl.constexpr):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < count
        value = tl.load(x + offsets, mask=mask, other=0.0).to(tl.float32)
        if OP == 0:
            # ONNX Relu preserves NaN; tl.maximum alone does not on every backend.
            result = tl.where(value != value, value, tl.maximum(value, 0.0))
        elif OP == 1:
            result = 0.5 * value * (1.0 + tl.erf(value * 0.7071067811865476))
        elif OP == 2:
            result = 1.0 / (1.0 + tl.exp(-value))
        else:
            result = value
        tl.store(output + offsets, result, mask=mask)

    @triton.jit
    def binary_kernel(a, b, output, count, BLOCK: tl.constexpr, OP: tl.constexpr):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < count
        left = tl.load(a + offsets, mask=mask, other=0.0)
        right = tl.load(b + offsets, mask=mask, other=0.0)
        if OP == 0:
            result = left + right
        elif OP == 1:
            result = left - right
        elif OP == 2:
            result = left * right
        else:
            result = left / right
        tl.store(output + offsets, result, mask=mask)

    @triton.jit
    def binary_broadcast_kernel(
        a,
        b,
        output,
        count,
        out0,
        out1,
        out2,
        out3,
        a0,
        a1,
        a2,
        a3,
        astride0,
        astride1,
        astride2,
        astride3,
        b0,
        b1,
        b2,
        b3,
        bstride0,
        bstride1,
        bstride2,
        bstride3,
        BLOCK: tl.constexpr,
        OP: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < count
        coordinate3 = offsets % out3
        remaining = offsets // out3
        coordinate2 = remaining % out2
        remaining //= out2
        coordinate1 = remaining % out1
        coordinate0 = remaining // out1
        a_offset = (
            tl.where(a0 == 1, 0, coordinate0) * astride0
            + tl.where(a1 == 1, 0, coordinate1) * astride1
            + tl.where(a2 == 1, 0, coordinate2) * astride2
            + tl.where(a3 == 1, 0, coordinate3) * astride3
        )
        b_offset = (
            tl.where(b0 == 1, 0, coordinate0) * bstride0
            + tl.where(b1 == 1, 0, coordinate1) * bstride1
            + tl.where(b2 == 1, 0, coordinate2) * bstride2
            + tl.where(b3 == 1, 0, coordinate3) * bstride3
        )
        left = tl.load(a + a_offset, mask=mask, other=0.0)
        right = tl.load(b + b_offset, mask=mask, other=0.0)
        if OP == 0:
            result = left + right
        elif OP == 1:
            result = left - right
        elif OP == 2:
            result = left * right
        else:
            result = left / right
        tl.store(output + offsets, result, mask=mask)

    @triton.jit
    def softmax_kernel(x, output, rows, columns, BLOCK: tl.constexpr):
        row = tl.program_id(0)
        offsets = tl.arange(0, BLOCK)
        mask = offsets < columns
        values = tl.load(x + row * columns + offsets, mask=mask, other=-float("inf")).to(tl.float32)
        values -= tl.max(values, axis=0)
        numerator = tl.exp(values)
        denominator = tl.sum(numerator, axis=0)
        tl.store(output + row * columns + offsets, numerator / denominator, mask=mask)

    @triton.jit
    def layer_norm_kernel(
        x,
        scale,
        bias,
        output,
        columns,
        epsilon,
        HAS_BIAS: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0)
        offsets = tl.arange(0, BLOCK)
        mask = offsets < columns
        values = tl.load(x + row * columns + offsets, mask=mask, other=0.0).to(tl.float32)
        mean = tl.sum(values, axis=0) / columns
        centered = tl.where(mask, values - mean, 0.0)
        variance = tl.sum(centered * centered, axis=0) / columns
        normalized = centered * tl.rsqrt(variance + epsilon)
        weights = tl.load(scale + offsets, mask=mask, other=0.0).to(tl.float32)
        result = normalized * weights
        if HAS_BIAS:
            result += tl.load(bias + offsets, mask=mask, other=0.0).to(tl.float32)
        tl.store(output + row * columns + offsets, result, mask=mask)

    @triton.jit
    def global_average_pool_kernel(x, output, spatial, BLOCK: tl.constexpr):
        row = tl.program_id(0)
        offsets = tl.arange(0, BLOCK)
        mask = offsets < spatial
        values = tl.load(x + row * spatial + offsets, mask=mask, other=0.0).to(tl.float32)
        tl.store(output + row, tl.sum(values, axis=0) / spatial)

    matmul_configs = [
        triton.Config(
            {"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 8}, num_warps=4, num_stages=3
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 8}, num_warps=4, num_stages=4
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 8}, num_warps=8, num_stages=3
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 8}, num_warps=8, num_stages=3
        ),
    ]

    @triton.jit
    def matmul_kernel(
        a,
        b,
        bias,
        output,
        m,
        n,
        k,
        stride_am,
        stride_ak,
        stride_bk,
        stride_bn,
        stride_cm,
        stride_cn,
        alpha,
        beta,
        BIAS_MODE: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        GROUP_M: tl.constexpr,
    ):
        pid = tl.program_id(0)
        grid_m = tl.cdiv(m, BLOCK_M)
        grid_n = tl.cdiv(n, BLOCK_N)
        group_width = GROUP_M * grid_n
        group_id = pid // group_width
        first_m = group_id * GROUP_M
        group_m = tl.minimum(grid_m - first_m, GROUP_M)
        pid_m = first_m + ((pid % group_width) % group_m)
        pid_n = (pid % group_width) // group_m
        offsets_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offsets_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offsets_k = tl.arange(0, BLOCK_K)
        accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for start in range(0, tl.cdiv(k, BLOCK_K)):
            current_k = start * BLOCK_K + offsets_k
            a_values = tl.load(
                a + offsets_m[:, None] * stride_am + current_k[None, :] * stride_ak,
                mask=(offsets_m[:, None] < m) & (current_k[None, :] < k),
                other=0.0,
            )
            b_values = tl.load(
                b + current_k[:, None] * stride_bk + offsets_n[None, :] * stride_bn,
                mask=(current_k[:, None] < k) & (offsets_n[None, :] < n),
                other=0.0,
            )
            accumulator = tl.dot(a_values, b_values, accumulator)
        output_offsets = offsets_m[:, None] * stride_cm + offsets_n[None, :] * stride_cn
        result = accumulator * alpha
        if BIAS_MODE == 1:
            result += tl.load(bias).to(tl.float32) * beta
        elif BIAS_MODE == 2:
            result += (
                tl.load(bias + offsets_n[None, :], mask=offsets_n[None, :] < n, other=0.0).to(
                    tl.float32
                )
                * beta
            )
        elif BIAS_MODE == 3:
            result += (
                tl.load(
                    bias + offsets_m[:, None] * n + offsets_n[None, :],
                    mask=(offsets_m[:, None] < m) & (offsets_n[None, :] < n),
                    other=0.0,
                ).to(tl.float32)
                * beta
            )
        tl.store(
            output + output_offsets,
            result,
            mask=(offsets_m[:, None] < m) & (offsets_n[None, :] < n),
        )

    tuned_matmul_kernel = triton.autotune(
        configs=matmul_configs,
        key=["m", "n", "k", "stride_am", "stride_ak", "stride_bk", "stride_bn", "BIAS_MODE"],
        cache_results=True,
    )(matmul_kernel)

    return {
        "unary": unary_kernel,
        "binary": binary_kernel,
        "binary_broadcast": binary_broadcast_kernel,
        "softmax": softmax_kernel,
        "layer_norm": layer_norm_kernel,
        "global_average_pool": global_average_pool_kernel,
        "matmul": tuned_matmul_kernel,
        "matmul_fixed": matmul_kernel,
        "triton": triton,
    }


def _execute_unary(node: Any, inputs: list[Any], metadata: Mapping[str, Any]) -> list[Any]:
    torch = __import__("torch")
    kernels = _kernels()
    source = inputs[0]
    if not source.is_contiguous():
        raise ValueError("Triton unary kernels require contiguous tensors")
    output = torch.empty_like(source)
    operations = {"Relu": 0, "Gelu": 1, "Sigmoid": 2, "Identity": 3}
    count = source.numel()
    grid = (kernels["triton"].cdiv(count, 256),)
    kernels["unary"][grid](
        source, output, count, BLOCK=256, OP=operations[node.op_type], num_warps=4
    )
    return [output]


def _execute_binary(node: Any, inputs: list[Any], metadata: Mapping[str, Any]) -> list[Any]:
    torch = __import__("torch")
    kernels = _kernels()
    left, right = inputs
    if not left.is_contiguous() or not right.is_contiguous():
        raise ValueError("Triton binary kernels require contiguous tensors")
    operations = {"Add": 0, "Sub": 1, "Mul": 2, "Div": 3}
    output_shape = torch.broadcast_shapes(left.shape, right.shape)
    output = torch.empty(output_shape, dtype=left.dtype, device=left.device)
    count = output.numel()
    grid = (kernels["triton"].cdiv(count, 256),)
    if left.shape == right.shape:
        kernels["binary"][grid](
            left, right, output, count, BLOCK=256, OP=operations[node.op_type], num_warps=4
        )
    else:
        output_dims = (1,) * (4 - output.ndim) + tuple(output.shape)
        left_dims = (1,) * (4 - left.ndim) + tuple(left.shape)
        right_dims = (1,) * (4 - right.ndim) + tuple(right.shape)
        left_strides = (0,) * (4 - left.ndim) + tuple(left.stride())
        right_strides = (0,) * (4 - right.ndim) + tuple(right.stride())
        kernels["binary_broadcast"][grid](
            left,
            right,
            output,
            count,
            *output_dims,
            *left_dims,
            *left_strides,
            *right_dims,
            *right_strides,
            BLOCK=256,
            OP=operations[node.op_type],
            num_warps=4,
        )
    return [output]


def _execute_softmax(node: Any, inputs: list[Any], metadata: Mapping[str, Any]) -> list[Any]:
    torch = __import__("torch")
    kernels = _kernels()
    source = inputs[0]
    if not source.is_contiguous():
        raise ValueError("Triton softmax requires a contiguous tensor")
    columns = source.shape[-1]
    block = kernels["triton"].next_power_of_2(columns)
    if block > 65_536:
        raise ValueError("softmax axis exceeds v1 block limit")
    rows = source.numel() // columns
    output = torch.empty_like(source)
    warps = 8 if block >= 2_048 else 4
    kernels["softmax"][(rows,)](source, output, rows, columns, BLOCK=block, num_warps=warps)
    return [output]


def _execute_matmul(
    node: Any, inputs: list[Any], metadata: Mapping[str, Any], autotune: bool = True
) -> list[Any]:
    torch = __import__("torch")
    kernels = _kernels()
    left, right = inputs
    if left.ndim != 2 or right.ndim != 2:
        raise ValueError("v1 Triton matmul requires rank-2 tensors")
    m, k = left.shape
    other_k, n = right.shape
    if k != other_k:
        raise ValueError("matmul inner dimensions do not match")
    output = torch.empty((m, n), dtype=left.dtype, device=left.device)

    def grid(meta: Mapping[str, Any]) -> tuple[int]:
        return (
            kernels["triton"].cdiv(m, meta["BLOCK_M"]) * kernels["triton"].cdiv(n, meta["BLOCK_N"]),
        )

    kernel = kernels["matmul"] if autotune else kernels["matmul_fixed"]
    launch_options = (
        {}
        if autotune
        else {
            "BLOCK_M": 64,
            "BLOCK_N": 64,
            "BLOCK_K": 32,
            "GROUP_M": 8,
            "num_warps": 4,
            "num_stages": 3,
        }
    )
    kernel[grid](
        left,
        right,
        left,
        output,
        m,
        n,
        k,
        left.stride(0),
        left.stride(1),
        right.stride(0),
        right.stride(1),
        output.stride(0),
        output.stride(1),
        1.0,
        0.0,
        BIAS_MODE=0,
        **launch_options,
    )
    return [output]


def _execute_gemm(
    node: Any, inputs: list[Any], metadata: Mapping[str, Any], autotune: bool = True
) -> list[Any]:
    torch = __import__("torch")
    kernels = _kernels()
    left, right = inputs[:2]
    bias = inputs[2] if len(inputs) == 3 else left
    if any(not tensor.is_contiguous() for tensor in inputs):
        raise ValueError("Triton Gemm requires contiguous input tensors")
    trans_a = bool(_attribute(node, "transA", 0))
    trans_b = bool(_attribute(node, "transB", 0))
    m, k = (left.shape[1], left.shape[0]) if trans_a else left.shape
    other_k, n = (right.shape[1], right.shape[0]) if trans_b else right.shape
    if k != other_k:
        raise ValueError("Gemm inner dimensions do not match")
    stride_am, stride_ak = (
        (left.stride(1), left.stride(0)) if trans_a else (left.stride(0), left.stride(1))
    )
    stride_bk, stride_bn = (
        (right.stride(1), right.stride(0)) if trans_b else (right.stride(0), right.stride(1))
    )
    bias_mode = 0
    if len(inputs) == 3:
        if bias.numel() == 1:
            bias_mode = 1
        elif bias.numel() == n:
            bias_mode = 2
        elif tuple(bias.shape) == (m, n):
            bias_mode = 3
        else:
            raise ValueError("unsupported Gemm bias shape")
    output = torch.empty((m, n), dtype=left.dtype, device=left.device)

    def grid(meta: Mapping[str, Any]) -> tuple[int]:
        return (
            kernels["triton"].cdiv(m, meta["BLOCK_M"]) * kernels["triton"].cdiv(n, meta["BLOCK_N"]),
        )

    kernel = kernels["matmul"] if autotune else kernels["matmul_fixed"]
    launch_options = (
        {}
        if autotune
        else {
            "BLOCK_M": 64,
            "BLOCK_N": 64,
            "BLOCK_K": 32,
            "GROUP_M": 8,
            "num_warps": 4,
            "num_stages": 3,
        }
    )
    kernel[grid](
        left,
        right,
        bias,
        output,
        m,
        n,
        k,
        stride_am,
        stride_ak,
        stride_bk,
        stride_bn,
        output.stride(0),
        output.stride(1),
        float(_attribute(node, "alpha", 1.0)),
        float(_attribute(node, "beta", 1.0)),
        BIAS_MODE=bias_mode,
        **launch_options,
    )
    return [output]


def _execute_layer_norm(node: Any, inputs: list[Any], metadata: Mapping[str, Any]) -> list[Any]:
    torch = __import__("torch")
    kernels = _kernels()
    source, scale = inputs[:2]
    bias = inputs[2] if len(inputs) == 3 else scale
    if any(not tensor.is_contiguous() for tensor in inputs):
        raise ValueError("Triton LayerNormalization requires contiguous tensors")
    columns = source.shape[-1]
    rows = source.numel() // columns
    block = kernels["triton"].next_power_of_2(columns)
    output = torch.empty_like(source)
    kernels["layer_norm"][(rows,)](
        source,
        scale,
        bias,
        output,
        columns,
        float(_attribute(node, "epsilon", 1e-5)),
        HAS_BIAS=len(inputs) == 3,
        BLOCK=block,
        num_warps=8 if block >= 2048 else 4,
    )
    return [output]


def _execute_global_average_pool(
    node: Any, inputs: list[Any], metadata: Mapping[str, Any]
) -> list[Any]:
    torch = __import__("torch")
    kernels = _kernels()
    source = inputs[0]
    if not source.is_contiguous() or source.ndim != 4:
        raise ValueError("Triton GlobalAveragePool requires contiguous NCHW input")
    batch, channels, height, width = source.shape
    spatial = height * width
    block = kernels["triton"].next_power_of_2(spatial)
    if block > 65_536:
        raise ValueError("GlobalAveragePool spatial area exceeds the v1 block limit")
    output = torch.empty((batch, channels, 1, 1), dtype=source.dtype, device=source.device)
    kernels["global_average_pool"][(batch * channels,)](
        source, output, spatial, BLOCK=block, num_warps=8 if block >= 2048 else 4
    )
    return [output]


def _execute_flatten(node: Any, inputs: list[Any], metadata: Mapping[str, Any]) -> list[Any]:
    source = inputs[0]
    if not source.is_contiguous():
        raise ValueError("zero-copy Flatten requires a contiguous tensor")
    axis = int(_attribute(node, "axis", 1))
    if axis < 0:
        axis += source.ndim
    rows = math.prod(source.shape[:axis])
    columns = math.prod(source.shape[axis:])
    return [source.view(rows, columns)]


def create_default_registry(*, autotune: bool = True) -> KernelRegistry:
    registry = KernelRegistry()
    registry.register(
        KernelSpec(
            "unary.contiguous",
            ("Relu", "Gelu", "Sigmoid", "Identity"),
            _execute_unary,
            _floating_contiguous_capability,
        )
    )
    registry.register(
        KernelSpec(
            "binary.broadcast_rank4",
            ("Add", "Sub", "Mul", "Div"),
            _execute_binary,
            _binary_capability,
        )
    )
    registry.register(
        KernelSpec(
            "softmax.last_axis", ("Softmax",), _execute_softmax, _softmax_capability, min_opset=13
        )
    )

    def execute_matmul(node: Any, inputs: list[Any], metadata: Mapping[str, Any]) -> list[Any]:
        return _execute_matmul(node, inputs, metadata, autotune=autotune)

    def execute_gemm(node: Any, inputs: list[Any], metadata: Mapping[str, Any]) -> list[Any]:
        return _execute_gemm(node, inputs, metadata, autotune=autotune)

    registry.register(KernelSpec("matmul.rank2", ("MatMul",), execute_matmul, _matmul_capability))
    registry.register(KernelSpec("gemm.rank2", ("Gemm",), execute_gemm, _gemm_capability))
    registry.register(
        KernelSpec(
            "layer_norm.last_axis",
            ("LayerNormalization",),
            _execute_layer_norm,
            _layer_norm_capability,
            min_opset=17,
        )
    )
    registry.register(
        KernelSpec(
            "pool.global_average_nchw",
            ("GlobalAveragePool",),
            _execute_global_average_pool,
            _global_average_pool_capability,
        )
    )
    registry.register(
        KernelSpec("view.flatten", ("Flatten",), _execute_flatten, _flatten_capability)
    )
    return registry
