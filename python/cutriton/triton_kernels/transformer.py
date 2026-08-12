"""AOT FP16 Transformer kernels used by the C++ runtime.

The module intentionally targets the bounded BERT-tiny shapes used by the
benchmark.  Unsupported shapes remain explicit compiler errors rather than
silently selecting a numerically invalid kernel.
"""

from __future__ import annotations

import triton
import triton.language as tl
from triton.language.extra import libdevice

from cutriton.kernel_sdk import ArgumentSpec, KernelSpec, KernelVariant, register


@triton.jit
def gemm_fp16(
    a,
    b,
    output,
    m,
    n,
    k,
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
        left = tl.load(
            a + offsets_m[:, None] * k + current_k[None, :],
            mask=(offsets_m[:, None] < m) & (current_k[None, :] < k),
            other=0.0,
        )
        right = tl.load(
            b + current_k[:, None] * n + offsets_n[None, :],
            mask=(current_k[:, None] < k) & (offsets_n[None, :] < n),
            other=0.0,
        )
        accumulator = tl.dot(left, right, accumulator)
    tl.store(
        output + offsets_m[:, None] * n + offsets_n[None, :],
        accumulator,
        mask=(offsets_m[:, None] < m) & (offsets_n[None, :] < n),
    )


@triton.jit
def gemm_gelu_fp16(
    a,
    b,
    output,
    m,
    n,
    k,
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
        left = tl.load(
            a + offsets_m[:, None] * k + current_k[None, :],
            mask=(offsets_m[:, None] < m) & (current_k[None, :] < k),
            other=0.0,
        )
        right = tl.load(
            b + current_k[:, None] * n + offsets_n[None, :],
            mask=(current_k[:, None] < k) & (offsets_n[None, :] < n),
            other=0.0,
        )
        accumulator = tl.dot(left, right, accumulator)
    value = 0.5 * accumulator * (1.0 + libdevice.erf(accumulator * 0.7071067811865476))
    tl.store(
        output + offsets_m[:, None] * n + offsets_n[None, :],
        value,
        mask=(offsets_m[:, None] < m) & (offsets_n[None, :] < n),
    )


@triton.jit
def gelu_fp16(x, output, total, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < total
    value = tl.load(x + offsets, mask=mask, other=0.0).to(tl.float32)
    result = 0.5 * value * (1.0 + libdevice.erf(value * 0.7071067811865476))
    tl.store(output + offsets, result, mask=mask)


@triton.jit
def add_fp16(a, b, output, total, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < total
    value = tl.load(a + offsets, mask=mask, other=0.0)
    value += tl.load(b + offsets, mask=mask, other=0.0)
    tl.store(output + offsets, value, mask=mask)


@triton.jit
def layer_norm_fp16(
    x,
    scale,
    bias,
    output,
    rows,
    columns,
    epsilon,
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
    offsets_bias = tl.load(bias + offsets, mask=mask, other=0.0).to(tl.float32)
    tl.store(output + row * columns + offsets, normalized * weights + offsets_bias, mask=mask)


@triton.jit
def skip_layer_norm_fp16(
    x,
    residual,
    scale,
    bias,
    output,
    rows,
    columns,
    epsilon,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK)
    mask = offsets < columns
    values = tl.load(x + row * columns + offsets, mask=mask, other=0.0).to(tl.float32)
    values += tl.load(residual + row * columns + offsets, mask=mask, other=0.0).to(
        tl.float32
    )
    mean = tl.sum(values, axis=0) / columns
    centered = tl.where(mask, values - mean, 0.0)
    variance = tl.sum(centered * centered, axis=0) / columns
    normalized = centered * tl.rsqrt(variance + epsilon)
    weights = tl.load(scale + offsets, mask=mask, other=0.0).to(tl.float32)
    offsets_bias = tl.load(bias + offsets, mask=mask, other=0.0).to(tl.float32)
    tl.store(output + row * columns + offsets, normalized * weights + offsets_bias, mask=mask)


@triton.jit
def softmax_fp16(x, output, rows, columns, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK)
    mask = offsets < columns
    values = tl.load(x + row * columns + offsets, mask=mask, other=-float("inf")).to(
        tl.float32
    )
    values -= tl.max(values, axis=0)
    numerator = tl.exp(values)
    tl.store(output + row * columns + offsets, numerator / tl.sum(numerator, axis=0), mask=mask)


def arg(argument_name: str, type_name: str, kind: str, **source: object) -> ArgumentSpec:
    return ArgumentSpec(argument_name, type_name, {"kind": kind, **source})


def reserved() -> tuple[ArgumentSpec, ArgumentSpec]:
    return (
        arg("runtime_scratch", "*i8", "runtime_reserved"),
        arg("profile_scratch", "*i8", "runtime_reserved"),
    )


def signature(arguments: tuple[ArgumentSpec, ...]) -> dict[str, str]:
    return {
        item.name: item.type
        for item in arguments
        if item.source["kind"] != "runtime_reserved"
    }


def numel_grid(block: int = 256) -> dict[str, object]:
    return {
        "op": "ceil_div",
        "value": {"kind": "output_numel", "index": 0},
        "divisor": block,
    }


def tiled_grid() -> tuple[dict[str, object], ...]:
    def ceil_dim(axis: int, meta: str) -> dict[str, object]:
        return {
            "kind": "ceil_div",
            "args": [
                {"kind": "output_dim", "index": 0, "axis": axis},
                {"kind": "meta", "name": meta},
            ],
        }

    return (
        {"kind": "mul", "args": [ceil_dim(0, "BLOCK_M"), ceil_dim(1, "BLOCK_N")]},
    )


gemm_variants = (
    KernelVariant(
        "m32_n64_k32", num_warps=4, num_stages=3,
        meta={"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 8},
    ),
    KernelVariant(
        "m64_n64_k32", num_warps=4, num_stages=4,
        meta={"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 8},
        default=True,
    ),
    KernelVariant(
        "m64_n128_k32", num_warps=8, num_stages=3,
        meta={"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 8},
    ),
    KernelVariant(
        "m128_n128_k32", num_warps=8, num_stages=3,
        meta={"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 8},
    ),
)


gemm_arguments = (
    arg("a", "*fp16", "input", index=0),
    arg("b", "*fp16", "input", index=1),
    arg("output", "*fp16", "output", index=0),
    arg("m", "i32", "output_dim", index=0, axis=0),
    arg("n", "i32", "output_dim", index=0, axis=1),
    arg("k", "i32", "input_dim", index=0, axis=1),
) + reserved()

gemm_constraints = (
    {"kind": "input_count_eq", "value": 2},
    {"kind": "attribute_i64_eq", "name": "transA", "value": 0, "default": 0},
    {"kind": "attribute_i64_eq", "name": "transB", "value": 0, "default": 0},
    {"kind": "tensor_rank_eq", "source": "input", "index": 0, "value": 2},
    {"kind": "tensor_rank_eq", "source": "input", "index": 1, "value": 2},
)

for kernel_id, op_type, function in (
    ("gemm.fp16.tensor_core", "Gemm", gemm_fp16),
    ("gemm_gelu.fp16.tensor_core", "GemmGelu", gemm_gelu_fp16),
):
    register(
        KernelSpec(
            kernel_id,
            op_type,
            1,
            function,
            signature(gemm_arguments),
            gemm_arguments,
            tiled_grid(),
            gemm_variants,
            dtype="float16",
            layout="",
            min_compute_capability=80,
            constraints=gemm_constraints,
        )
    )


def elementwise_spec(kernel_id: str, op_type: str, function: object, input_count: int) -> None:
    names = ("x",) if input_count == 1 else ("a", "b")
    arguments = tuple(
        arg(name, "*fp16", "input", index=index) for index, name in enumerate(names)
    ) + (
        arg("output", "*fp16", "output", index=0),
        arg("total", "i32", "output_numel", index=0),
    ) + reserved()
    register(
        KernelSpec(
            kernel_id,
            op_type,
            1,
            function,
            signature(arguments),
            arguments,
            numel_grid(),
            (KernelVariant("warps4", num_warps=4, meta={"BLOCK": 256}, default=True),),
            dtype="float16",
            layout="",
        )
    )


elementwise_spec("gelu.fp16.contiguous", "Gelu", gelu_fp16, 1)
elementwise_spec("add.fp16.contiguous", "Add", add_fp16, 2)


def norm_spec(kernel_id: str, op_type: str, function: object, skip: bool) -> None:
    input_names = ("x", "residual", "scale", "bias") if skip else ("x", "scale", "bias")
    arguments = tuple(
        arg(name, "*fp16", "input", index=index)
        for index, name in enumerate(input_names)
    ) + (
        arg("output", "*fp16", "output", index=0),
        arg("rows", "i32", "input_dim", index=0, axis=0),
        arg("columns", "i32", "input_dim", index=0, axis=1),
        arg("epsilon", "fp32", "attribute_f64", name="epsilon", default=1e-5),
    ) + reserved()
    register(
        KernelSpec(
            kernel_id,
            op_type,
            1,
            function,
            signature(arguments),
            arguments,
            ({"kind": "output_dim", "index": 0, "axis": 0},),
            (
                KernelVariant(
                    "block1024_warps4", num_warps=4,
                    meta={"BLOCK": 1024}, default=True,
                ),
                KernelVariant(
                    "block1024_warps8", num_warps=8,
                    meta={"BLOCK": 1024},
                ),
            ),
            dtype="float16",
            layout="",
            constraints=(
                {"kind": "input_count_eq", "value": len(input_names)},
                {"kind": "tensor_rank_eq", "source": "input", "index": 0, "value": 2},
            ),
        )
    )


norm_spec(
    "layer_norm.fp16.last_axis", "LayerNormalization", layer_norm_fp16, skip=False
)
norm_spec(
    "skip_layer_norm.fp16.last_axis",
    "SkipLayerNormalization",
    skip_layer_norm_fp16,
    skip=True,
)


softmax_arguments = (
    arg("x", "*fp16", "input", index=0),
    arg("output", "*fp16", "output", index=0),
    arg("rows", "i32", "input_dim", index=0, axis=0),
    arg("columns", "i32", "input_dim", index=0, axis=1),
) + reserved()
register(
    KernelSpec(
        "softmax.fp16.last_axis",
        "Softmax",
        1,
        softmax_fp16,
        signature(softmax_arguments),
        softmax_arguments,
        ({"kind": "output_dim", "index": 0, "axis": 0},),
        (KernelVariant("block1024", num_warps=8, meta={"BLOCK": 1024}, default=True),),
        dtype="float16",
        layout="",
        constraints=(
            {"kind": "tensor_rank_eq", "source": "input", "index": 0, "value": 2},
        ),
    )
)
