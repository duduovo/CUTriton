"""Compile CUTriton's stable FP32 kernel ABI to PTX and a JSON manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import triton
import triton.language as tl
from triton.compiler import ASTSource
from triton.backends.compiler import GPUTarget


@triton.jit
def fused_conv_batch_norm_relu(
    x, weight, scale, bias, mean, variance, output,
    n, c, h, w, k, r, s, out_h, out_w,
    stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w, epsilon,
):
    offsets = tl.program_id(0) * 256 + tl.arange(0, 256)
    total = n * k * out_h * out_w
    mask = offsets < total
    ow = offsets % out_w
    q = offsets // out_w
    oh = q % out_h
    q = q // out_h
    oc = q % k
    batch = q // k
    accumulator = tl.zeros((256,), dtype=tl.float32)
    for linear in tl.range(0, c * r * s):
        kw = linear % s
        q2 = linear // s
        kh = q2 % r
        ic = q2 // r
        ih = oh * stride_h - pad_h + kh * dilation_h
        iw = ow * stride_w - pad_w + kw * dilation_w
        in_bounds = mask & (ih >= 0) & (ih < h) & (iw >= 0) & (iw < w)
        x_offset = ((batch * c + ic) * h + ih) * w + iw
        w_offset = ((oc * c + ic) * r + kh) * s + kw
        accumulator += tl.load(x + x_offset, mask=in_bounds, other=0.0) * tl.load(
            weight + w_offset, mask=mask, other=0.0
        )
    normalized = (
        (accumulator - tl.load(mean + oc, mask=mask, other=0.0))
        * tl.rsqrt(tl.load(variance + oc, mask=mask, other=0.0) + epsilon)
        * tl.load(scale + oc, mask=mask, other=0.0)
        + tl.load(bias + oc, mask=mask, other=0.0)
    )
    tl.store(output + offsets, tl.maximum(normalized, 0.0), mask=mask)


@triton.jit
def fused_conv_batch_norm(
    x, weight, scale, bias, mean, variance, output,
    n, c, h, w, k, r, s, out_h, out_w,
    stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w, epsilon,
):
    offsets = tl.program_id(0) * 256 + tl.arange(0, 256)
    total = n * k * out_h * out_w
    mask = offsets < total
    ow = offsets % out_w
    q = offsets // out_w
    oh = q % out_h
    q = q // out_h
    oc = q % k
    batch = q // k
    accumulator = tl.zeros((256,), dtype=tl.float32)
    for linear in tl.range(0, c * r * s):
        kw = linear % s
        q2 = linear // s
        kh = q2 % r
        ic = q2 // r
        ih = oh * stride_h - pad_h + kh * dilation_h
        iw = ow * stride_w - pad_w + kw * dilation_w
        in_bounds = mask & (ih >= 0) & (ih < h) & (iw >= 0) & (iw < w)
        x_offset = ((batch * c + ic) * h + ih) * w + iw
        w_offset = ((oc * c + ic) * r + kh) * s + kw
        accumulator += tl.load(x + x_offset, mask=in_bounds, other=0.0) * tl.load(
            weight + w_offset, mask=mask, other=0.0
        )
    normalized = (
        (accumulator - tl.load(mean + oc, mask=mask, other=0.0))
        * tl.rsqrt(tl.load(variance + oc, mask=mask, other=0.0) + epsilon)
        * tl.load(scale + oc, mask=mask, other=0.0)
        + tl.load(bias + oc, mask=mask, other=0.0)
    )
    tl.store(output + offsets, normalized, mask=mask)


@triton.jit
def max_pool(
    x, output, n, c, h, w, out_h, out_w, kernel_h, kernel_w,
    stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w,
):
    offsets = tl.program_id(0) * 256 + tl.arange(0, 256)
    total = n * c * out_h * out_w
    mask = offsets < total
    ow = offsets % out_w
    q = offsets // out_w
    oh = q % out_h
    q = q // out_h
    channel = q % c
    batch = q // c
    accumulator = tl.full((256,), -float("inf"), dtype=tl.float32)
    for linear in tl.range(0, kernel_h * kernel_w):
        kw = linear % kernel_w
        kh = linear // kernel_w
        ih = oh * stride_h - pad_h + kh * dilation_h
        iw = ow * stride_w - pad_w + kw * dilation_w
        in_bounds = mask & (ih >= 0) & (ih < h) & (iw >= 0) & (iw < w)
        x_offset = ((batch * c + channel) * h + ih) * w + iw
        value = tl.load(x + x_offset, mask=in_bounds, other=-float("inf"))
        accumulator = tl.maximum(accumulator, value)
    tl.store(output + offsets, accumulator, mask=mask)


@triton.jit
def add_relu(a, b, output, total):
    offsets = tl.program_id(0) * 256 + tl.arange(0, 256)
    mask = offsets < total
    value = tl.load(a + offsets, mask=mask, other=0.0)
    value += tl.load(b + offsets, mask=mask, other=0.0)
    tl.store(output + offsets, tl.maximum(value, 0.0), mask=mask)


@triton.jit
def global_average_pool(x, output, n, c, h, w):
    indices = tl.program_id(0) * 256 + tl.arange(0, 256)
    total = n * c
    mask = indices < total
    batch = indices // c
    channel = indices % c
    accumulator = tl.zeros((256,), dtype=tl.float32)
    spatial = h * w
    for offset in tl.range(0, spatial):
        accumulator += tl.load(
            x + ((batch * c + channel) * h * w + offset),
            mask=mask,
            other=0.0,
        )
    tl.store(output + indices, accumulator / spatial, mask=mask)


@triton.jit
def gemm(a, b, output, m, n, k):
    indices = tl.program_id(0) * 256 + tl.arange(0, 256)
    total = m * n
    mask = indices < total
    row = indices // n
    column = indices % n
    accumulator = tl.zeros((256,), dtype=tl.float32)
    for inner in tl.range(0, k):
        accumulator += tl.load(a + row * k + inner, mask=mask, other=0.0) * tl.load(
            b + inner * n + column, mask=mask, other=0.0
        )
    tl.store(output + indices, accumulator, mask=mask)


KERNELS = [
    (
        "FusedConvBatchNormRelu",
        fused_conv_batch_norm_relu,
        {
            "x": "*fp32", "weight": "*fp32", "scale": "*fp32",
            "bias": "*fp32", "mean": "*fp32", "variance": "*fp32",
            "output": "*fp32", "n": "i32", "c": "i32", "h": "i32",
            "w": "i32", "k": "i32", "r": "i32", "s": "i32",
            "out_h": "i32", "out_w": "i32", "stride_h": "i32",
            "stride_w": "i32", "pad_h": "i32", "pad_w": "i32",
            "dilation_h": "i32", "dilation_w": "i32", "epsilon": "fp32",
        },
        "NCHW",
    ),
    (
        "FusedConvBatchNorm",
        fused_conv_batch_norm,
        {
            "x": "*fp32", "weight": "*fp32", "scale": "*fp32",
            "bias": "*fp32", "mean": "*fp32", "variance": "*fp32",
            "output": "*fp32", "n": "i32", "c": "i32", "h": "i32",
            "w": "i32", "k": "i32", "r": "i32", "s": "i32",
            "out_h": "i32", "out_w": "i32", "stride_h": "i32",
            "stride_w": "i32", "pad_h": "i32", "pad_w": "i32",
            "dilation_h": "i32", "dilation_w": "i32", "epsilon": "fp32",
        },
        "NCHW",
    ),
    (
        "GlobalAveragePool",
        global_average_pool,
        {"x": "*fp32", "output": "*fp32", "n": "i32", "c": "i32",
         "h": "i32", "w": "i32"},
        "NCHW",
    ),
    (
        "MaxPool",
        max_pool,
        {
            "x": "*fp32", "output": "*fp32", "n": "i32", "c": "i32",
            "h": "i32", "w": "i32", "out_h": "i32", "out_w": "i32",
            "kernel_h": "i32", "kernel_w": "i32", "stride_h": "i32",
            "stride_w": "i32", "pad_h": "i32", "pad_w": "i32",
            "dilation_h": "i32", "dilation_w": "i32",
        },
        "NCHW",
    ),
    (
        "AddRelu",
        add_relu,
        {"a": "*fp32", "b": "*fp32", "output": "*fp32", "total": "i32"},
        "NCHW",
    ),
    (
        "Gemm",
        gemm,
        {"a": "*fp32", "b": "*fp32", "output": "*fp32", "m": "i32",
         "n": "i32", "k": "i32"},
        "",
    ),
]

# Triton 3.6 appends two pointer-sized runtime parameters to every CUDA entry
# point. They are reserved for runtime/profile scratch storage and are unused
# by the kernels above, but CUDA's kernelParams array must still contain them.
TRITON_RUNTIME_ABI_SUFFIX = ["*i8", "*i8"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    entries = []
    for op_type, function, signature, layout in KERNELS:
        source = ASTSource(function, signature)
        compiled = triton.compile(
            source,
            target=GPUTarget("cuda", 80, 32),
            options={"num_warps": 4, "num_stages": 1},
        )
        ptx = compiled.asm["ptx"]
        filename = f"{op_type}.ptx"
        (args.output / filename).write_text(ptx, encoding="utf-8")
        entries.append(
            {
                "op_type": op_type,
                "symbol": compiled.name,
                "ptx": filename,
                "dtype": "float32",
                "layout": layout,
                "min_compute_capability": 80,
                "num_warps": 4,
                "shared_memory_bytes": int(compiled.metadata.shared),
                "sha256": hashlib.sha256(ptx.encode("utf-8")).hexdigest(),
                "abi": list(signature.values()) + TRITON_RUNTIME_ABI_SUFFIX,
            }
        )

    manifest = {
        "schema_version": 1,
        "triton_version": triton.__version__,
        "kernels": entries,
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
