from __future__ import annotations

import triton
import triton.language as tl

from cutriton.kernel_sdk import ArgumentSpec, KernelSpec, KernelVariant, register


BLOCK = 256


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
    q //= out_h
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
        inside = mask & (ih >= 0) & (ih < h) & (iw >= 0) & (iw < w)
        x_offset = ((batch * c + ic) * h + ih) * w + iw
        weight_offset = ((oc * c + ic) * r + kh) * s + kw
        accumulator += tl.load(x + x_offset, mask=inside, other=0.0) * tl.load(
            weight + weight_offset, mask=mask, other=0.0
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
    q //= out_h
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
        inside = mask & (ih >= 0) & (ih < h) & (iw >= 0) & (iw < w)
        x_offset = ((batch * c + ic) * h + ih) * w + iw
        weight_offset = ((oc * c + ic) * r + kh) * s + kw
        accumulator += tl.load(x + x_offset, mask=inside, other=0.0) * tl.load(
            weight + weight_offset, mask=mask, other=0.0
        )
    normalized = (
        (accumulator - tl.load(mean + oc, mask=mask, other=0.0))
        * tl.rsqrt(tl.load(variance + oc, mask=mask, other=0.0) + epsilon)
        * tl.load(scale + oc, mask=mask, other=0.0)
        + tl.load(bias + oc, mask=mask, other=0.0)
    )
    tl.store(output + offsets, normalized, mask=mask)


@triton.jit
def conv(x, weight, output, n, c, h, w, k, r, s, out_h, out_w,
         stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w):
    offsets = tl.program_id(0) * 256 + tl.arange(0, 256)
    total = n * k * out_h * out_w
    mask = offsets < total
    ow = offsets % out_w
    q = offsets // out_w
    oh = q % out_h
    q //= out_h
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
        inside = mask & (ih >= 0) & (ih < h) & (iw >= 0) & (iw < w)
        x_offset = ((batch * c + ic) * h + ih) * w + iw
        weight_offset = ((oc * c + ic) * r + kh) * s + kw
        accumulator += tl.load(x + x_offset, mask=inside, other=0.0) * tl.load(
            weight + weight_offset, mask=mask, other=0.0
        )
    tl.store(output + offsets, accumulator, mask=mask)


@triton.jit
def batch_norm(x, scale, bias, mean, variance, output, total, channels,
               spatial, epsilon):
    offsets = tl.program_id(0) * 256 + tl.arange(0, 256)
    mask = offsets < total
    channel = (offsets // spatial) % channels
    value = tl.load(x + offsets, mask=mask, other=0.0)
    normalized = (
        (value - tl.load(mean + channel, mask=mask, other=0.0))
        * tl.rsqrt(tl.load(variance + channel, mask=mask, other=0.0) + epsilon)
        * tl.load(scale + channel, mask=mask, other=0.0)
        + tl.load(bias + channel, mask=mask, other=0.0)
    )
    tl.store(output + offsets, normalized, mask=mask)


@triton.jit
def relu(x, output, total):
    offsets = tl.program_id(0) * 256 + tl.arange(0, 256)
    mask = offsets < total
    value = tl.load(x + offsets, mask=mask, other=0.0)
    tl.store(output + offsets, tl.maximum(value, 0.0), mask=mask)


@triton.jit
def add(a, b, output, total):
    offsets = tl.program_id(0) * 256 + tl.arange(0, 256)
    mask = offsets < total
    value = tl.load(a + offsets, mask=mask, other=0.0)
    value += tl.load(b + offsets, mask=mask, other=0.0)
    tl.store(output + offsets, value, mask=mask)


@triton.jit
def add_relu(a, b, output, total):
    offsets = tl.program_id(0) * 256 + tl.arange(0, 256)
    mask = offsets < total
    value = tl.load(a + offsets, mask=mask, other=0.0)
    value += tl.load(b + offsets, mask=mask, other=0.0)
    tl.store(output + offsets, tl.maximum(value, 0.0), mask=mask)


@triton.jit
def max_pool(x, output, n, c, h, w, out_h, out_w, kernel_h, kernel_w,
             stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w):
    offsets = tl.program_id(0) * 256 + tl.arange(0, 256)
    total = n * c * out_h * out_w
    mask = offsets < total
    ow = offsets % out_w
    q = offsets // out_w
    oh = q % out_h
    q //= out_h
    channel = q % c
    batch = q // c
    accumulator = tl.full((256,), -float("inf"), dtype=tl.float32)
    for linear in tl.range(0, kernel_h * kernel_w):
        kw = linear % kernel_w
        kh = linear // kernel_w
        ih = oh * stride_h - pad_h + kh * dilation_h
        iw = ow * stride_w - pad_w + kw * dilation_w
        inside = mask & (ih >= 0) & (ih < h) & (iw >= 0) & (iw < w)
        x_offset = ((batch * c + channel) * h + ih) * w + iw
        accumulator = tl.maximum(
            accumulator, tl.load(x + x_offset, mask=inside, other=-float("inf"))
        )
    tl.store(output + offsets, accumulator, mask=mask)


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
            x + ((batch * c + channel) * spatial + offset), mask=mask, other=0.0
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


def arg(arg_name: str, type_name: str, kind: str, **source: object) -> ArgumentSpec:
    return ArgumentSpec(arg_name, type_name, {"kind": kind, **source})


def reserved() -> tuple[ArgumentSpec, ArgumentSpec]:
    return (
        arg("runtime_scratch", "*i8", "runtime_reserved"),
        arg("profile_scratch", "*i8", "runtime_reserved"),
    )


def variants() -> tuple[KernelVariant, ...]:
    return (
        KernelVariant("warps2", num_warps=2),
        KernelVariant("warps4", num_warps=4, default=True),
    )


def grid() -> dict[str, object]:
    return {
        "op": "ceil_div",
        "value": {"kind": "output_numel", "index": 0},
        "divisor": BLOCK,
    }


conv_dimensions = (
    arg("n", "i32", "input_dim", index=0, axis=0),
    arg("c", "i32", "input_dim", index=0, axis=1),
    arg("h", "i32", "input_dim", index=0, axis=2),
    arg("w", "i32", "input_dim", index=0, axis=3),
    arg("k", "i32", "input_dim", index=1, axis=0),
    arg("r", "i32", "input_dim", index=1, axis=2),
    arg("s", "i32", "input_dim", index=1, axis=3),
    arg("out_h", "i32", "output_dim", index=0, axis=2),
    arg("out_w", "i32", "output_dim", index=0, axis=3),
    arg("stride_h", "i32", "attribute_i64", name="strides", index=0, default=1),
    arg("stride_w", "i32", "attribute_i64", name="strides", index=1, default=1),
    arg("pad_h", "i32", "attribute_i64", name="pads", index=0, default=0),
    arg("pad_w", "i32", "attribute_i64", name="pads", index=1, default=0),
    arg("dilation_h", "i32", "attribute_i64", name="dilations", index=0, default=1),
    arg("dilation_w", "i32", "attribute_i64", name="dilations", index=1, default=1),
)


def register_conv_family() -> None:
    tensor_args = tuple(arg(name, "*fp32", "input", index=index) for index, name in enumerate(
        ("x", "weight", "scale", "bias", "mean", "variance")
    ))
    for op_type, kernel_id, function in (
        ("FusedConvBatchNormRelu", "conv_bn_relu.fp32.nchw.direct", fused_conv_batch_norm_relu),
        ("FusedConvBatchNorm", "conv_bn.fp32.nchw.direct", fused_conv_batch_norm),
    ):
        arguments = tensor_args + (arg("output", "*fp32", "output", index=0),) + conv_dimensions + (
            arg("epsilon", "fp32", "attribute_f64", name="batchnorm_epsilon", default=1e-5),
        ) + reserved()
        register(KernelSpec(
            kernel_id, op_type, 1, function,
            {item.name: item.type for item in arguments if item.source["kind"] != "runtime_reserved"},
            arguments, grid(), variants(), layout="NCHW",
        ))
    conv_arguments = (
        arg("x", "*fp32", "input", index=0),
        arg("weight", "*fp32", "input", index=1),
        arg("output", "*fp32", "output", index=0),
    ) + conv_dimensions + reserved()
    register(KernelSpec(
        "conv.fp32.nchw.direct", "Conv", 1, conv,
        {item.name: item.type for item in conv_arguments if item.source["kind"] != "runtime_reserved"},
        conv_arguments, grid(), variants(), layout="NCHW",
    ))


register_conv_family()


def simple_spec(kernel_id: str, op_type: str, function: object,
                inputs: tuple[str, ...]) -> None:
    arguments = tuple(
        arg(name, "*fp32", "input", index=index) for index, name in enumerate(inputs)
    ) + (
        arg("output", "*fp32", "output", index=0),
        arg("total", "i32", "output_numel", index=0),
    ) + reserved()
    register(KernelSpec(
        kernel_id, op_type, 1, function,
        {item.name: item.type for item in arguments if item.source["kind"] != "runtime_reserved"},
        arguments, grid(), variants(), layout="NCHW",
    ))


simple_spec("relu.fp32.contiguous", "Relu", relu, ("x",))
simple_spec("add.fp32.contiguous", "Add", add, ("a", "b"))
simple_spec("add_relu.fp32.contiguous", "AddRelu", add_relu, ("a", "b"))


bn_arguments = (
    arg("x", "*fp32", "input", index=0),
    arg("scale", "*fp32", "input", index=1),
    arg("bias", "*fp32", "input", index=2),
    arg("mean", "*fp32", "input", index=3),
    arg("variance", "*fp32", "input", index=4),
    arg("output", "*fp32", "output", index=0),
    arg("total", "i32", "output_numel", index=0),
    arg("channels", "i32", "input_dim", index=0, axis=1),
    arg("spatial", "i32", "input_spatial", index=0, begin_axis=2),
    arg("epsilon", "fp32", "attribute_f64", name="epsilon",
        alternate_name="batchnorm_epsilon", default=1e-5),
) + reserved()

# input_spatial is normalized into a product expression before pack emission.
bn_arguments = tuple(
    ArgumentSpec(item.name, item.type, {"kind": "input_dim_product", **{k: v for k, v in item.source.items() if k != "kind"}})
    if item.source["kind"] == "input_spatial" else item
    for item in bn_arguments
)

# The SDK currently accepts this additional safe source used by BatchNorm.
register(KernelSpec(
    "batch_norm.fp32.nchw", "BatchNormalization", 1, batch_norm,
    {item.name: item.type for item in bn_arguments if item.source["kind"] != "runtime_reserved"},
    bn_arguments, grid(), variants(), layout="NCHW",
))


pool_arguments = (
    arg("x", "*fp32", "input", index=0),
    arg("output", "*fp32", "output", index=0),
    arg("n", "i32", "input_dim", index=0, axis=0),
    arg("c", "i32", "input_dim", index=0, axis=1),
    arg("h", "i32", "input_dim", index=0, axis=2),
    arg("w", "i32", "input_dim", index=0, axis=3),
    arg("out_h", "i32", "output_dim", index=0, axis=2),
    arg("out_w", "i32", "output_dim", index=0, axis=3),
    arg("kernel_h", "i32", "attribute_i64", name="kernel_shape", index=0, default=1),
    arg("kernel_w", "i32", "attribute_i64", name="kernel_shape", index=1, default=1),
    arg("stride_h", "i32", "attribute_i64", name="strides", index=0, default=1),
    arg("stride_w", "i32", "attribute_i64", name="strides", index=1, default=1),
    arg("pad_h", "i32", "attribute_i64", name="pads", index=0, default=0),
    arg("pad_w", "i32", "attribute_i64", name="pads", index=1, default=0),
    arg("dilation_h", "i32", "attribute_i64", name="dilations", index=0, default=1),
    arg("dilation_w", "i32", "attribute_i64", name="dilations", index=1, default=1),
) + reserved()
register(KernelSpec(
    "max_pool.fp32.nchw", "MaxPool", 1, max_pool,
    {item.name: item.type for item in pool_arguments if item.source["kind"] != "runtime_reserved"},
    pool_arguments, grid(), variants(), layout="NCHW",
))


gap_arguments = (
    arg("x", "*fp32", "input", index=0),
    arg("output", "*fp32", "output", index=0),
    arg("n", "i32", "input_dim", index=0, axis=0),
    arg("c", "i32", "input_dim", index=0, axis=1),
    arg("h", "i32", "input_dim", index=0, axis=2),
    arg("w", "i32", "input_dim", index=0, axis=3),
) + reserved()
register(KernelSpec(
    "global_average_pool.fp32.nchw", "GlobalAveragePool", 1,
    global_average_pool,
    {item.name: item.type for item in gap_arguments if item.source["kind"] != "runtime_reserved"},
    gap_arguments, grid(), variants(), layout="NCHW",
))


gemm_arguments = (
    arg("a", "*fp32", "input", index=0),
    arg("b", "*fp32", "input", index=1),
    arg("output", "*fp32", "output", index=0),
    arg("m", "i32", "input_dim", index=0, axis=0),
    arg("n", "i32", "input_dim", index=1, axis=1),
    arg("k", "i32", "input_dim", index=0, axis=1),
) + reserved()
register(KernelSpec(
    "gemm.fp32.direct", "Gemm", 1, gemm,
    {item.name: item.type for item in gemm_arguments if item.source["kind"] != "runtime_reserved"},
    gemm_arguments, grid(), variants(), layout="",
    constraints=(
        {"kind": "input_count_eq", "value": 2},
        {"kind": "attribute_i64_eq", "name": "transB", "value": 0, "default": 0},
        {"kind": "tensor_rank_eq", "source": "input", "index": 0, "value": 2},
        {"kind": "tensor_rank_eq", "source": "input", "index": 1, "value": 2},
    ),
))
