#!/usr/bin/env python3
"""Validate and benchmark CUTriton's full FP32 ResNet-50 against PyTorch."""

from __future__ import annotations

import argparse
import math
import re
import subprocess
import tempfile
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as functional


class WeightFactory:
    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.seed = 1

    def make(self, shape: tuple[int, ...], fan_in: int) -> torch.Tensor:
        count = math.prod(shape)
        indices = torch.arange(count, dtype=torch.int64, device=self.device)
        integers = (indices * 17 + self.seed * 13).remainder(19) - 9
        scale = torch.tensor(
            0.02 / math.sqrt(float(fan_in)),
            dtype=torch.float32,
            device=self.device,
        )
        self.seed += 1
        return (integers.to(torch.float32) * scale).reshape(shape)


class ConvBn(nn.Module):
    def __init__(
        self,
        factory: WeightFactory,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
        relu: bool,
    ) -> None:
        super().__init__()
        self.stride = stride
        self.padding = padding
        self.relu = relu
        self.register_buffer(
            "weight",
            factory.make(
                (out_channels, in_channels, kernel_size, kernel_size),
                in_channels * kernel_size * kernel_size,
            ),
        )
        self.register_buffer(
            "scale", torch.ones(out_channels, device=factory.device)
        )
        self.register_buffer(
            "bias", torch.full((out_channels,), 0.01, device=factory.device)
        )
        self.register_buffer(
            "mean", torch.zeros(out_channels, device=factory.device)
        )
        self.register_buffer(
            "variance",
            torch.full((out_channels,), 0.99999, device=factory.device),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value = functional.conv2d(
            value,
            self.weight,
            bias=None,
            stride=self.stride,
            padding=self.padding,
        )
        value = functional.batch_norm(
            value,
            self.mean,
            self.variance,
            self.scale,
            self.bias,
            training=False,
            eps=1e-5,
        )
        return functional.relu(value) if self.relu else value


class Bottleneck(nn.Module):
    def __init__(
        self,
        factory: WeightFactory,
        in_channels: int,
        planes: int,
        stride: int,
    ) -> None:
        super().__init__()
        self.conv1 = ConvBn(factory, in_channels, planes, 1, 1, 0, True)
        self.conv2 = ConvBn(factory, planes, planes, 3, stride, 1, True)
        self.conv3 = ConvBn(factory, planes, planes * 4, 1, 1, 0, False)
        self.downsample = (
            ConvBn(factory, in_channels, planes * 4, 1, stride, 0, False)
            if stride != 1 or in_channels != planes * 4
            else None
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        identity = value if self.downsample is None else self.downsample(value)
        result = self.conv1(value)
        result = self.conv2(result)
        result = self.conv3(result)
        return functional.relu(result + identity)


class ResNet50(nn.Module):
    def __init__(self, device: torch.device) -> None:
        super().__init__()
        factory = WeightFactory(device)
        self.stem = ConvBn(factory, 3, 64, 7, 2, 3, True)
        blocks: list[nn.Module] = []
        channels = 64
        for stage, (count, planes) in enumerate(
            zip((3, 4, 6, 3), (64, 128, 256, 512))
        ):
            for block in range(count):
                stride = 2 if stage != 0 and block == 0 else 1
                blocks.append(Bottleneck(factory, channels, planes, stride))
                channels = planes * 4
        self.blocks = nn.ModuleList(blocks)
        self.register_buffer("fc_weight", factory.make((2048, 1000), 2048))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value = self.stem(value)
        value = functional.max_pool2d(value, kernel_size=3, stride=2, padding=1)
        for block in self.blocks:
            value = block(value)
        value = functional.adaptive_avg_pool2d(value, (1, 1)).flatten(1)
        return value @ self.fc_weight


def make_input(device: torch.device) -> torch.Tensor:
    indices = torch.arange(3 * 224 * 224, dtype=torch.int64, device=device)
    values = ((indices.remainder(29) - 14).to(torch.float32) / 29.0).reshape(
        1, 3, 224, 224
    )
    return values


def benchmark_pytorch(
    model: nn.Module, value: torch.Tensor, warmup: int, iterations: int
) -> tuple[torch.Tensor, float]:
    with torch.inference_mode():
        for _ in range(warmup):
            output = model(value)
        torch.cuda.synchronize()
        elapsed = 0.0
        for _ in range(iterations):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            output = model(value)
            end.record()
            end.synchronize()
            elapsed += start.elapsed_time(end)
    return output, elapsed / iterations


def run_cutriton(
    executable: Path, output_path: Path, warmup: int, iterations: int
) -> tuple[float, float, str]:
    command = [
        str(executable),
        "--output",
        str(output_path),
        "--warmup",
        str(warmup),
        "--iterations",
        str(iterations),
        "--cuda-graph",
        "1",
    ]
    process = subprocess.run(command, check=True, text=True, capture_output=True)
    match = re.search(
        r"compile_ms=([0-9.eE+-]+) latency_ms=([0-9.eE+-]+)", process.stdout
    )
    if match is None:
        raise RuntimeError(f"Cannot parse CUTriton output:\n{process.stdout}")
    return float(match.group(1)), float(match.group(2)), process.stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument("--rtol", type=float, default=1e-4)
    args = parser.parse_args()
    if args.warmup < 1 or args.iterations < 1:
        parser.error("warmup and iterations must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required")

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda:0")
    model = ResNet50(device).eval()
    value = make_input(device)
    reference, pytorch_ms = benchmark_pytorch(
        model, value, args.warmup, args.iterations
    )

    with tempfile.TemporaryDirectory(prefix="cutriton-resnet50-") as directory:
        output_path = Path(directory) / "output.bin"
        compile_ms, cutriton_ms, native_output = run_cutriton(
            args.executable.resolve(), output_path, args.warmup, args.iterations
        )
        actual = torch.from_file(
            str(output_path), shared=False, size=1000, dtype=torch.float32
        ).clone()

    expected = reference.detach().cpu().flatten()
    difference = (actual - expected).abs()
    max_absolute_error = difference.max().item()
    mean_absolute_error = difference.mean().item()
    max_relative_error = (
        difference / expected.abs().clamp_min(1e-7)
    ).max().item()
    matches = torch.allclose(actual, expected, atol=args.atol, rtol=args.rtol)

    print(native_output)
    print(
        "RESNET50_CORRECTNESS"
        f" allclose={str(matches).lower()}"
        f" atol={args.atol:g} rtol={args.rtol:g}"
        f" max_abs={max_absolute_error:.8g}"
        f" mean_abs={mean_absolute_error:.8g}"
        f" max_rel={max_relative_error:.8g}"
    )
    print("\nFP32 batch=1, NCHW 1x3x224x224, CUDA event latency")
    print(f"{'implementation':<28} {'latency (ms)':>14} {'relative':>12}")
    print(f"{'PyTorch eager / cuDNN':<28} {pytorch_ms:>14.4f} {'1.00x':>12}")
    print(
        f"{'CUTriton / CUDA Graph':<28} {cutriton_ms:>14.4f}"
        f" {cutriton_ms / pytorch_ms:>11.2f}x"
    )
    print(f"CUTriton compile + engine setup: {compile_ms:.2f} ms")
    if not matches:
        raise SystemExit("CUTriton output does not match the PyTorch reference")


if __name__ == "__main__":
    main()
