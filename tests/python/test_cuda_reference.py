"""Compare the native CUDA/Triton ResNet stem result with PyTorch CPU."""

from __future__ import annotations

import argparse
import subprocess
import tempfile
from pathlib import Path

import torch
import torch.nn.functional as functional


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--executable", type=Path, required=True)
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as directory:
        output_path = Path(directory) / "output.bin"
        subprocess.run(
            [str(args.executable), "--dump", str(output_path)], check=True
        )
        actual = torch.from_file(
            str(output_path), shared=False, size=1000, dtype=torch.float32
        ).clone()

    input_tensor = (
        (torch.arange(3 * 224 * 224, dtype=torch.int64) % 29 - 14)
        .to(torch.float32)
        .div(29.0)
        .reshape(1, 3, 224, 224)
    )
    conv_weight = torch.zeros((64, 3, 7, 7), dtype=torch.float32)
    for output_channel in range(64):
        conv_weight[output_channel, output_channel % 3, 3, 3] = 0.05 * (
            1 + output_channel % 5
        )
    fc_weight = (
        (torch.arange(64 * 1000, dtype=torch.int64) % 11 - 5)
        .to(torch.float32)
        .div(101.0)
        .reshape(64, 1000)
    )
    value = functional.conv2d(input_tensor, conv_weight, stride=2, padding=3)
    value = functional.batch_norm(
        value,
        running_mean=torch.zeros(64),
        running_var=torch.full((64,), 0.99999),
        weight=torch.ones(64),
        bias=torch.zeros(64),
        training=False,
        eps=1e-5,
    )
    value = functional.relu(value)
    value = functional.adaptive_avg_pool2d(value, (1, 1)).flatten(1)
    expected = (value @ fc_weight).flatten()
    torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)


if __name__ == "__main__":
    main()
