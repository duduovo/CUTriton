#!/usr/bin/env python3
"""Validate and benchmark the AOT FP16 BERT-tiny FFN path."""

from __future__ import annotations

import argparse
import json
import platform
import re
import statistics
import subprocess
import tempfile
from pathlib import Path
from typing import Callable

import numpy as np
import onnx
import onnxruntime as ort
import torch


HIDDEN = 128
INTERMEDIATE = 512
SEED = 20260811


def deterministic_values(count: int, modulus: int, offset: int, divisor: float) -> torch.Tensor:
    indices = torch.arange(count, dtype=torch.int64)
    return ((indices.remainder(modulus) - offset).to(torch.float32) / divisor)


def make_tensors(tokens: int, device: torch.device) -> dict[str, torch.Tensor]:
    x = deterministic_values(tokens * HIDDEN, 23, 11, 37.0).reshape(tokens, HIDDEN)
    residual = deterministic_values(tokens * INTERMEDIATE, 17, 8, 53.0).reshape(
        tokens, INTERMEDIATE
    )
    weight = deterministic_values(HIDDEN * INTERMEDIATE, 19, 9, 181.0).reshape(
        HIDDEN, INTERMEDIATE
    )
    columns = torch.arange(INTERMEDIATE)
    scale = 0.9 + 0.001 * columns.remainder(13).to(torch.float32)
    bias = (columns.remainder(7).to(torch.float32) - 3.0) / 100.0
    return {
        "x": x.to(device=device, dtype=torch.float16),
        "residual": residual.to(device=device, dtype=torch.float16),
        "weight": weight.to(device=device, dtype=torch.float16),
        "scale": scale.to(device=device, dtype=torch.float16),
        "bias": bias.to(device=device, dtype=torch.float16),
    }


def pytorch_ffn(values: dict[str, torch.Tensor]) -> torch.Tensor:
    projected = torch.nn.functional.gelu(values["x"] @ values["weight"], approximate="none")
    return torch.nn.functional.layer_norm(
        projected + values["residual"],
        (INTERMEDIATE,),
        values["scale"],
        values["bias"],
        1e-5,
    )


def make_onnx_model(values: dict[str, torch.Tensor], tokens: int) -> bytes:
    helper = onnx.helper
    tensor = onnx.TensorProto
    inputs = [
        helper.make_tensor_value_info("x", tensor.FLOAT16, [tokens, HIDDEN]),
        helper.make_tensor_value_info(
            "residual", tensor.FLOAT16, [tokens, INTERMEDIATE]
        ),
    ]
    output = helper.make_tensor_value_info(
        "output", tensor.FLOAT16, [tokens, INTERMEDIATE]
    )

    def initializer(name: str) -> onnx.TensorProto:
        array = values[name].detach().cpu().numpy()
        return onnx.numpy_helper.from_array(array, name=name)

    nodes = [
        helper.make_node("MatMul", ["x", "weight"], ["projected"]),
        helper.make_node(
            "Gelu", ["projected"], ["activated"], domain="com.microsoft"
        ),
        helper.make_node("Add", ["activated", "residual"], ["sum"]),
        helper.make_node(
            "LayerNormalization",
            ["sum", "scale", "bias"],
            ["output"],
            axis=-1,
            epsilon=1e-5,
        ),
    ]
    graph = helper.make_graph(
        nodes,
        "bert_tiny_ffn",
        inputs,
        [output],
        initializer=[initializer("weight"), initializer("scale"), initializer("bias")],
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 18), helper.make_opsetid("com.microsoft", 1)],
    )
    onnx.checker.check_model(model)
    return model.SerializeToString()


class OrtCudaRunner:
    def __init__(self, model: bytes, values: dict[str, torch.Tensor], tokens: int) -> None:
        stream = torch.cuda.current_stream()
        providers = [
            (
                "CUDAExecutionProvider",
                {
                    "device_id": 0,
                    "user_compute_stream": str(int(stream.cuda_stream)),
                    "do_copy_in_default_stream": "1",
                },
            )
        ]
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        self.session = ort.InferenceSession(model, sess_options=options, providers=providers)
        self.binding = self.session.io_binding()
        for name in ("x", "residual"):
            value = values[name]
            self.binding.bind_input(
                name,
                "cuda",
                0,
                np.float16,
                tuple(value.shape),
                value.data_ptr(),
            )
        self.output = torch.empty(
            (tokens, INTERMEDIATE), dtype=torch.float16, device="cuda"
        )
        self.binding.bind_output(
            "output",
            "cuda",
            0,
            np.float16,
            tuple(self.output.shape),
            self.output.data_ptr(),
        )

    def run(self) -> torch.Tensor:
        self.session.run_with_iobinding(self.binding)
        return self.output


def percentile(samples: list[float], fraction: float) -> float:
    ordered = sorted(samples)
    index = min(len(ordered) - 1, max(0, int(np.ceil(len(ordered) * fraction)) - 1))
    return ordered[index]


def cuda_summary(
    call: Callable[[], torch.Tensor], warmup: int, iterations: int, rounds: int
) -> dict[str, object]:
    per_round = []
    with torch.inference_mode():
        for _round in range(rounds):
            for _ in range(warmup):
                call()
            torch.cuda.synchronize()
            samples = []
            for _ in range(iterations):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                call()
                end.record()
                end.synchronize()
                samples.append(start.elapsed_time(end))
            per_round.append(
                {
                    "median_ms": statistics.median(samples),
                    "p95_ms": percentile(samples, 0.95),
                }
            )
    return {
        "median_ms": statistics.median(item["median_ms"] for item in per_round),
        "p95_ms": statistics.median(item["p95_ms"] for item in per_round),
        "rounds": per_round,
    }


def native_summary(
    executable: Path,
    output: Path,
    tokens: int,
    warmup: int,
    iterations: int,
    rounds: int,
) -> dict[str, object]:
    per_round = [
        run_native(executable, output, tokens, warmup, iterations) for _ in range(rounds)
    ]
    return {
        "compile_ms": statistics.median(item["compile_ms"] for item in per_round),
        "fused_ms": statistics.median(item["fused_ms"] for item in per_round),
        "fused_p95_ms": statistics.median(item["fused_p95_ms"] for item in per_round),
        "unfused_ms": statistics.median(item["unfused_ms"] for item in per_round),
        "unfused_p95_ms": statistics.median(
            item["unfused_p95_ms"] for item in per_round
        ),
        "fusion_speedup": statistics.median(item["fusion_speedup"] for item in per_round),
        "rounds": per_round,
    }


def software_stack() -> dict[str, str]:
    try:
        gpu = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        gpu = torch.cuda.get_device_name(0)
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": str(torch.version.cuda),
        "onnx": onnx.__version__,
        "onnxruntime": ort.__version__,
        "gpu": gpu,
    }


def markdown(report: dict[str, object]) -> str:
    native = report["native"]
    pytorch = report["pytorch"]
    ort_result = report["ort"]
    return (
        "# CUTriton FP16 Transformer FFN report\n\n"
        "| Implementation | p50 (ms) | p95 (ms) |\n"
        "|:--|--:|--:|\n"
        f"| CUTriton fused | {native['fused_ms']:.6f} | "
        f"{native['fused_p95_ms']:.6f} |\n"
        f"| CUTriton unfused | {native['unfused_ms']:.6f} | "
        f"{native['unfused_p95_ms']:.6f} |\n"
        f"| ONNX Runtime CUDA | {ort_result['median_ms']:.6f} | "
        f"{ort_result['p95_ms']:.6f} |\n"
        f"| PyTorch eager | {pytorch['median_ms']:.6f} | "
        f"{pytorch['p95_ms']:.6f} |\n\n"
        f"ORT subgraph speedup: `{report['cutriton_vs_ort']:.3f}x`. "
        f"Gate: `{report['ort_subgraph_gate']}`.\n"
    )


def run_native(
    executable: Path, output: Path, tokens: int, warmup: int, iterations: int
) -> dict[str, float]:
    process = subprocess.run(
        [
            str(executable),
            "--tokens",
            str(tokens),
            "--warmup",
            str(warmup),
            "--iterations",
            str(iterations),
            "--output",
            str(output),
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    print(process.stdout.strip())
    names = (
        "compile_ms",
        "fused_ms",
        "fused_p95_ms",
        "unfused_ms",
        "unfused_p95_ms",
        "fusion_speedup",
    )
    result: dict[str, float] = {}
    for name in names:
        match = re.search(rf"{name}=([0-9.eE+-]+)", process.stdout)
        if match is None:
            raise RuntimeError(f"native benchmark omitted {name}: {process.stdout}")
        result[name] = float(match.group(1))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--tokens", type=int, default=1024)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    if min(args.tokens, args.warmup, args.iterations, args.rounds) <= 0:
        parser.error("tokens, warmup, iterations and rounds must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.manual_seed(SEED)
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device("cuda:0")
    values = make_tensors(args.tokens, device)
    reference = pytorch_ffn(values)
    ort_runner = OrtCudaRunner(make_onnx_model(values, args.tokens), values, args.tokens)
    ort_output = ort_runner.run()
    torch.cuda.synchronize()
    torch.testing.assert_close(ort_output, reference, rtol=1e-2, atol=1e-2)

    with tempfile.TemporaryDirectory(prefix="cutriton-transformer-") as directory:
        output = Path(directory) / "output.fp16"
        native = native_summary(
            args.executable.resolve(),
            output,
            args.tokens,
            args.warmup,
            args.iterations,
            args.rounds,
        )
        actual = torch.from_file(
            str(output), shared=False, size=args.tokens * INTERMEDIATE, dtype=torch.float16
        ).reshape(args.tokens, INTERMEDIATE)
    expected_cpu = reference.detach().cpu()
    difference = (actual.float() - expected_cpu.float()).abs()
    torch.testing.assert_close(actual, expected_cpu, rtol=2e-2, atol=2e-2)

    pytorch_result = cuda_summary(
        lambda: pytorch_ffn(values), args.warmup, args.iterations, args.rounds
    )
    ort_result = cuda_summary(ort_runner.run, args.warmup, args.iterations, args.rounds)
    speedup = ort_result["median_ms"] / native["fused_ms"]
    report = {
        "schema_version": 1,
        "software": software_stack(),
        "shape": {
            "tokens": args.tokens,
            "hidden": HIDDEN,
            "intermediate": INTERMEDIATE,
            "dtype": "fp16",
        },
        "methodology": {
            "seed": SEED,
            "warmup": args.warmup,
            "iterations": args.iterations,
            "rounds": args.rounds,
            "timing": "CUDA events; H2D/D2H excluded",
        },
        "native": native,
        "pytorch": pytorch_result,
        "ort": ort_result,
        "cutriton_vs_ort": speedup,
        "ort_subgraph_gate": "passed" if speedup >= 1.20 else "failed",
        "max_abs_error": float(difference.max().item()),
        "mean_abs_error": float(difference.mean().item()),
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        args.json.with_suffix(".md").write_text(markdown(report), encoding="utf-8")


if __name__ == "__main__":
    main()
