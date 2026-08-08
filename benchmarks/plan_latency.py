from __future__ import annotations

import argparse
from pathlib import Path
import statistics
import time

import cutriton


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="对 CUTriton Python 计划做基准测试")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--target", default="cpu_reference")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument(
        "--input-size",
        type=int,
        default=1024,
        help="演示 JSON 图使用的一维输入大小",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    engine = cutriton.compile(args.model, target=args.target)
    context = engine.create_context()
    inputs = {"x": [0.01 * i for i in range(args.input_size)]}

    for _ in range(args.warmup):
        context.run(inputs)

    samples = []
    for _ in range(args.iters):
        start = time.perf_counter()
        context.run(inputs)
        samples.append((time.perf_counter() - start) * 1000.0)

    print(f"model={args.model}")
    print(f"target={args.target}")
    print(f"iters={args.iters}")
    print(f"latency_ms_mean={statistics.mean(samples):.6f}")
    print(f"latency_ms_p50={statistics.median(samples):.6f}")
    print(f"latency_ms_min={min(samples):.6f}")


if __name__ == "__main__":
    main()
