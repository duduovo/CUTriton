#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "usage: $0 <benchmark-executable> <output-prefix> [tokens]" >&2
  exit 2
fi

executable=$(realpath "$1")
output_prefix=$2
tokens=${3:-1024}

if [[ ! -x "$executable" ]]; then
  echo "benchmark executable is not runnable: $executable" >&2
  exit 2
fi
if ! command -v nsys >/dev/null 2>&1; then
  echo "nsys was not found; install NVIDIA Nsight Systems CLI" >&2
  exit 2
fi
if ! command -v ncu >/dev/null 2>&1; then
  echo "ncu was not found; install NVIDIA Nsight Compute CLI" >&2
  exit 2
fi
if [[ ! "$tokens" =~ ^[1-9][0-9]*$ ]]; then
  echo "tokens must be a positive integer" >&2
  exit 2
fi

mkdir -p "$(dirname "$output_prefix")"

nsys profile \
  --force-overwrite=true \
  --trace=cuda,nvtx,osrt \
  --sample=none \
  --output="${output_prefix}_nsys" \
  "$executable" --tokens "$tokens" --warmup 50 --iterations 200

nsys stats \
  --report=cuda_gpu_kern_sum,cuda_gpu_mem_time_sum \
  --format=csv \
  --output="${output_prefix}_nsys_stats" \
  "${output_prefix}_nsys.nsys-rep"

ncu \
  --force-overwrite \
  --target-processes=all \
  --replay-mode=kernel \
  --set=full \
  --output="${output_prefix}_ncu" \
  "$executable" --tokens "$tokens" --warmup 1 --iterations 1

ncu \
  --import "${output_prefix}_ncu.ncu-rep" \
  --page=details \
  --csv \
  --log-file "${output_prefix}_ncu.csv"

echo "Nsight reports written under ${output_prefix}_{nsys,ncu}.*"
