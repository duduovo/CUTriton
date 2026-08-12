#!/usr/bin/env bash
set -euo pipefail

if ! grep -qi microsoft /proc/version; then
  echo "错误：本脚本只用于 WSL2。" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPOSITORY="$(cd "$SCRIPT_DIR/.." && pwd)"
VENV_DIR="${CUTRITON_VENV:-$HOME/.venvs/cutriton}"
CUDA_PACKAGE="${CUTRITON_CUDA_PACKAGE:-cuda-toolkit-13-0}"
STORAGE_ROOT="${CUTRITON_STORAGE_ROOT:-/mnt/g/Ubuntu_/CUTriton}"
BUILD_DIR="${CUTRITON_BUILD_DIR:-$STORAGE_ROOT/build-wsl-cuda}"

mkdir -p "$STORAGE_ROOT"

echo "[1/6] 安装 Linux 构建依赖"
sudo apt-get update
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y \
  build-essential ca-certificates cmake git ninja-build pkg-config \
  python3 python3-dev python3-pip python3-venv wget

echo "[2/6] 检查 Windows 提供的 NVIDIA GPU"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
elif [[ -x /usr/lib/wsl/lib/nvidia-smi ]]; then
  /usr/lib/wsl/lib/nvidia-smi
else
  echo "错误：WSL 中找不到 nvidia-smi。请先更新 Windows NVIDIA 驱动并执行 wsl --update。" >&2
  exit 1
fi

echo "[3/6] 安装 CUDA Toolkit（不安装 Linux NVIDIA 驱动）"
TEMP_DIR="$(mktemp -d)"
trap 'rm -rf -- "$TEMP_DIR"' EXIT
wget -q \
  https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb \
  -O "$TEMP_DIR/cuda-keyring.deb"
sudo dpkg -i "$TEMP_DIR/cuda-keyring.deb"
sudo apt-get update
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y "$CUDA_PACKAGE"

CUDA_HOME="/usr/local/cuda-13.0"
if [[ ! -d "$CUDA_HOME" ]]; then
  CUDA_HOME="/usr/local/cuda"
fi
if [[ ! -x "$CUDA_HOME/bin/nvcc" ]]; then
  echo "错误：安装 $CUDA_PACKAGE 后仍找不到 nvcc。" >&2
  exit 1
fi
export CUDA_HOME
export PATH="$CUDA_HOME/bin:$PATH"

echo "[4/6] 创建 Python 虚拟环境: $VENV_DIR"
python3 -m venv "$VENV_DIR"
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip setuptools wheel
python -m pip install --upgrade "torch==2.11.0" \
  --index-url https://download.pytorch.org/whl/cu130
python -m pip install --upgrade -e "$REPOSITORY[dev,triton,benchmark]"

echo "[5/6] 验证 CUDA、PyTorch 与 Triton"
nvcc --version
python - <<'PY'
import torch
import triton
import onnx
import onnxruntime as ort

if triton.__version__ != "3.6.0":
    raise SystemExit(f"需要 Triton 3.6.0，当前为 {triton.__version__}")
if not torch.cuda.is_available():
    raise SystemExit("torch.cuda.is_available() 为 False")
if "CUDAExecutionProvider" not in ort.get_available_providers():
    raise SystemExit("ONNX Runtime 缺少 CUDAExecutionProvider")

print(f"PyTorch: {torch.__version__}")
print(f"Triton: {triton.__version__}")
print(f"ONNX: {onnx.__version__}")
print(f"ONNX Runtime: {ort.__version__}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Compute Capability: {torch.cuda.get_device_capability(0)}")
PY

echo "[6/6] 构建并测试 CUTriton CUDA 目标"
cmake -S "$REPOSITORY" -B "$BUILD_DIR" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCUTRITON_BUILD_TESTS=ON \
  -DCUTRITON_ENABLE_CUDA=ON \
  -DPython3_EXECUTABLE="$VENV_DIR/bin/python"
cmake --build "$BUILD_DIR" --parallel
ctest --test-dir "$BUILD_DIR" --output-on-failure

echo
echo "CUTriton WSL/CUDA 环境已就绪。"
echo "下次进入环境：source '$VENV_DIR/bin/activate'"
echo "CUDA 构建目录：$BUILD_DIR"
