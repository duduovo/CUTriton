# benchmarks

本目录存放 CUTriton 的可复现基准与性能对照。

## Transformer FFN

- `transformer_ffn_native.cpp`：构造 BERT-tiny 尺寸的 FP16 FFN，分别运行融合和未融合
  AOT 候选，用 CUDA Event 测量并导出输出。
- `transformer_ffn_compare.py`：使用完全相同的确定性输入/权重对照 ORT CUDA 和
  PyTorch eager，检查 FP16 误差并生成 JSON 报告。

```bash
cmake --build /mnt/g/Ubuntu_/CUTriton/build-wsl-cuda --parallel \
  --target cutriton_transformer_ffn_benchmark
python benchmarks/transformer_ffn_compare.py \
  --executable /mnt/g/Ubuntu_/CUTriton/build-wsl-cuda/cutriton_transformer_ffn_benchmark \
  --tokens 1024 --warmup 50 --iterations 200 \
  --rounds 5 --json /mnt/g/Ubuntu_/CUTriton/build-wsl-cuda/transformer_ffn_report.json
```

当前 RTX 4060 五轮基线中，融合 AOT p50/p95 为 `0.033728/0.038912 ms`，未融合为
`0.038848/0.044704 ms`，ORT CUDA 子图为 `0.113728/0.143360 ms`，PyTorch eager
为 `0.030816/0.031712 ms`。因此 AOT 相对 ORT 子图达到约 `3.372x`，但仍慢于
PyTorch；融合相对未融合约为 `1.154x`。

Nsight 一键采集：

```bash
bash tools/profile_transformer_ffn.sh \
  /mnt/g/Ubuntu_/CUTriton/build-wsl-cuda/cutriton_transformer_ffn_benchmark \
  /mnt/g/Ubuntu_/CUTriton/build-wsl-cuda/nsight-transformer 1024
```

## 完整 ResNet-50

- `resnet50_native.cpp`：用公共 C++ API 构造完整 ResNet-50，运行原生 CUDA/Triton
  Engine，使用 CUDA Event 测量 CUDA Graph replay，并导出 1000 维输出。
- `resnet50_compare.py`：用相同的确定性权重构造 PyTorch FP32 参考网络，检查输出误差，
  比较 PyTorch eager/cuDNN 与 CUTriton 的 GPU 延迟。

在已经完成 WSL CUDA 构建后运行：

```bash
cmake --build /mnt/g/Ubuntu_/CUTriton/build-wsl-cuda --parallel \
  --target cutriton_resnet50_benchmark
source "$HOME/.venvs/cutriton/bin/activate"
python benchmarks/resnet50_compare.py \
  --executable /mnt/g/Ubuntu_/CUTriton/build-wsl-cuda/cutriton_resnet50_benchmark \
  --warmup 5 --iterations 20
```

正确性默认使用 `atol=1e-4`、`rtol=1e-4`。两个延迟都由 CUDA Event 测量，
不包含 H2D/D2H；CUTriton 的 compile/Engine 建立时间单独报告。

## 轻量 Python 基准

`plan_latency.py` 测的是 Python 参考门面的轻量执行路径，适合验证 benchmark 流程和
输入参数，不代表 C++/CUDA/Triton 性能。

## 与其他目录的关系

- 读取 `demo/` 中的示例模型，例如 `demo/gelu_graph.json`。
- 通过 `python/cutriton/` 暴露的 Python API 编译和运行模型。
- ResNet-50 基准直接运行 `src/` 中的 C++ Runtime、CUDA 后端和 Triton Kernel。
- benchmark 结论应反哺 `docs/`，尤其是性能报告和开发路线。

## 放什么

- 可复现的延迟、吞吐、内存占用测试。
- 对不同后端、不同输入形状、不同模型的对比脚本。
- 只放测试工具，不放核心库代码。
