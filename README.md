# CUTriton

CUTriton 是一个 C++17 静态图推理引擎实验项目：C++ 负责 IR、图优化、Lowering、
内存规划和 CUDA Runtime，Python/Triton 只负责定义并离线编译 Kernel。生产运行时不
启动 Python，只读取版本化 Kernel Pack，通过 CUDA Driver API 执行 PTX。

当前里程碑已经在 RTX 4060 上真实执行完整 FP32 ResNet-50，包括动态 shape profile、
融合与非融合候选、AOT 变体调优、CUDA Graph LRU、Module Cache 和 GPU profiling。
生产路径没有 NoOp；没有真实实现时会明确返回 `Unsupported`。

> 项目状态：`0.1`，适合学习、架构验证和 Kernel 研究，尚不能替代 TensorRT、
> ONNX Runtime 或 cuDNN。当前直接卷积 Kernel 以正确性为主，性能仍有明显差距。

## 架构

```text
C++ Model / IR（不依赖 Triton）
  -> OpSchemaRegistry：参数合法性与具体 shape 推导
  -> Graph Pass：规范化并识别融合子图
  -> FusionRegistry：生成语义等价的融合/非融合候选
  -> ArtifactRepository：解析并验证 Kernel Pack v2
  -> KernelCatalog + Backend::Lower：生成 ExecutionCandidate
  -> ExecutablePlan：保存 KernelInvocation / ViewInvocation
  -> EngineState：共享常量显存与 PTX Module/Function Cache
  -> ExecutionContext：解析 profile/shape、选择调优结果
  -> 声明式参数绑定 + 通用 cuLaunchKernel
  -> CUDA Graph LRU + CUDA Event profiling
```

这里最重要的边界是：

- IR 不认识 Triton，也不保存 Python callable。
- Kernel 的参数 ABI、约束、grid 和产物身份都来自 `pack.json`。
- Backend 只在编译期 Lower；Context 不再根据 `op_type` 决定怎样启动 Kernel。
- 所有计算 Kernel 走同一个通用 launcher；`Flatten` 是零拷贝 `ViewInvocation`。
- 新增已有 IR op 的 Kernel 变体，不需要修改 C++ launcher 或 Runtime。
- 新增 IR op 需要注册 C++ `OpSchema`，但不需要增加专用 launcher。

详细说明见 [架构文档](docs/architecture.md) 和 [开发文档](docs/development.md)。

## 已完成功能

| 模块 | 当前能力 |
| --- | --- |
| Core | `Status`、CPU/CUDA `Buffer`、Host↔Device 拷贝、Tensor view、严格边界校验 |
| IR | Model/Graph/Node/Value/Attribute、Host 常量、线程安全 `OpSchemaRegistry` |
| Pass | 拓扑排序、shape 推导、DCE、Conv-BN(-ReLU)/Add-ReLU 融合、Flatten/Gemm 规范化 |
| Kernel SDK | 模块化 `KernelSpec` 注册、安全参数来源和约束 AST、薄构建 CLI |
| Kernel Pack | schema v2、ABI schema、版本/符号/grid/约束、PTX SHA-256、多个 pack 路径 |
| Compiler | `Backend::Lower`、`KernelCatalog`、融合/非融合 `ExecutionCandidate`、profile 最大内存规划 |
| 调优 | Disabled/UseCache/TuneOnMiss/ForceRetune、独立 stream/event、中位数、原子 JSON 缓存、数值校验 |
| Runtime | 常量一次上传、共享 Module Cache、通用 launcher、内部/外部 stream、异步执行 |
| Dynamic shape | 有边界 min/opt/max profile、运行时 shape 推导、最大 Workspace、输出描述查询 |
| CUDA Graph | 按 profile/shape/candidate/地址/workspace 建键，Context 内默认 4 项 LRU |
| Profiling | 每个计算步骤的真实 CUDA Event 耗时；view 事件为 0 |
| 模型验证 | ResNet Stem 和完整 ResNet-50 均与 PyTorch FP32 对齐 |

内置 FP32 AOT 实现包括：

- `Conv`、`BatchNormalization`、`Relu`、`Add`
- `FusedConvBatchNorm`、`FusedConvBatchNormRelu`、`AddRelu`
- `MaxPool`、`GlobalAveragePool`、`Gemm`
- `Flatten` 零拷贝 view

每个计算 op 当前生成 `warps2` 和 `warps4` 两个 AOT 变体。融合节点还会得到可执行的
非融合候选，调优以完整候选序列为单位计时。

## 当前限制

- 单 NVIDIA GPU，Compute Capability 8.0+。
- FP32；图像 Tensor 为 NCHW，卷积权重为 OIHW。
- 动态 shape 必须落在明确的 min/opt/max profile 内。
- CUDA 输入输出必须已在同一 `device_id`，Runtime 不自动插入 H2D/D2H 节点。
- 不支持跨设备 CPU fallback、FP16/TF32、动态共享库插件 ABI。
- `cpu_reference` 生产后端尚未实现真实 C++ 算子，会返回 `Unsupported`。
- Python 轻量 API 尚未通过 pybind11 连接原生 C++ Engine。

## Kernel Pack v2

开启 CUDA 构建时，CMake 调用薄入口
[`tools/build_triton_kernels.py`](tools/build_triton_kernels.py)。真正的 SDK 和 Kernel
分别位于：

- [`python/cutriton/kernel_sdk/`](python/cutriton/kernel_sdk/)：`KernelSpec`、校验与 pack builder。
- [`python/cutriton/triton_kernels/`](python/cutriton/triton_kernels/)：Triton 实现模块。

产物结构：

```text
triton_kernels/
  ├─ pack.json
  ├─ kernels/
  │   └─ <kernel_id>/<variant_id>.ptx
  └─ tuning/                         # 默认调优缓存目录
      └─ <tuning_key>.json
```

`pack.json` 保存 pack/生成器/Triton/ABI 版本，Kernel 与 variant ID，有序参数 ABI，
安全参数来源，dtype/layout/rank/Attribute 约束，grid、warp、shared memory、目标 SM、
符号、PTX 路径和 SHA-256。C++ 只接受 schema v2 和支持的 ABI schema；manifest v1
会提示重新生成。它不要求 Triton 版本字符串必须恰好等于某个值，Triton 版本会进入
调优和产物身份。

## 快速开始

### WSL2 + CUDA（主线）

首次安装由管理员 PowerShell 执行：

```powershell
powershell -ExecutionPolicy Bypass -File .\tools\install_wsl.ps1 `
  -StorageRoot G:\Ubuntu_
```

重启、首次进入 Ubuntu 完成初始化后，在普通 PowerShell 执行：

```powershell
powershell -ExecutionPolicy Bypass -File .\tools\run_wsl_setup.ps1
```

项目约定所有大型持久数据位于 `G:\Ubuntu_`：

```text
G:\Ubuntu_\Distro\ext4.vhdx
G:\Ubuntu_\wsl-swap.vhdx
G:\Ubuntu_\CUTriton\build-wsl-cuda
G:\Ubuntu_\CUTriton\build-wsl-cpu
```

日常构建和测试：

```bash
source /root/.venvs/cutriton/bin/activate
cd /mnt/g/CUTriton/CUTriton
cmake --build /mnt/g/Ubuntu_/CUTriton/build-wsl-cuda --parallel
ctest --test-dir /mnt/g/Ubuntu_/CUTriton/build-wsl-cuda --output-on-failure
python -m pytest tests/python -q
```

WSL 使用 Windows 主机的 NVIDIA 驱动；不要在 WSL 安装 Linux `cuda-drivers`，只安装
CUDA Toolkit。

### CPU-only

CPU-only 构建不查找 CUDA、Triton 或 PyTorch：

```powershell
cmake -S . -B G:\Ubuntu_\CUTriton\build-cpu `
  -DCUTRITON_ENABLE_CUDA=OFF `
  -DCUTRITON_BUILD_BENCHMARKS=OFF `
  -DCUTRITON_BUILD_TESTS=ON
cmake --build G:\Ubuntu_\CUTriton\build-cpu
ctest --test-dir G:\Ubuntu_\CUTriton\build-cpu --output-on-failure
```

架构测试使用显式 `MockBackend`，不会把未实现的 CPU 计算伪装为成功。

## C++ 使用示例

```cpp
#include <cutriton/cutriton.h>

cutriton::Model model;
// 构建 model.graph()，并用 model.AddConstant() 写入静态 Host 权重。

cutriton::CompileOptions options;
options.target = "cuda_triton";
options.device_id = 0;
options.kernel_artifact_paths = {
    "/mnt/g/Ubuntu_/CUTriton/build-wsl-cuda/triton_kernels"};
options.tuning_cache_dir =
    "/mnt/g/Ubuntu_/CUTriton/tuning-cache";
options.tuning_mode = cutriton::TuningMode::kUseCache;
options.enable_cuda_graph = true;
options.cuda_graph_cache_capacity = 4;
options.enable_profiling = true;

std::unique_ptr<cutriton::Engine> engine;
cutriton::Status status = cutriton::BuildEngine(model, options, &engine);
if (!status.ok()) return 1;

auto context = engine->CreateExecutionContext();
// 绑定目标 GPU 上的 Tensor。
context->BindInput("input", input);
context->BindOutput("output", output);
context->Run();
```

动态 shape 配置：

```cpp
cutriton::ShapeProfile profile;
profile.name = "images";
profile.inputs["input"] = {
    {1, 3, 160, 160},
    {1, 3, 224, 224},
    {2, 3, 256, 256},
};
options.shape_profiles = {profile};

// 一个 profile 时 Context 自动选择；多个 profile 时先 SelectShapeProfile。
context->SetInputShape("input", {1, 3, 192, 192});
context->ResolveShapes();
const auto* output_desc = context->GetResolvedTensorDesc("output");
```

`Run()` 等价于 `RunAsync(nullptr) + Synchronize()`。`RunAsync()` 也接受外部
`CUstream`；同一 Context 只允许一个未完成任务，多个 Context 可以并行。

## 自动调优语义

| 模式 | 行为 |
| --- | --- |
| `kDisabled` | 使用 manifest 默认变体 |
| `kUseCache` | 使用精确缓存；miss 时使用默认/profile opt 选择，不现场测量 |
| `kTuneOnMiss` | 对已有 AOT 候选现场计时并写缓存，不启动 Python 编译 |
| `kForceRetune` | 忽略旧值，重新测量并原子覆盖 |

默认预热 5 次、测量 20 次并取中位数。首次选择会把各候选输出复制到 Host，与确定性
基线按 `atol=1e-4, rtol=1e-4` 比较。缓存键包含 op/融合方案、完整属性、shape、
dtype/layout、GPU UUID/SM、Driver、pack、PTX、生成器、Triton 和 ABI 身份；每个 key
一个 JSON 文件，通过临时文件和原子 rename 写入。

## ResNet-50 正确性与性能

完整链路为 stem、4 个 Bottleneck stage、GAP、Flatten view 和 Gemm。测试使用确定性
权重，与 PyTorch FP32 使用相同输入和参数；正确性阈值为 `atol=1e-4, rtol=1e-4`。

RTX 4060、FP32 batch=1、NCHW 224×224，5 次预热/20 次测量：

| 实现 | CUDA Event 延迟 | 相对 PyTorch |
| --- | ---: | ---: |
| PyTorch 2.11 eager / cuDNN | 3.2476 ms | 1.00× |
| CUTriton / CUDA Graph | 39.2066 ms | 12.07× slower |

CUTriton Engine 建立约 `99.37 ms`，1000 维输出最大绝对误差 `6.98e-9`。当前直接
卷积 Kernel 为每个输出遍历 C×R×S，尚无块化矩阵乘、Tensor Core 或充分形状特化，
因此这只是可复现的优化基线，不代表 Triton 的性能上限，也不设置“必须快于 cuDNN”
的测试断言。

复现：

```bash
python benchmarks/resnet50_compare.py \
  --executable /mnt/g/Ubuntu_/CUTriton/build-wsl-cuda/cutriton_resnet50_benchmark \
  --warmup 5 --iterations 20
```

## 测试覆盖

- KernelSpec 注册、ABI 顺序、安全 source/grid/constraint AST。
- pack 缺字段/旧 schema、PTX SHA 损坏、SM 不兼容和缺失产物。
- OpSchema、FusionRegistry、注册表并发和 Engine/Context 生命周期。
- 256 B 对齐、best-fit、Flatten alias、候选临时空间和真实 Workspace view。
- Module Cache：同一 PTX 在多节点、多 Context 中只加载一次。
- 调优数值检查、cache hit 不重复测量、产物/GPU/shape 进入缓存键。
- profile min/opt/max、越界 shape、运行时输出 shape 和最大 Workspace。
- 不同 shape/绑定地址的 CUDA Graph LRU、内部/外部 stream、异步限制。
- ResNet Stem、动态 H/W 和完整 ResNet-50 与 PyTorch 对齐。
- Linux CPU-only 与 WSL2 CUDA 构建。

## 扩展方式

### 为已有 IR op 增加 Kernel 或变体

1. 在 `python/cutriton/triton_kernels/` 定义 Triton Kernel。
2. 注册 `KernelSpec`，声明有序参数来源、grid、约束和 variants。
3. 用 `--module` 让薄 CLI 发现模块并生成 pack。
4. 加入 round-trip、错误产物和数值测试。

不需要修改 `ExecutionContext`、通用参数绑定器或 CUDA launcher。

### 增加新的 IR op

1. 在 C++ 注册 `OpSchema`，定义输入输出数量、Attribute 校验和 shape 函数。
2. 如有等价融合方案，在 `FusionRegistry` 注册候选。
3. 在 Python 注册对应 `KernelSpec`。
4. 增加编译期和 GPU correctness 测试。

仍不需要新增 C++ 专用 launcher。当前 `src/backend/backend.cpp` 中为内置融合候选构造
临时值的代码下一步会继续数据驱动化。

## 仓库导航

| 路径 | 作用 |
| --- | --- |
| [`include/cutriton/`](include/cutriton/) | 公共 C++ API |
| [`src/`](src/) | IR、Compiler、Backend、Runtime 实现 |
| [`python/cutriton/kernel_sdk/`](python/cutriton/kernel_sdk/) | AOT Kernel SDK 与 pack builder |
| [`python/cutriton/triton_kernels/`](python/cutriton/triton_kernels/) | Triton Kernel 模块 |
| [`python/cutriton/kernels/`](python/cutriton/kernels/) | Python 数值参考算子 |
| [`tools/`](tools/) | WSL 安装和薄构建 CLI |
| [`tests/`](tests/) | CPU、CUDA、SDK 与 PyTorch 对照测试 |
| [`benchmarks/`](benchmarks/) | ResNet-50 和轻量 Python benchmark |
| [`docs/`](docs/) | 架构与开发文档 |

## 下一阶段

1. 优化 Conv/Gemm：blocking、shared memory、Tensor Core、更多 shape 特化。
2. 补齐独立 `AutoTuner`/融合策略层的离线工具和 profile min/opt/max 预调优。
3. 实现原生 ONNX importer 和 C++ CPU reference backend。
4. 用 pybind11 连接 Python API 与原生 Engine。
5. 扩展 FP16/TF32、`cudaMallocAsync` 内存池和安装/发布流程。

## 许可证与资料

CUTriton 使用 [MIT License](LICENSE)。

- [Triton 官方文档](https://triton-lang.org/)
- [NVIDIA CUDA Driver API](https://docs.nvidia.com/cuda/cuda-driver-api/)
- [Microsoft WSL 安装文档](https://learn.microsoft.com/windows/wsl/install)
- [NVIDIA CUDA on WSL 指南](https://docs.nvidia.com/cuda/wsl-user-guide/)
