# CUTriton

一个以 C++17 为核心、使用 Triton 生成 CUDA Kernel 的静态图推理引擎实验项目。

CUTriton 把模型语义、图优化、后端能力检查、内存规划与运行时执行分成清晰的模块。
当前已经能在单张 NVIDIA GPU 上真实执行完整 FP32 ResNet-50（batch=1、NCHW、
224×224、1000 类输出），而不是用 NoOp Kernel 模拟成功。

> 项目状态：`0.1 / V1`。核心链路已经跑通并有真实 GPU 集成测试，但它仍是学习、研究
> 和架构验证阶段的推理引擎，不是可直接替代 TensorRT 或 ONNX Runtime 的生产部署套件。

## 项目做了什么

CUTriton 当前可以把一个带常量权重的 C++ `Model` 编译成 `ExecutablePlan`，然后由
`Engine` 在 CUDA 设备上执行：

```text
Model + Host constants
        |
        v
Compiler
  ├─ 复制 Graph，不修改原始 Model
  ├─ 拓扑排序与形状推导
  ├─ DCE 与算子规范化
  ├─ Conv + BatchNorm(+ReLU)、Add + ReLU 融合
  ├─ Backend::CheckSupport
  └─ Workspace / alias 内存规划
        |
        v
ExecutablePlan
  ├─ 优化后的 Graph
  ├─ 拓扑有序的 PlanOp
  ├─ 常量 Tensor
  ├─ 目标设备与 Kernel 产物目录
  └─ MemoryPlan / CUDA Graph / profiling 配置
        |
        v
EngineState（Plan 与设备常量共享）
        |
        v
ExecutionContext（每个并发任务独占）
  ├─ 输入输出绑定与完整校验
  ├─ CUDA Stream
  ├─ Workspace 与中间 Tensor view
  ├─ Triton PTX Kernel
  ├─ CUDA Graph capture / replay
  └─ CUDA Event profiling
```

完整 ResNet-50 基准执行如下链路：

```text
Input [1, 3, 224, 224]
  -> Conv7x7 + BatchNorm + ReLU
  -> MaxPool
  -> layer1: Bottleneck x3
  -> layer2: Bottleneck x4
  -> layer3: Bottleneck x6
  -> layer4: Bottleneck x3
  -> GlobalAveragePool
  -> Flatten（零拷贝 view）
  -> Gemm
  -> Output [1, 1000]
```

每个 Bottleneck 都包含 1×1、3×3、1×1 三个卷积和残差 Add+ReLU；需要改变
空间尺寸或通道数时还包含 1×1 投影支路。输出会与同权重的 PyTorch FP32 参考实现按
`atol=1e-4`、`rtol=1e-4` 比较。

## 设计重点

- **编译期与运行期分离**：Pass、后端选择和内存规划在编译期完成，运行时只执行
  稳定的 Plan。
- **IR 是中心契约**：Importer、Pass、Backend、MemoryPlanner 和 Engine 围绕同一套
  `Graph`、`Node`、`ValueDesc` 与 `TensorDesc` 工作。
- **没有虚假成功**：生产后端不存在 NoOp Kernel；缺少算子实现、Kernel 产物或设备
  能力时返回带原因的 `Status`。
- **真实常量生命周期**：Model 保存 Host 常量，Plan 携带常量，EngineState 一次性上传
  并在多个 Context 之间共享设备常量。
- **显式内存规划**：中间 Tensor 使用单块 Workspace、256 字节对齐和 best-fit 复用；
  Flatten 使用 alias，不产生额外拷贝。
- **Context 级资源隔离**：每个 Context 独占 Stream、Workspace、Kernel、CUDA Graph
  和 Event；多个 Context 可以并行，单个 Context 非线程安全。
- **离线 Triton 产物**：构建阶段生成 PTX 和版本化 manifest，C++ 运行时不依赖 Python。

## 当前能力

| 模块 | 状态 | 当前范围 |
| --- | --- | --- |
| C++ IR | 完成首版 | Model、Graph、Node、Value、Attribute、Host 常量 |
| Graph Pass | 完成首版 | 拓扑排序、形状推导、DCE、融合、规范化、静态校验 |
| Backend 系统 | 完成首版 | 详细能力检查、Kernel 工厂、线程安全注册表 |
| CUDA Buffer | 可用 | Driver API 显存分配、同步 H2D/D2H、外部 Buffer 包装 |
| MemoryPlanner | 可用 | 静态 shape、256 B 对齐、best-fit、Flatten alias |
| CUDA Runtime | 可用 | 内部/外部 Stream、异步提交、同步、Graph、Event |
| Triton Kernel | 可用 | FP32 FusedConvBN、FusedConvBNReLU、MaxPool、AddRelu、GAP、Gemm；Flatten 为 view |
| GPU 数值测试 | 通过 | 完整 ResNet-50 与 PyTorch FP32 对齐 |
| C++ CPU backend | 未完成 | `cpu_reference` 对真实计算明确返回 `Unsupported` |
| Python API | 参考门面 | JSON/ONNX 结构读取和少量 Python 参考算子 |
| 原生 ONNX importer | 未完成 | 尚未把 ONNX initializer/attribute 完整 lowering 到 C++ IR |
| Python/C++ 绑定 | 未完成 | 尚未使用 pybind11 暴露原生 C++ Engine |

## 当前约束

首个里程碑刻意限制范围，以保证执行语义和错误边界清晰：

- 静态 shape。
- FP32。
- 图像算子采用 NCHW，卷积权重采用 OIHW。
- 单张 NVIDIA GPU，默认 `device_id=0`。
- Kernel 产物最低 Compute Capability 为 8.0；当前已在 RTX 4060（SM 8.9）验证。
- CUDA 图中不允许 CPU 节点；跨设备 fallback 尚未实现。
- GPU 输入输出必须已经位于同一目标 CUDA 设备，不自动插入 H2D/D2H 节点。
- 没有动态 shape、FP16/TF32、自动调优、显存池和 `cudaMallocAsync`。

## ResNet-50 性能基准

基准使用确定性合成权重，两个实现共享完全相同的输入、权重和 BatchNorm 参数。
计时只包含 GPU 推理，使用 CUDA Event；CUTriton 测量 CUDA Graph replay，PyTorch
测量 eager/cuDNN。当前机器在 5 次预热、20 次测量下的结果为：

| 实现 | FP32 batch=1 延迟 | 相对 PyTorch |
| --- | ---: | ---: |
| PyTorch 2.11 eager / cuDNN | 3.2522 ms | 1.00× |
| CUTriton / CUDA Graph | 39.6088 ms | 12.18× slower |
| CUTriton（关闭 CUDA Graph） | 40.3838 ms | 12.42× slower |

1000 维输出最大绝对误差为 `6.98e-9`。CUTriton 当前明显更慢，主要原因是卷积 Kernel
仍按单个输出元素直接遍历 C×R×S，没有共享内存分块、矩阵化、Tensor Core、算子形状
特化或自动调优；约 8 秒的首次 Engine 建立时间还包括为每个节点加载/创建 Kernel，
也尚未实现模块缓存。这份结果是优化基线，不代表 Triton 的性能上限。

复现实测：

```bash
cmake --build /mnt/g/Ubuntu_/CUTriton/build-wsl-cuda --parallel \
  --target cutriton_resnet50_benchmark
python benchmarks/resnet50_compare.py \
  --executable /mnt/g/Ubuntu_/CUTriton/build-wsl-cuda/cutriton_resnet50_benchmark \
  --warmup 5 --iterations 20
```

## 已验证环境

以下组合已在本仓库实际构建并通过测试：

| 组件 | 版本 |
| --- | --- |
| Windows | Windows 10 22H2 + WSL2 |
| Linux | Ubuntu 24.04 LTS |
| GPU | NVIDIA GeForce RTX 4060，Compute Capability 8.9 |
| Windows NVIDIA Driver | 610.74 |
| CUDA Toolkit | 13.0.88 |
| Python | 3.12.3 |
| PyTorch | 2.11.0+cu130 |
| Triton | 3.6.0 |
| CMake / Compiler | CMake 3.28，GCC 13.3，Ninja 1.11 |

CUDA/Triton 主线以 WSL2/Linux 为准；原生 Windows 保留 CPU-only 构建与架构测试。

## 快速开始

### 1. Windows CPU-only 构建

需要 CMake 和支持 C++17 的 MSVC 或 MinGW：

```powershell
cmake -S . -B build `
  -DCUTRITON_BUILD_TESTS=ON `
  -DCUTRITON_ENABLE_CUDA=OFF
cmake --build build
ctest --test-dir build --output-on-failure
```

CPU-only 测试使用显式 MockBackend 验证编译、内存规划、绑定、生命周期和注册表，
不把未实现的 `cpu_reference` 当作数值成功。

### 2. Windows + WSL2 CUDA 自动安装

项目提供两阶段安装脚本。第一步需要管理员 PowerShell：

```powershell
powershell -ExecutionPolicy Bypass -File .\tools\install_wsl.ps1 `
  -StorageRoot G:\Ubuntu_
```

重启 Windows 并首次打开 Ubuntu 后，在普通 PowerShell 中执行：

```powershell
powershell -ExecutionPolicy Bypass -File .\tools\run_wsl_setup.ps1
```

脚本会完成以下工作：

1. 安装 Ubuntu 构建工具、CMake 和 Ninja。
2. 验证 WSL 可以访问 Windows NVIDIA 驱动。
3. 只安装 `cuda-toolkit-13-0`，不安装 Linux NVIDIA 驱动。
4. 创建 `$HOME/.venvs/cutriton`。
5. 安装 PyTorch 2.11、Triton 3.6 和开发依赖。
6. 离线生成 PTX/manifest，构建 C++ CUDA 目标并运行测试。

默认存储位置：

```text
G:\Ubuntu_\Distro\ext4.vhdx                 WSL Linux 文件系统
G:\Ubuntu_\wsl-swap.vhdx                    WSL swap
G:\Ubuntu_\CUTriton\build-wsl-cuda         CUDA 构建与 Kernel 产物
```

可以通过 `CUTRITON_STORAGE_ROOT` 或 `CUTRITON_BUILD_DIR` 覆盖 Linux 构建位置。

> WSL 直接使用 Windows 主机提供的 NVIDIA 驱动。不要在 WSL 中安装 `cuda`、
> `cuda-drivers` 或 Linux 显卡驱动包，只安装 `cuda-toolkit-*`。

### 3. 已配置 WSL 环境中的日常开发

先从 Windows 终端进入 Ubuntu：

```powershell
wsl -d Ubuntu
```

然后在 Ubuntu 中执行：

```bash
source "$HOME/.venvs/cutriton/bin/activate"
cd /mnt/g/CUTriton/CUTriton

cmake --build /mnt/g/Ubuntu_/CUTriton/build-wsl-cuda --parallel
ctest --test-dir /mnt/g/Ubuntu_/CUTriton/build-wsl-cuda \
  --output-on-failure
```

如需在其他 Linux 环境手动配置：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install "torch==2.11.0" \
  --index-url https://download.pytorch.org/whl/cu130
python -m pip install -e ".[dev,triton]"

cmake -S . -B build-cuda -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCUTRITON_BUILD_TESTS=ON \
  -DCUTRITON_ENABLE_CUDA=ON \
  -DPython3_EXECUTABLE="$PWD/.venv/bin/python"
cmake --build build-cuda --parallel
ctest --test-dir build-cuda --output-on-failure
```

## C++ 使用流程

公共聚合头为：

```cpp
#include <cutriton/cutriton.h>
```

典型生命周期如下：

```cpp
cutriton::Model model;
// 1. 向 model.graph() 添加输入、节点和输出。
// 2. 用 model.AddConstant() 添加静态 Host 权重。

cutriton::CompileOptions options;
options.target = "cuda_triton";
options.device_id = 0;
options.kernel_artifact_dir = "/path/to/build/triton_kernels";
options.enable_cuda_graph = true;
options.enable_profiling = true;

std::unique_ptr<cutriton::Engine> engine;
cutriton::Status status = cutriton::BuildEngine(model, options, &engine);
if (!status.ok()) {
  std::cerr << status.ToString() << '\n';
  return 1;
}

auto context = engine->CreateExecutionContext();
// 3. 分配同一 device_id 上的 CUDA Buffer，并 BindInput/BindOutput。
// 4. context->Run()，或 RunAsync(stream) 后调用 Synchronize()。
// 5. Synchronize() 后读取 context->profiler().events()。
```

完整的可执行构图、权重上传、输入输出绑定、CUDA Graph replay、重新绑定和外部 Stream
示例见 [`tests/cpp/test_cuda.cpp`](tests/cpp/test_cuda.cpp)。

## Python 参考门面

Python 包目前用于 API 形态验证、JSON/ONNX 结构读取和参考算子，不会调用原生 C++
CUDA Engine。不要用这条路径评估 CUTriton GPU 性能。

```bash
python -m pip install -e ".[dev]"
```

```python
import cutriton

engine = cutriton.compile("demo/gelu_graph.json", target="cpu_reference")
context = engine.create_context()
result = context.run({"x": [-1.0, 0.0, 1.0]})
print(result["y"])
```

当前 Python 参考执行覆盖 `Identity`、`Relu`、`Gelu`、`Softmax`、
`LayerNormalization` 和 `Add`。ONNX 路径目前只读取图结构，不是完整 lowering。

运行 Python 测试与轻量 benchmark：

```bash
python -m pytest tests/python -q
python benchmarks/plan_latency.py \
  --model demo/gelu_graph.json \
  --target cpu_reference
```

该 benchmark 测量 Python 参考路径，不代表 C++/CUDA/Triton 延迟。

## Triton Kernel 产物

开启 `CUTRITON_ENABLE_CUDA` 后，CMake 会调用
[`tools/build_triton_kernels.py`](tools/build_triton_kernels.py)，生成：

```text
triton_kernels/
  ├─ FusedConvBatchNormRelu.ptx
  ├─ FusedConvBatchNorm.ptx
  ├─ MaxPool.ptx
  ├─ AddRelu.ptx
  ├─ GlobalAveragePool.ptx
  ├─ Gemm.ptx
  └─ manifest.json
```

manifest 固定记录 schema、Triton 版本、op type、符号、完整参数 ABI、dtype/layout、
最低 Compute Capability、warp 数、共享内存和 PTX SHA-256。运行时在创建 Kernel
前检查 schema/Triton 版本、op、符号、PTX 文件与哈希、dtype/layout 和设备能力，
并使用 manifest 中的启动配置，通过 `cuModuleLoadDataEx`、`cuModuleGetFunction` 和
`cuLaunchKernel` 执行 PTX。

Triton 3.6 会在用户参数后追加两个运行时指针参数；生成器和 C++ launcher 都显式
维护这段 ABI，避免 PTX 参数表与 Driver launch 参数不一致。

## 测试覆盖

### CPU-only 架构测试

- Tensor/Buffer 和常量描述校验。
- Graph Pass 与 BatchNorm epsilon 保留。
- 256 字节对齐、best-fit 复用和 Flatten alias。
- KernelRegistry 并发注册与重复注册。
- EngineState/Context 生命周期。
- 输入输出名称、描述、设备和 Buffer 边界。
- 单个 Context 的 pending run 限制。
- profiling 开关行为。

### WSL2 GPU 集成测试

- 缺失 Triton 产物和 SM 不兼容时编译失败。
- 常量权重上传和 CUDA Workspace 实际接线。
- 六种真实 Triton Kernel 与 Flatten view。
- CUDA Graph capture/replay 和重新绑定后重新捕获。
- 内部 Stream 与外部 `CUstream`。
- CUDA Event profiling。
- ResNet Stem 集成测试与 PyTorch FP32 对齐。
- 完整 ResNet-50 的独立 C++/PyTorch 正确性和性能基准。

当前本机测试结果：

```text
cutriton_core_tests                 Passed
cutriton_cuda_tests                 Passed
cutriton_cuda_pytorch_reference     Passed
cutriton_resnet50_pytorch_reference Passed
tests/python                        1 passed
```

## 仓库导航

| 路径 | 作用 |
| --- | --- |
| [`include/cutriton/`](include/cutriton/) | 公共 C++ API；每个模块都有 README |
| [`src/`](src/) | Core、IR、Backend、Compiler 和 Runtime 实现 |
| [`python/cutriton/`](python/cutriton/) | Python 参考门面和算子实验区 |
| [`tools/`](tools/) | WSL/CUDA 安装脚本和 Triton 离线生成器 |
| [`tests/cpp/`](tests/cpp/) | CPU 架构测试与真实 CUDA 集成测试 |
| [`tests/python/`](tests/python/) | Python API 与 PyTorch 数值参考测试 |
| [`demo/`](demo/) | 最小 JSON IR 示例 |
| [`benchmarks/`](benchmarks/) | 完整 ResNet-50 C++ CUDA 基准、PyTorch 对照与轻量 Python 基准 |
| [`docs/`](docs/) | 架构与开发文档 |

公共模块入口：

- [`core`](include/cutriton/core/README.md)：Status、Device、Tensor、Buffer。
- [`ir`](include/cutriton/ir/README.md)：Model、Graph、Node、Value 和 Pass。
- [`backend`](include/cutriton/backend/README.md)：Backend、OpKernel 和注册表。
- [`compiler`](include/cutriton/compiler/README.md)：CompileOptions、Compiler 和
  BuildEngine。
- [`runtime`](include/cutriton/runtime/README.md)：ExecutablePlan、MemoryPlanner、
  Engine、ExecutionContext 和 Profiler。

## 新增一个 CUDA 算子的基本步骤

1. 在 IR 中确定 op type、输入输出和 Attribute 约定。
2. 在 `src/ir/pass.cpp` 增加形状推导或规范化规则。
3. 在 Triton 生成器中实现 Kernel 并把完整 ABI 写入 manifest。
4. 在 `Backend::CheckSupport()` 中校验 dtype、shape、layout、属性和设备能力。
5. 在 CUDA Kernel launcher 中按相同 ABI 组织参数。
6. 增加 Python/PyTorch 参考结果和 C++ GPU 集成测试。
7. 更新模块 README，明确支持范围和限制。

## 路线图

下一阶段按以下顺序推进：

1. 实现原生 ONNX importer，把 initializer 和属性完整映射到 C++ IR。
2. 实现真实 C++ CPU reference Kernel，形成与 CUDA 后端独立的数值基线。
3. 使用 pybind11 让 Python `Engine` 包装原生 C++ Engine。
4. 扩展 FP16/TF32、更多通用算子和模型覆盖。
5. 加入 Kernel 自动调优、模块缓存、`cudaMallocAsync` 和显存池。
6. 扩展当前 PyTorch Eager 对照，加入 ONNX Runtime CUDA 和 TensorRT 报告。
7. 增加安装规则、CMake package export、Python wheel 和稳定的动态库导出宏。

性能数字依赖 GPU、驱动、温度和软件版本；上表保留环境与命令，不宣称固定性能比例。

## 常见问题

### 为什么不支持原生 Windows Triton？

当前 CUDA/Triton 主线以 Linux 为支持平台，因此 Windows 开发使用 WSL2；Windows
本机构建用于 CPU-only API 和架构测试。

### 为什么不能自动回退到 CPU？

CUDA 与 CPU 节点之间需要显式数据搬运、同步和新的生命周期规划。在这些语义实现前，
直接 fallback 会产生看似成功但数据错误的混合设备图，因此当前选择明确失败。

### 为什么 Python `compile()` 没有使用 C++ Engine？

pybind11 桥接尚未实现。Python 侧目前刻意保持为轻量参考门面，原生执行入口是 C++
`Compiler`、`Engine` 和 `ExecutionContext`。

### CUDA 编译提示找不到 Kernel 产物怎么办？

确保先构建 `cutriton_triton_kernels`，并把
`CompileOptions::kernel_artifact_dir` 指向包含 `manifest.json` 与 PTX 的目录。

## 许可证

CUTriton 使用 [MIT License](LICENSE)。

## 相关资料

- [Triton 官方文档](https://triton-lang.org/)
- [Microsoft WSL 安装文档](https://learn.microsoft.com/windows/wsl/install)
- [NVIDIA CUDA on WSL 指南](https://docs.nvidia.com/cuda/wsl-user-guide/)
