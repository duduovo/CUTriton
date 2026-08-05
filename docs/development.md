# CUTriton 开发指南

## 开发原则

- IR 只描述语义，不依赖 Triton/CUDA。
- Python/Triton 只生成 AOT 产物，生产 Runtime 不执行 Python。
- Compiler 承担 shape、融合候选、Lowering 和内存规划；Runtime 不重新决策 op。
- 所有 CUDA 计算使用声明式参数绑定和统一 launcher。
- 没有真实实现必须返回错误，不使用 NoOp 伪装成功。
- 新的 source/grid/constraint 语法必须是受限 AST，并同时增加 parser 和负向测试。

## 修改落点

### 新增已有 op 的 Kernel 变体

1. 在 `python/cutriton/triton_kernels/` 实现 Kernel。
2. 注册 `KernelSpec`：稳定 kernel ID、参数 ABI、grid、constraints 和 variants。
3. 运行 pack builder 和 SDK 测试。
4. 增加 C++ ArtifactRepository round-trip 与 GPU correctness 测试。

不要修改 `ExecutionContext` 或 `cuda_launcher.cpp` 的 op 分派；核心 launcher 没有这种
分派。只有引入新的参数来源类型时才扩展通用 binder。

### 新增 IR op

1. 在 `OpSchemaRegistry` 注册输入输出契约和 shape 推导。
2. 如需图规范化或模式识别，增加小而可组合的 Graph Pass。
3. 如有融合等价方案，在 `FusionRegistry` 注册。
4. 增加 Python KernelSpec 和 reference correctness。

### 新增融合方案

FusionRegistry 只声明语义等价候选，不直接决定最快方案。候选必须描述完整步骤和临时
Tensor；AutoTuner 对整段序列计时。若候选结果与确定性基线不一致，它不能进入选择。

### 修改动态 shape

shape 函数必须只依赖 Graph TensorDesc 与 Node Attribute，并能在 profile min/opt/max
和 Runtime 实际 shape 上重复调用。MemoryPlan 必须以 profile max 为上界；同一
profile 内不得因普通 shape 切换重新分配 Workspace。

### 修改 Runtime

Runtime 只允许解释 `KernelInvocation`/`ViewInvocation`。Graph cache key、调优 key 或
绑定 ABI 有变化时，必须检查缓存失效维度是否完整，并增加跨 Context/跨运行测试。

## 构建与测试

所有持久构建和缓存放在 `G:\Ubuntu_` 对应路径：

```bash
source /root/.venvs/cutriton/bin/activate
cd /mnt/g/CUTriton/CUTriton

cmake --build /mnt/g/Ubuntu_/CUTriton/build-wsl-cuda --parallel
ctest --test-dir /mnt/g/Ubuntu_/CUTriton/build-wsl-cuda --output-on-failure
python -m pytest tests/python -q

cmake -S . -B /mnt/g/Ubuntu_/CUTriton/build-wsl-cpu -G Ninja \
  -DCUTRITON_ENABLE_CUDA=OFF -DCUTRITON_BUILD_BENCHMARKS=OFF
cmake --build /mnt/g/Ubuntu_/CUTriton/build-wsl-cpu --parallel
ctest --test-dir /mnt/g/Ubuntu_/CUTriton/build-wsl-cpu --output-on-failure
```

性能报告必须给出 GPU、Driver、CUDA、Triton、shape、dtype、预热/测量次数和正确性
误差。性能测试只报告结果，不以快于 cuDNN 作为断言。

## 代码地图

| 文件/目录 | 职责 |
| --- | --- |
| `include/cutriton/ir/op_schema.h` | op 合法性与 shape 函数注册表 |
| `include/cutriton/ir/fusion.h` | 语义等价融合候选注册表 |
| `include/cutriton/backend/kernel_artifact.h` | pack v2、约束、Repository/Catalog |
| `include/cutriton/backend/kernel_invocation.h` | PlanStep 和 ExecutionCandidate |
| `src/backend/kernel_artifact.cpp` | JSON、schema、SHA 与 catalog 实现 |
| `src/backend/backend.cpp` | 编译期能力检查和 Lowering |
| `src/backend/cuda_launcher.cpp` | Module Cache、ArgumentBinder、通用 launch |
| `src/compiler/compiler.cpp` | Pass、profile、Lowering、MemoryPlan 汇合点 |
| `src/runtime/engine.cpp` | Context、调优、Graph LRU、stream/event |
| `python/cutriton/kernel_sdk/` | KernelSpec 与 pack builder |
| `python/cutriton/triton_kernels/` | AOT Triton 实现 |

## 尚未完成

- Conv/Gemm 高性能 blocking、Tensor Core、FP16/TF32。
- profile min/opt/max 离线批量调优命令行工具。
- 原生 ONNX importer、C++ CPU reference 和 pybind11。
- `cudaMallocAsync` 内存池、多 GPU/跨设备图、稳定插件 ABI。
