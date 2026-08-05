# include/cutriton/compiler

本目录定义从 `Model` 到 `ExecutablePlan` 的编译入口，以及直接构建 `Engine` 的
便捷函数。

## CompileOptions

- `target`：目标后端，默认 `cuda_triton`。
- `device_id`：目标 CUDA 设备编号。
- `kernel_artifact_dir`：PTX 和版本化 `manifest.json` 所在目录。
- `enable_cuda_graph`：是否在 Context 首次执行时捕获 CUDA Graph。
- `enable_profiling`：是否创建计时 Event 并收集每个 PlanOp 的耗时。
- `allow_cpu_fallback`：预留开关；当前 CUDA 图不支持跨设备 fallback，开启会明确
  返回 `Unsupported`。

## 编译结果

`Compiler::Compile()` 不修改原始 Model，而是复制 Graph 和常量表、运行默认 Pass、
把可执行图的 Tensor 设备改写为目标设备、逐节点检查后端能力，并生成 256 字节
对齐的 Workspace 规划。结果中的 Graph、PlanOp、常量、设备和运行开关共同组成
完整 `ExecutablePlan`。

`BuildEngine()` 等价于先调用 `Compiler::Compile()`，再用成功生成的 Plan 创建
`Engine`。具体流水线实现位于 `src/compiler/compiler.cpp`。
