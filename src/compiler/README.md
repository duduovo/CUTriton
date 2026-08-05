# src/compiler

本目录实现 `Compiler::Compile()` 和 `BuildEngine()`，是 IR、Backend 与 Runtime 的
汇合点。

## Compile 流程

1. 校验输出指针和运行选项，幂等注册内置后端。
2. 复制 Model 的 Graph 与常量表，原 Model 保持不变。
3. 运行默认 Pass 流水线。
4. CUDA target 下仅改写 Plan Graph 中 Tensor 的设备类型和 device ID。
5. 对每个拓扑节点调用目标 Backend 的 `CheckSupport()`，生成 `PlanOp`。
6. 调用 `MemoryPlanner` 生成 Workspace 和 alias 规划。
7. 原子地把完整结果移动给调用方。

当前一个 Plan 只使用一个目标后端。CUDA 图中的 CPU fallback 需要显式设备搬运节点，
尚未实现，因此 `allow_cpu_fallback=true` 与 `cuda_triton` 组合会直接返回
`Unsupported`，不会生成缺少数据搬运的混合设备 Plan。

具体算子实现属于 `src/backend/`，图变换属于 `src/ir/`。
