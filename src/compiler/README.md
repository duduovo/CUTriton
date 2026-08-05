# src/compiler

`Compiler::Compile` 的实际流程：

1. 复制 Model Graph 和常量，校验 ShapeProfile。
2. 用首个 profile opt 点运行默认 Pass；CUDA 仅改写 Plan 内 Tensor 设备。
3. 一次加载 ArtifactRepository，建立 KernelCatalog。
4. 对每个节点调用目标 `Backend::Lower`，保存全部 ExecutionCandidate。
5. 静态图直接规划内存；动态图分别推导 min/opt/max，并以 max 图规划 Workspace。
6. 为候选临时值增加可复用、256 B 对齐的 scratch slot。

原 Model 不被修改。一个 Plan 当前只使用一个目标后端；CUDA 图开启 CPU fallback 会
明确失败，因为跨设备 transfer 节点尚未实现。
