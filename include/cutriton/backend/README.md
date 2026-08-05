# include/cutriton/backend

本目录定义编译期 Lowering 和 AOT Kernel 产物契约。

- `backend.h`：`Backend::Lower`、`LoweringContext`、线程安全 `KernelRegistry`，以及
  Invocation Kernel/Module Cache 工厂。`CreateKernel` 仅保留给 Mock/旧式后端。
- `kernel_artifact.h`：pack v2 的参数来源、受限约束、`ArtifactRepository` 和
  `KernelCatalog`。
- `kernel_invocation.h`：`KernelInvocation`、`ViewInvocation`、候选临时值和
  `ExecutionCandidate`。

CUDA Backend 在编译期把 Node Lower 为候选；Runtime 不再调用 Backend 做能力检查。
缺少匹配 dtype/layout/rank/Attribute/SM 的产物时必须返回带原因的错误。
