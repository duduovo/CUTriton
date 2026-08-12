# src/backend

- `kernel_artifact.cpp`：递归 JSON parser、pack v2/ABI 校验、安全 PTX 路径、SHA-256、
  `ArtifactRepository` 和 `KernelCatalog`。
- `backend.cpp`：线程安全 Backend 注册、CUDA 能力检查、编译期 Lowering，以及融合/
  非融合候选构造。它不包含按 op 启动 Kernel 的 Runtime switch。
- `cuda_launcher.cpp`：共享 `CudaModuleCache`、manifest 驱动的参数绑定和统一
  `cuLaunchKernel`。

当前 AOT op 为 FP32 Conv、BN、Relu、Add、三种融合 op、MaxPool、GAP 和 Gemm；
Flatten 由 `ViewInvocation` 执行。Module 按 PTX SHA 共享，Function 按 SHA+symbol 共享。
pack 缺失、schema/ABI 不支持、SHA 损坏、约束不满足或 SM 不兼容都会明确失败。
