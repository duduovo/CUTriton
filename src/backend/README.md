# src/backend

本目录实现后端注册、能力检查和真实 Kernel 创建。目前实现集中在 `backend.cpp`。

## KernelRegistry

全局注册表使用 `shared_mutex` 保护读写。普通重复注册返回 `AlreadyExists`；内置后端
注册采用“存在则跳过”的幂等语义，便于 Compiler 和 Context 安全调用。

## cuda_triton

CUDA 后端当前支持静态 FP32、单设备和以下节点：

- `FusedConvBatchNormRelu`：NCHW/OIHW，使用 stride、pad、dilation 和 epsilon。
- `FusedConvBatchNorm`：残差分支末端使用的不带 ReLU 融合卷积。
- `MaxPool`：NCHW 二维最大池化，支持 kernel、stride、pad 和 dilation。
- `AddRelu`：等形状残差逐元素相加并执行 ReLU。
- `GlobalAveragePool`：NCHW 全局空间平均。
- `Gemm`：二维 FP32 矩阵乘法。
- `Flatten`：不启动 Kernel，创建零拷贝 Tensor view。

后端从 `kernel_artifact_dir` 读取版本化 manifest，校验 Triton 版本、dtype/layout、
最低 Compute Capability、PTX SHA-256 和符号信息，再通过 CUDA Driver API 加载
module、取得 function 并在传入的 `CUstream` 上启动。缺失产物或 SM 不兼容会在
`CheckSupport()` 阶段返回具体错误。

`cpu_reference` 目前只有显式 Unsupported 实现，生产路径不存在 NoOp 成功行为。
