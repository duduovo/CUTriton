# include/cutriton/backend

本目录定义后端系统的公共契约。后端负责回答“节点是否受支持”和“如何创建真实
可执行 Kernel”，没有计算实现时必须返回 `Unsupported`，不能用 NoOp 冒充成功。

## 主要接口

- `BackendOptions`：目标 `Device` 和离线 Kernel 产物目录。
- `Backend::CheckSupport()`：检查 dtype、静态 shape、layout、节点属性、设备能力
  和 Kernel 产物，并返回具体失败原因。
- `Backend::CreateKernel()`：采用相同上下文创建 `OpKernel`。
- `KernelContext`：单次执行使用的 Node、Tensor 表、Stream 和 Profiler 非拥有视图。
- `OpKernel::Compute()`：向指定后端提交一个节点的真实计算。
- `KernelRegistry`：线程安全的进程级后端注册表；重复名称返回 `AlreadyExists`。

## 当前内置后端

- `cuda_triton`：支持 FP32 `FusedConvBatchNorm`、`FusedConvBatchNormRelu`、
  `MaxPool`、`AddRelu`、`GlobalAveragePool`、`Gemm`，并把 `Flatten` 实现为
  零拷贝 view。这组算子可以覆盖 batch=1 的完整 ResNet-50。
- `cpu_reference`：保留注册入口，但生产数值 Kernel 尚未实现，会明确返回
  `Unsupported`。

具体 PTX 加载、manifest 校验和 CUDA Kernel 启动位于 `src/backend/backend.cpp`。
