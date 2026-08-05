# include/cutriton/core

本目录定义其他模块共同依赖的基础类型，不包含图优化或算子调度逻辑。

## 文件与职责

- `status.h`：`Status`、`ErrorCode` 和错误传播宏。
- `device.h`：CPU/CUDA `Device` 描述和设备属性结构。
- `tensor.h`：`DataType`、`DeviceType`、`TensorDesc` 和 Tensor view。
- `buffer.h`：Host/CUDA 内存的所有权与拷贝接口。

## 重要契约

- `TensorDesc::Validate()` 校验 dtype、静态 shape 和非零字节大小；设备与布局是否
  符合某个算子的要求由绑定检查和具体 Backend 继续验证。
  `ByteSize()` 由元素数量与 dtype 大小计算。
- `Tensor` 由描述、共享 `Buffer` 和字节偏移组成，多个 Tensor 可以引用同一块内存，
  因而可表达 Workspace 子区间和 Flatten alias。
- `Buffer::AllocateHost()` 和 `AllocateCuda()` 分别分配主机内存与 CUDA 显存；
  `CopyFromHost()`/`CopyToHost()` 提供同步边界拷贝。
- `WrapExternal()` 不接管外部地址的释放，调用方必须保证外部内存活得足够久。
- CUDA 功能仅在 `CUTRITON_ENABLE_CUDA=1` 时可用，否则返回 `Unsupported`。

对应实现位于 `src/core/`。
