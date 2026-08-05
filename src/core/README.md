# src/core

本目录实现所有上层模块共享的基础类型。

## 当前实现

- `status.cpp`：错误类别名称和 `Status::ToString()`。
- `tensor.cpp`：dtype/设备名称、元素大小、静态 shape、元素数量、字节数和描述校验。
- `buffer.cpp`：Host 分配、外部地址包装、CUDA Driver 显存分配，以及同步 H2D/D2H
  拷贝。

CUDA Buffer 使用目标设备的 primary context，所有 Driver API 失败都会转换为包含
CUDA 错误名称和说明的 `Status::RuntimeError`。CPU-only 构建仍保留相同公共接口，
但调用 CUDA 分配或拷贝会返回 `Unsupported`。

`WrapExternal()` 只记录地址、大小和设备，不释放调用方内存；拥有型 Buffer 通过
`shared_ptr<void>` 管理 Host 或 CUDA 分配的生命周期。

本层不负责 Tensor 的图语义、Workspace 规划或 Kernel 调度。
