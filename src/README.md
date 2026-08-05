# src

本目录实现 `include/cutriton/` 声明的 C++ API，并由根目录 `CMakeLists.txt` 编译为
`cutriton` 库。公共类型和函数声明不应放在这里。

## 模块

- `core/`：Status、Tensor 和 Host/CUDA Buffer。
- `ir/`：Graph 操作、形状推导和默认优化 Pass。
- `backend/`：线程安全注册表、Triton manifest/PTX 加载及 CUDA Kernel 启动。
- `compiler/`：复制 Model、运行 Pass、检查后端并构建 ExecutablePlan。
- `runtime/`：内存规划、Engine/Context、Stream、CUDA Graph 和 profiling。

## 实现边界

- CUDA 代码受 `CUTRITON_ENABLE_CUDA` 控制；关闭时 CPU-only 构建不链接 CUDA
  Driver，并对 CUDA 操作返回明确错误。
- 生产路径没有 NoOp Kernel。缺少真实实现、产物不匹配或设备不受支持时必须失败。
- `tests/cpp/` 验证公共行为，测试替身应显式实现为 MockBackend，不进入生产后端。
- Python/Triton 源码和离线构建工具分别位于 `python/` 与 `tools/`，不放入本目录。
