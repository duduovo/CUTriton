# src

本目录实现 `include/cutriton/` 声明的 C++ API，并由根目录 `CMakeLists.txt` 编译为
`cutriton` 库。公共类型和函数声明不应放在这里。

## 模块

- `core/`：Status、Tensor 和 Host/CUDA Buffer。
- `ir/`：Graph、OpSchema、FusionRegistry 和默认优化 Pass。
- `backend/`：Kernel Pack 仓库/目录、编译期 Lowering、参数绑定和通用 CUDA launcher。
- `compiler/`：运行 Pass、生成候选和 profile，并构建 ExecutablePlan。
- `runtime/`：内存规划、调优、Engine/Context、Stream、CUDA Graph LRU 和 profiling。

## 实现边界

- CUDA 代码受 `CUTRITON_ENABLE_CUDA` 控制；关闭时 CPU-only 构建不链接 CUDA
  Driver，并对 CUDA 操作返回明确错误。
- 生产路径没有 NoOp Kernel。缺少真实实现、产物不匹配或设备不受支持时必须失败。
- `tests/cpp/` 验证公共行为，测试替身应显式实现为 MockBackend，不进入生产后端。
- Triton 定义和 Kernel SDK 位于 `python/cutriton/`；`tools/` 只保留薄 CLI 和环境脚本。
