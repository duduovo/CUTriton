# include/cutriton/core

本目录定义 CUTriton 最基础、最稳定的公共类型。

Core 层是其他模块共同依赖的底座，包括错误状态、张量描述、内存 buffer 和设备描述。

## 与其他目录的关系

- `include/cutriton/ir/` 用 `TensorDesc` 描述图中 value。
- `include/cutriton/backend/` 和 `include/cutriton/runtime/` 用 `Tensor`、`Buffer` 和 `Status` 做执行契约。
- `src/core/` 实现本目录声明的基础方法。
- `tests/cpp/` 直接测试 Tensor 和 Buffer 的基础行为。

## 放什么

- `Status` 和错误码。
- `DataType`、`DeviceType`、`TensorDesc`、`Tensor`。
- `Buffer` 和 `Device`。
- 不放图优化、后端选择或运行调度逻辑。

