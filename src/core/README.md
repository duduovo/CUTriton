# src/core

本目录实现 CUTriton 的基础类型逻辑。

这些实现服务于所有上层模块，例如错误字符串、张量大小计算、静态形状校验和 host buffer 分配。

## 与其他目录的关系

- 实现 `include/cutriton/core/` 中声明的基础 API。
- 被 `src/ir/` 用于形状推导和校验。
- 被 `src/runtime/` 用于 Tensor 绑定和 buffer 管理。
- 被 `tests/cpp/` 直接覆盖。

## 放什么

- `Status` 字符串化。
- Tensor dtype、shape、byte size 相关计算。
- Buffer 分配和外部内存包装。
- 不放图结构、编译流程或后端执行逻辑。

