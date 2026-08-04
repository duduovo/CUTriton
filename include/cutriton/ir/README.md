# include/cutriton/ir

本目录定义 CUTriton 的中间表示，也就是计算图和图 Pass 的公共接口。

IR 是模型导入、图优化、后端选择和内存规划之间的中心契约。

## 与其他目录的关系

- 依赖 `include/cutriton/core/` 的 `TensorDesc` 和 `Status`。
- 被 `include/cutriton/compiler/` 作为编译输入。
- 被 `include/cutriton/backend/` 用来判断单个 `Node` 是否支持。
- 被 `include/cutriton/runtime/` 的 `ExecutablePlan` 保存，供运行期按节点执行。
- `src/ir/` 实现图操作和默认 Pass。

## 放什么

- `Graph`、`Node`、`Model`、`ValueDesc`。
- 节点 attribute 类型和读取辅助函数。
- `GraphPass`、`PassManager` 和 Pass 创建函数。

