# src/ir

本目录实现 CUTriton 的计算图和图优化 Pass。

这里是模型语义被规范化和优化的地方，决定进入后端选择前的图长什么样。

## 与其他目录的关系

- 实现 `include/cutriton/ir/` 中声明的 Graph 和 Pass API。
- 使用 `src/core/` 的 TensorDesc 校验能力。
- 被 `src/compiler/` 调用，作为编译流水线的一部分。
- 编译后的图会进入 `src/runtime/` 的 ExecutablePlan 和 MemoryPlanner。

## 放什么

- 图增删查改。
- 拓扑排序。
- 形状推导。
- 死节点消除。
- 算子融合和规范化 Pass。
- 不放具体后端 Kernel。

