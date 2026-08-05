# include/cutriton/ir

本目录定义 CUTriton 的中间表示和 Graph Pass 公共接口。IR 只描述“计算什么”，
不决定由哪个后端执行，也不持有运行期 Stream 或 Workspace。

## 图结构

- `ValueDesc`：名称、`TensorDesc` 和 `is_constant` 标记。
- `Node`：算子类型、输入输出名称及类型安全的 Attribute 表。
- `Graph`：统一管理 Value、Node、图输入输出、拓扑关系和节点编号。
- `Model`：持有 Graph 以及真实常量 Tensor 表。

`Model::AddConstant()` 只接受已定义、静态 shape 的 Host Tensor，并同步在 Graph 中
建立 `is_constant=true` 的 Value。常量数据之后会复制进 `ExecutablePlan`，由
Engine 在首次准备设备资源时上传。

## 默认 Pass 流水线

1. 拓扑排序和形状推导。
2. 常量折叠扩展点与死节点消除。
3. `Conv + BatchNormalization + Relu` 融合，并保留 BatchNorm epsilon。
4. 再次拓扑排序、形状推导。
5. Flatten/Gemm 规范化、再次 DCE 和静态 shape 校验。

当前 `ConstantFoldingPass` 仍是扩展点，不计算新的常量数据；模型常量本身已经由
`Model` 的常量表完整保存。具体实现位于 `src/ir/`。
