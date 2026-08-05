# src/ir

本目录实现 Graph 数据结构、Model 常量和编译前图优化。

## graph.cpp

- 校验并添加 Value、输入、输出和 Node。
- 管理节点编号、拓扑排序、节点删除及 producer/consumer map。
- `Model::AddConstant()` 校验 Host Buffer、静态 shape 和描述/Buffer 范围，并同时
  更新 Graph 常量描述与 Model 常量表。

## pass.cpp

当前形状推导覆盖 Conv/融合 Conv、BatchNorm、激活、Add、Flatten、Gemm/MatMul、
普通 Pool 和 GlobalAveragePool。默认流水线还包含 DCE、Conv+BN+ReLU 融合、
Flatten/Gemm 规范化和静态 shape 校验。

融合 Pass 要求中间结果为单消费者，并把 BatchNorm 的 epsilon 保存为融合节点属性；
Normalization Pass 将 MatMul 规范成 Gemm，并为缺省 Flatten axis 填入 1。

`ConstantFoldingPass` 当前只保留流水线扩展位置，不执行带数据的常量计算。添加真实
折叠时，需要同步更新 Model/Plan 的常量 Tensor，而不能只标记 `ValueDesc`。
