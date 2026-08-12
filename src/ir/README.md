# src/ir

本目录实现 Graph 数据结构、Model 常量和编译前图优化。

## graph.cpp

- 校验并添加 Value、输入、输出和 Node。
- 管理节点编号、拓扑排序、节点删除及 producer/consumer map。
- `Model::AddConstant()` 校验 Host Buffer、静态 shape 和描述/Buffer 范围，并同时
  更新 Graph 常量描述与 Model 常量表。

## pass.cpp

形状推导通过线程安全 `OpSchemaRegistry` 分派，内置 schema 覆盖 Conv/融合 Conv、
BatchNorm、激活、Add/AddRelu、Flatten、Gemm/MatMul、普通 Pool 和 GAP。默认流水线还包含 DCE、
Conv+BN(+ReLU) 融合、Add+ReLU 融合、Flatten/Gemm 规范化和静态 shape 校验。

融合 Pass 要求中间结果为单消费者，并把 BatchNorm 的 epsilon 保存为融合节点属性；
没有紧随 ReLU 的 Conv+BN 会生成 `FusedConvBatchNorm`，残差末端会生成 `AddRelu`。
Normalization Pass 将 MatMul 规范成 Gemm，并为缺省 Flatten axis 填入 1。

`fusion.cpp` 的 `FusionRegistry` 注册融合节点对应的等价独立 op 序列。Pass 负责识别
语义子图，Backend/调优器决定最终采用融合还是非融合执行候选。

`ConstantFoldingPass` 当前只保留流水线扩展位置，不执行带数据的常量计算。添加真实
折叠时，需要同步更新 Model/Plan 的常量 Tensor，而不能只标记 `ValueDesc`。
