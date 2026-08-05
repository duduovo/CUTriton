# CUTriton 架构

## 分层边界

```text
Model / IR
  -> OpSchema + Graph Pass + FusionRegistry
  -> ArtifactRepository + KernelCatalog
  -> Backend::Lower
  -> ExecutionCandidate {KernelInvocation | ViewInvocation}
  -> ExecutablePlan + MemoryPlan/ProfilePlan
  -> EngineState
  -> ExecutionContext
  -> ArgumentBinder + generic CUDA launcher
```

- Core/IR 不依赖 CUDA、Triton 或 Python。
- Python/Triton 只构建 AOT Kernel Pack。
- Compiler 决定图语义、候选、设备、shape 和内存。
- Runtime 只解释已经 Lower 的 Invocation，不按 `op_type` 重新调度。

## 编译期

`OpSchemaRegistry` 保存每个 IR op 的输入输出契约和具体 shape 推导函数。默认 Pass
执行拓扑排序、shape 推导、DCE、融合识别、规范化和静态实例验证。动态模型在每个
ShapeProfile 的 min/opt/max 点重用相同 shape 函数。

`ArtifactRepository` 使用真正的 JSON parser 一次读取并校验 pack v2，包括 ABI、
安全路径、字段类型和 PTX SHA-256。`KernelCatalog` 按 op、dtype、layout 和 SM 做初筛；
Backend 再评估 Node Attribute、rank 等声明式约束。

`Backend::Lower` 输出一个或多个 `ExecutionCandidate`。候选由 `KernelInvocation` 和
`ViewInvocation` 组成；融合节点同时可以携带融合 Kernel 及独立 op 序列。最终选择或
可调优候选被完整保存到 `ExecutablePlan`，运行时不再次 Lower。

## Shape 与内存

静态图直接生成一个 MemoryPlan。动态图要求有限 min/opt/max profile；Compiler 为每个
profile 保存三个具体 Graph，并按 max 点规划 Workspace。Context 在同一 profile 内
切换 shape 时重做具体 shape 推导和 Tensor view，但不重新分配最大 Workspace。

中间 Tensor 采用单块 Workspace、256 字节对齐和 best-fit 复用。候选临时 Tensor
使用可复用 scratch slot；Flatten alias 会传播生命周期且不启动 Kernel。

## 运行期

`EngineState` 是共享只读状态，持有 Plan、上传后的常量以及按 PTX SHA/符号缓存的
CUDA Module/Function。Context 可以晚于原 Engine 销毁；多个 Context 可并行，单个
Context 非线程安全且只允许一个 pending run。

Context 的职责只有：

1. 选择 profile 并解析实际 shape。
2. 根据调优缓存选择 ExecutionCandidate。
3. 创建实际 Tensor/Workspace view。
4. 根据 manifest 的有序规则绑定指针和标量。
5. 使用统一 `cuLaunchKernel` 提交所有计算步骤。
6. 捕获或 replay CUDA Graph，并在同步后整理 Event 时间。

## 自动调优

调优只测试 Kernel Pack 中已有的 AOT 候选，不启动 Python。它使用独立非阻塞 stream、
CUDA Event、预热/多次测量和中位数。候选首次使用时会与确定性 FP32 基线进行数值比较。
每个 tuning key 一个版本化 JSON 文件，经临时文件和原子 rename 写入。

键覆盖 op/融合候选、属性、shape/dtype/layout、GPU UUID/SM、Driver，以及 pack、PTX、
生成器、Triton 和 ABI 身份，防止环境或产物变化后误用旧结果。

## CUDA Graph LRU

每个 Context 维护独立 LRU，键包含 profile、具体 shape、候选 ID、输入输出地址与
byte offset、Workspace 地址。重新绑定地址只会令不匹配项失效；同一 profile 内切换
shape 可以复用历史 Graph。捕获失败返回错误，不静默回退。

## 扩展规则

- 新 Kernel 变体：只增加 Python `KernelSpec` 和测试。
- 已有 op 的新实现：同样不修改 C++ launcher。
- 新 IR op：增加 C++ `OpSchema` 与 Python `KernelSpec`。
- 新融合：注册语义等价候选，并为候选临时值提供描述。
- 新参数来源或约束操作：必须扩展受限 schema、C++ parser/binder 和负向测试；禁止
  在 manifest 中嵌入任意 Python/C++ 表达式。
