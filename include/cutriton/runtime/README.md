# include/cutriton/runtime

本目录定义执行已编译 Plan 所需的公共接口。运行期不再改变 Graph 的计算语义。

## ExecutablePlan

Plan 保存优化后的 Graph、拓扑有序的 `PlanOp`、常量 Host Tensor、目标设备、Kernel
产物目录、Workspace 规划以及 CUDA Graph/profiling 开关。交给 Engine 后应按只读
对象使用。

## Engine 与 ExecutionContext

- `Engine` 通过共享的 EngineState 持有 Plan 和设备常量。
- `ExecutionContext` 独占 Stream、Workspace、中间 Tensor、CUDA Graph 和 Event；
  Context 可以晚于创建它的 Engine 销毁。
- 单个 Context 非线程安全且只允许一个未完成任务；多个 Context 可以并行执行。
- `BindInput()`/`BindOutput()` 严格校验名称、shape、dtype、layout、设备、device ID
  和 Buffer 边界。重新绑定会使已有 CUDA Graph 失效。
- `Run()` 等价于 `RunAsync(nullptr) + Synchronize()`；`RunAsync()` 也可接收外部
  `CUstream`。

## 内存与计时

`MemoryPlanner` 排除图输入输出和常量，按 256 字节对齐对中间 Tensor 做 best-fit
复用，并传播 Flatten alias 生命周期。启用 profiling 时 CUDA Kernel 使用 GPU
Event 计时，Flatten 记录零耗时 view 事件；关闭时事件列表为空。

具体实现位于 `src/runtime/`。
