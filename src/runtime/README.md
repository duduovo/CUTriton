# src/runtime

本目录实现 ExecutablePlan 的资源准备、内存规划、异步执行和计时。

## engine.cpp

- `EngineState` 共享 Plan 和一次性初始化的设备常量；Context 不依赖原 Engine 寿命。
- 每个 `ExecutionContext` 惰性创建 Kernel、Workspace、中间 Tensor、CUDA primary
  context、内部非阻塞 Stream、Graph 和可选 Event。
- 输入输出绑定执行完整描述与 Buffer 边界校验；运行期间禁止重新绑定。
- `RunAsync()` 支持内部或外部 `CUstream`，同一 Context 只允许一个 pending run；
  `Synchronize()` 等待 Stream 并整理 profiling 结果。
- 开启 CUDA Graph 后首次执行 capture/instantiate，之后 replay；重新绑定会销毁旧
  Graph，捕获失败直接返回错误。

## memory_planner.cpp

规划器排除外部输入输出和常量，计算每个中间 Value 的最后使用位置，按 256 字节
对齐使用 best-fit 空闲块复用。Flatten 输出复用源 Buffer，并把源值生命周期延长到
alias 的最后一次使用。

## profiler.cpp

CPU Kernel 可使用 `ScopedCpuTimer`；CUDA Kernel 的真实 GPU 耗时由 `engine.cpp` 中的
CUDA Event 产生。关闭 profiling 时不创建 Event，结果列表为空。
