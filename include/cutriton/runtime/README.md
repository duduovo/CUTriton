# include/cutriton/runtime

Runtime 执行已 Lower 的 Plan，不修改图语义。

- `ExecutablePlan`：Graph、候选 Invocation、常量、MemoryPlan/ProfilePlan 和运行配置。
- `Engine`/共享 EngineState：Plan、常量显存、PTX Module/Function Cache。
- `ExecutionContext`：shape/profile、绑定、Workspace、stream、调优选择、Graph LRU、Event。
- `MemoryPlanner`：256 B 对齐、best-fit、alias 和候选临时 scratch。
- `TuningMode`/`ShapeProfile`：AOT 候选选择和有限动态 shape 公共契约。

一个 profile 时自动选择；多个 profile 需调用 `SelectShapeProfile`。随后通过
`SetInputShape`、`ResolveShapes` 和 `GetResolvedTensorDesc` 获取本次具体描述。
`Run()` 等价于 `RunAsync(nullptr) + Synchronize()`。
