# include/cutriton/compiler

`Compiler::Compile` 把 Model 编译为只读 `ExecutablePlan`，`BuildEngine` 是 Compile 后
构造 Engine 的便捷入口。

`CompileOptions` 的主要配置：

- `target`、`device_id`
- `kernel_artifact_paths`：一个或多个 pack.json/父目录
- `tuning_cache_dir`、`tuning_mode`、预热和测量次数
- `shape_profiles`：输入 min/opt/max 范围
- `enable_cuda_graph`、`cuda_graph_cache_capacity`
- `enable_profiling`

`kernel_artifact_dir` 仅是 0.1 阶段的源码兼容桥，新代码应使用 paths。编译器复制
Model，运行 Pass，为 CUDA 图改写可执行 Tensor 的设备，Lower 每个节点，为每个
profile 生成具体图和最大 Workspace；原始 Model 不被修改。
