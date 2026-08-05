# include

本目录是 CUTriton 的公共 C++ 头文件边界。外部程序通过它构建 `Model`、编译
`ExecutablePlan`，并使用 `Engine` 执行推理；`src/` 则负责实现这里声明的接口。

## 目录结构

- `cutriton/cutriton.h`：聚合全部公共 API 的入口头文件。
- `cutriton/core/`：状态、设备、Tensor 描述和 Buffer。
- `cutriton/ir/`：Model、Graph、Node、Value 和 Graph Pass。
- `cutriton/backend/`：后端能力检查、Kernel 抽象和注册表。
- `cutriton/compiler/`：从 Model 到 ExecutablePlan/Engine 的编译入口。
- `cutriton/runtime/`：执行计划、内存规划、Engine、Context 和 profiling。

## 边界约定

- 公共声明放在本目录，后端加载、CUDA Driver 调用等实现细节放在 `src/`。
- 头文件应避免不必要的实现依赖；CUDA 私有状态以不透明指针隐藏，CPU-only
  构建不需要包含 `cuda.h`。
- 轻量 getter、构造函数和 template 可以内联，复杂逻辑应在对应 `.cpp` 中实现。
- 当前项目仍处于 0.1 阶段，API 尚未承诺 ABI 稳定性。

`tests/cpp/` 直接使用这些接口验证公共契约；未来的 pybind11 绑定也应建立在这层
API 上，而不是依赖 `src/` 私有实现。
