# demo

本目录存放最小可运行示例，用来展示 CUTriton 接受的模型或 IR 文件形态。

当前 `gelu_graph.json` 是一个单节点 GELU 示例，主要用于 Python API、benchmark 和文档演示。

## 与其他目录的关系

- 被 `benchmarks/` 读取，用作基准测试输入。
- 被 `python/cutriton/api.py` 读取，用来验证 JSON 图编译入口。
- 与 `include/cutriton/ir/` 和 `src/ir/` 的 C++ IR 设计保持概念一致。
- 示例文件的语义应在 `docs/` 中解释清楚。

## 放什么

- 小型 JSON IR 示例。
- 后续可加入 ResNet stem、MLP、LayerNorm、Softmax 等最小模型。
- 不放大模型权重或生成产物。
