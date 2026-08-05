# docs

本目录存放项目文档，用来解释架构、开发约定、设计路线和使用方式。

- [`architecture.md`](architecture.md)：从 Model 到 AOT Kernel Pack、Lowering、调优和
  CUDA Graph LRU 的整体架构。
- [`development.md`](development.md)：扩展 Kernel/op/融合/动态 shape 时的修改边界、
  构建测试命令和代码地图。
- 根目录 [`README.md`](../README.md)：面向使用者的项目门户、现状、性能与快速开始。

## 与其他目录的关系

- 从 `include/` 和 `src/` 总结 C++ API、实现边界和模块职责。
- 从 `python/` 总结 Python 门面和算子实验区设计。
- 从 `tests/` 和 `benchmarks/` 总结验证方法与性能评估方式。
- 文档不应替代测试；设计变化落地后要同步更新文档。

## 放什么

- 架构说明。
- 开发路线。
- 构建、测试、benchmark 指南。
- 设计决策记录和限制说明。
