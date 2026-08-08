# tests

本目录存放 CUTriton 的测试。

测试按语言分为 C++ 和 Python，分别覆盖核心库行为和 Python 门面行为。

## 与其他目录的关系

- `tests/cpp/` 链接 `src/` 编译出的 `cutriton` 库，并通过 `include/` 使用公共 API。
- `tests/python/` 通过 `python/` 包验证 Python API。
- 测试输入可以复用 `demo/` 中的示例，也可以在测试中临时构造。
- 测试结果应支撑 `docs/` 中对当前能力的描述。

## 放什么

- 单元测试。
- 小型集成测试。
- correctness 测试。
- 不放 benchmark；性能测试放到 `benchmarks/`。
