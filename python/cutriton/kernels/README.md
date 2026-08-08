# python/cutriton/kernels

本目录只存放 Python 数值参考算子。

真实 AOT Triton 定义位于相邻的 `triton_kernels/`，声明和构建 SDK 位于
`kernel_sdk/`。分开存放可以避免把 reference runner 与生产 Kernel 构建混在一起。

## 与其他目录的关系

- 被 `python/cutriton/api.py` 的 reference runner 调用。
- 被 `tests/python/` 用于验证 Python API 输出。
- 应与 AOT Triton Kernel 行为对齐，作为 correctness oracle。
- correctness 数据和设计说明可以沉淀到 `docs/`。

## 放什么

- 算子参考实现。
- 小规模 correctness helper。
- 不放 C++ Runtime 代码。
