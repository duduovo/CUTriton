# python

Python 目录有两条彼此独立的路径：

1. `cutriton/api.py` 与 `cutriton/kernels/`：轻量参考门面和数值参考算子，尚未绑定
   原生 C++ Engine。
2. `cutriton/kernel_sdk/` 与 `cutriton/triton_kernels/`：生产 CUDA 路径的 AOT
   Kernel SDK 和 Triton 定义。

Triton 只在构建 pack 时运行。C++ 生产 Runtime 读取 PTX/pack.json，不 import Python。
新增已有 IR op 的实现时注册 `KernelSpec`；新增 IR op 还要在 C++ 注册 OpSchema。
