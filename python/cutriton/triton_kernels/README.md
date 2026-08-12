# Triton Kernel modules

本目录只包含 AOT Triton 定义和 `KernelSpec` 注册。`resnet.py` 提供完整 FP32
ResNet-50 所需的融合与独立算子，每个计算 op 当前有 warps2/warps4 两个变体。

模块导入时只注册声明；pack builder 随后统一编译、计算 SHA-256 并写 pack.json。
生产 C++ Runtime 不导入本目录。
