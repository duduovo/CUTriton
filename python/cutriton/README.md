# python/cutriton

- `api.py`：JSON/ONNX 结构读取和轻量 Python reference Engine。
- `kernels/`：GELU、Softmax、LayerNorm 等 Python correctness 参考。
- `kernel_sdk/`：`ArgumentSpec`、`KernelVariant`、`KernelSpec`、注册表和 pack builder。
- `triton_kernels/`：模块化 Triton Kernel；当前覆盖 ResNet-50 FP32 路径。

KernelSpec 只能使用 SDK 列出的 Tensor 指针、维度/numel、Attribute、literal 和 reserved
参数来源。Grid 仅接受受限 `ceil_div(output_numel, divisor)` AST；约束仅接受白名单
操作。任意 Python 表达式不会进入 manifest 或 C++ Runtime。
