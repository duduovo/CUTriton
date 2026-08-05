# Kernel SDK

`spec.py` 定义 `ArgumentSpec`、`KernelVariant`、`KernelSpec` 和进程内注册表；
`builder.py` 发现模块、调用 Triton AOT 编译并生成 Kernel Pack v2。

允许的参数来源只有 input/output 指针、Tensor dim/product/numel、Node Attribute、
literal 和明确 reserved 参数。Grid 与 constraints 使用固定白名单 AST。`KernelSpec`
在编译 PTX 前校验 ABI 顺序、唯一默认变体和所有声明字段。

自定义模块可通过薄 CLI 加载：

```bash
python tools/build_triton_kernels.py \
  --output /mnt/g/Ubuntu_/CUTriton/custom-pack \
  --module my_package.my_kernels
```
