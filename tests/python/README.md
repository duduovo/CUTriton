# tests/python

- `test_api.py`：轻量 Python reference API/JSON 图执行。
- `test_kernel_sdk.py`：KernelSpec 注册、ABI 尾部 reserved 参数、安全 source/grid/
  constraint AST 和重复 ID。
- `test_cuda_reference.py`：用 PyTorch 生成确定性 CUDA 参考结果，并与 C++ 测试输出对齐。

运行：

```bash
python -m pytest tests/python -q
```

Python reference API 尚未通过 pybind11 调用原生 Engine；GPU 性能以 C++ benchmark 为准。
