# tests/cpp

- `test_core.cpp`：CPU-only 架构测试。覆盖常量、Buffer/Tensor、Pass、OpSchema、
  FusionRegistry、线程安全注册表、ShapeProfile、Workspace/alias、绑定和 Context 生命周期。
- `test_cuda.cpp`：真实 CUDA/Triton 集成测试。覆盖 pack v2/ABI/SHA/SM 错误、常量上传、
  Module Cache、融合与非融合调优、缓存命中、动态 H/W、Graph LRU、内部/外部 stream、
  profiling，以及 ResNet Stem 数值结果。

CPU 测试使用显式 MockBackend；生产后端没有 NoOp。CUDA 测试由 CMake 注入生成后的
Kernel Pack 路径，仅在 `CUTRITON_ENABLE_CUDA=ON` 时构建。
