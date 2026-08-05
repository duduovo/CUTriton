# src/runtime

`engine.cpp` 实现共享 EngineState 和 Context 私有执行状态：常量一次上传、Module Cache、
profile/实际 shape、候选选择、Workspace view、内部/外部 stream、CUDA Graph LRU 和
Event profiling。

调优在独立非阻塞 stream 上对完整候选序列预热并计时，验证 FP32 输出后把中位数结果
按 tuning key 原子写入单独 JSON。`kUseCache` 先查精确 shape，profile 下再查 opt 点。

Graph key 包含 profile、shape、候选、边界 Buffer 地址/offset 和 Workspace 地址。
重新绑定不会清空全部缓存，只是不再命中地址不同的 Graph；LRU 默认容量为 4。

`memory_planner.cpp` 排除输入输出和常量，按生命周期做 256 B 对齐 best-fit，并传播
Flatten alias 生命周期。`profiler.cpp` 保存 CPU/真实 CUDA Event 的统一结果。
