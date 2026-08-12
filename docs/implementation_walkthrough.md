# CUTriton 秋招强化实现导读

> 更新时间：2026-08-12  
> AOT 工作区：`G:\CUTriton\CUTriton`  
> JIT 工作区：`G:\CUTriton\CUTriton-jit`  
> WSL Python：`/root/.venvs/cutriton/`

这份文档回答两个问题：

1. 目前究竟完成了什么；
2. 如何按一条完整调用链把代码读懂，而不是在文件之间来回跳。

建议不要一口气通读全部源码。先理解 AOT 主线，再理解 Python JIT/ORT 扩展。两条线解决的问题不同：

- **AOT C++ Runtime** 展示编译器、Kernel ABI、Runtime、CUDA 和性能工程能力；
- **Python JIT/ORT Runtime** 展示真实 ONNX 模型部署、自动回退、服务化和稳定性能力。

---

## 1. 当前仓库状态

### 1.1 分支与 worktree

| 工作区 | 分支 | 用途 |
| --- | --- | --- |
| `G:\CUTriton\CUTriton` | `feature/aot-transformer-optimization` | C++ AOT Runtime 主线 |
| `G:\CUTriton\CUTriton-jit` | `feature/python-jit-runtime` | Python JIT/ORT 与服务化扩展 |

AOT 分支从 `aot-cpp-baseline-v0.1` 建立。原始 Python JIT 改动仍有 stash 备份，没有执行硬重置：

```text
stash: wip-python-jit-v0.2-from-aot-20260811
```

当前实现尚未提交，仍在两个 worktree 中等待审查和拆分 commit。

### 1.2 验证状态

| 测试组 | 结果 |
| --- | ---: |
| AOT CPU CTest | 1/1 passed |
| AOT CUDA CTest | 5/5 passed |
| AOT Python SDK | 9 passed |
| JIT Python/GPU/服务 | 28 passed |
| Ruff / compileall / pip check | passed |

---

## 2. 已完成能力总览

### 2.1 AOT C++ Runtime

- FP16 已贯通常量、Tensor 描述、shape inference、Backend Lowering、参数绑定、Workspace 和结果校验。
- Kernel Pack ABI 升级到 v2，同时兼容历史 v1。
- v2 grid 使用受限 AST，支持：
  - `literal`
  - `input_dim` / `output_dim`
  - `input_numel` / `output_numel`
  - `meta`
  - `ceil_div`
  - `mul`
- variant launch metadata 可描述 `BLOCK_M/N/K` 等编译期参数。
- Loader 会拒绝未知表达式、非法字段、非法索引和不匹配的 PTX 哈希。
- 增加 Transformer IR/schema：
  - `Gelu`
  - `LayerNormalization`
  - `GemmGelu`
  - `SkipLayerNormalization`
- 增加两个 Pattern Pass：
  - `Gemm + Gelu -> GemmGelu`
  - `Add + LayerNormalization -> SkipLayerNormalization`
- 融合节点保留未融合候选，Runtime 可以按调优结果选择。
- 增加 FP16 Tensor Core GEMM、GELU、Add、LayerNorm 和两个融合 Kernel。
- GEMM 使用 `tl.dot`、二维 blocking、grouped program ordering 和四组 tile 配置。
- 增加 BERT-tiny 尺寸 FFN benchmark 与五轮 p50/p95 报告。
- 增加 Nsight Systems/Compute 一键采集脚本。

### 2.2 Python JIT/ORT Runtime

- 使用真实 `prajjwal1/bert-tiny`，固定到不可变 commit：

```text
6f75de8b60a9f8a2fdf7b69cbd86d9e64bcb3837
```

- 导出 opset 18、动态 batch/sequence ONNX，并生成 SHA256 manifest。
- 模型权重和 ONNX 放在仓库外，Git 只保存参考 manifest。
- 覆盖 `B={1,8}`、`S={16,64,128}` 和 padding attention mask。
- 固定随机种子 `20260811`，验证 `last_hidden_state`。
- Triton 子图必须同时通过数值门禁和子图性能门禁。
- 混合计划必须通过整图性能门禁，否则自动回退整图 ORT。
- 增加进程内 decision 热缓存，SQLite 继续作为跨进程持久层。
- 新动态 shape 首次请求直接走整图 ORT，避免请求线程创建大量分段 Session。
- 冷 shape 默认累计 256 次才触发后台资格验证，抑制动态 batch 编译风暴。
- 修复 ORT DLPack bool/uint8 dtype 互操作问题。
- KServe V2 gRPC benchmark 覆盖并发 `1/8/32`，记录吞吐、p50、p95、排队时间和动态 batch 分布。

---

## 3. 先建立整体心智模型

### 3.1 AOT 链路

```text
C++ Model / Graph
    |
    v
OpSchema + shape inference
    |
    v
Pattern Pass：识别 GemmGelu / SkipLayerNormalization
    |
    v
FusionRegistry：生成融合与未融合候选
    |
    v
Backend::Lower：候选 -> KernelInvocation 序列
    |
    v
ArtifactRepository：读取 pack.json + 校验 PTX
    |
    v
Engine / ExecutionContext：选 shape、选候选、分配 Workspace
    |
    v
通用 CUDA Launcher：求值 grid AST、绑定参数、cuLaunchKernel
    |
    v
CUDA Graph / Event / 调优缓存
```

关键设计点是：**生产 Runtime 不启动 Python，也不认识 Triton callable。** Python 只在构建阶段生成 PTX 和 manifest；C++ 运行时只读取经过校验的 Kernel Pack。

### 3.2 JIT/ORT 链路

```text
ONNX model
    |
    v
load_model + shape inference + graph partition
    |
    +------ 不支持节点 -----------------> ORT CUDA segment
    |
    +------ 支持节点 -> Triton candidate
                            |
                            v
                    数值 + 子图性能资格验证
                            |
                            v
                    整个 hybrid plan 性能门禁
                            |
                 +----------+----------+
                 |                     |
              通过                   不通过
                 |                     |
          hybrid execution         whole-model ORT
```

服务请求还会先经过 KServe 解码、动态 batching 和 deadline/backpressure 控制。

---

# 第一部分：跟读 AOT C++ Runtime

## 4. 第 1 站：从编译入口开始

先打开：

- [`include/cutriton/compiler/compiler.h`](../include/cutriton/compiler/compiler.h)
- [`src/compiler/compiler.cpp`](../src/compiler/compiler.cpp)

重点找：

```cpp
bool enable_transformer_fusions{true};
CreateDefaultCompilePasses(options.enable_transformer_fusions);
```

这里是整条 AOT 编译流程的入口。你需要先看清：

1. `CompileOptions` 保存了哪些编译和运行策略；
2. Model 在哪里被复制和优化；
3. Pass 在什么时候运行；
4. Backend 在什么时候把 IR Lower 成执行计划。

读完后尝试回答：

> 为什么 benchmark 只修改 `enable_transformer_fusions`，就能构造语义相同的融合/未融合对照？

答案提示：开关控制图上是否出现融合 IR；融合 IR 又会通过 FusionRegistry 生成完整候选序列。

---

## 5. 第 2 站：看 schema、shape inference 和 Pattern Pass

打开：

- [`include/cutriton/ir/pass.h`](../include/cutriton/ir/pass.h)
- [`src/ir/pass.cpp`](../src/ir/pass.cpp)

按以下顺序搜索：

```text
InferSkipLayerNormalization
GemmGeluFusionPass
SkipLayerNormalizationFusionPass
RegisterDefaultSchemas
CreateDefaultCompilePasses
```

### 5.1 先读 schema

重点观察 `Gemm` 的 dtype、rank、K 维和 bias 检查，以及 `LayerNormalization` 对 axis 和输入数量的约束。

这里体现的是编译器的基本原则：

> 不能因为 op 名字匹配就融合；shape、dtype、广播语义和输出可观察性也必须匹配。

### 5.2 再读 Pattern Pass

`GemmGeluFusionPass` 的核心逻辑可以概括为：

1. 找到 `Gemm`；
2. 确认它只有一个消费者；
3. 消费者必须是 `Gelu`；
4. 中间值不能是 graph output；
5. 把 `Gemm` 改写为 `GemmGelu`，删除后继 `Gelu`。

`SkipLayerNormalizationFusionPass` 对 `Add + LayerNormalization` 做相同类型的检查。

读完后回答：

> 如果 Gemm 输出同时被 Gelu 和另一个节点消费，为什么不能融合？

因为融合后原始 Gemm 输出不再存在，会改变另一个消费者看到的值。

---

## 6. 第 3 站：理解“融合后仍保留未融合候选”

打开：

- [`src/ir/fusion.cpp`](../src/ir/fusion.cpp)

搜索：

```text
GemmGelu
SkipLayerNormalization
```

这里不要只看 op 融合，还要理解 **IR 融合** 与 **执行候选融合** 的区别：

- Pattern Pass 把图改写为更高层语义节点；
- FusionRegistry 告诉 Backend：这个高层节点可以由哪些等价序列实现；
- Runtime 最终根据实测结果选候选，而不是强制选择融合 Kernel。

这是面试中很值得讲的一点：

> 编译器生成候选集合，Runtime 使用设备和 shape 相关的实测数据决策。

---

## 7. 第 4 站：读 Kernel Pack ABI v2 的 Python 定义

打开：

- [`python/cutriton/kernel_sdk/spec.py`](../python/cutriton/kernel_sdk/spec.py)
- [`python/cutriton/kernel_sdk/builder.py`](../python/cutriton/kernel_sdk/builder.py)

先读这些对象：

```text
ArgumentSpec
KernelVariant
KernelSpec
_validate_grid_expression
```

再看 builder 如何：

1. 导入 Kernel 模块并收集注册项；
2. 将受限 grid 表达式写入 manifest；
3. 把 variant metadata 作为 constexpr 传入 Triton 编译；
4. 计算 PTX SHA256；
5. 生成 ABI v2 `pack.json`。

要特别理解为什么不用任意 Python 表达式描述 grid：

- 任意表达式无法由 C++ 安全、确定地解释；
- 白名单 AST 可以校验深度、字段、索引和算术类型；
- Runtime 不需要嵌入 Python 解释器。

---

## 8. 第 5 站：读 Transformer Triton Kernel

打开：

- [`python/cutriton/triton_kernels/transformer.py`](../python/cutriton/triton_kernels/transformer.py)

建议按以下顺序读：

1. `gemm_fp16`
2. `gemm_gelu_fp16`
3. `layer_norm_fp16`
4. `skip_layer_norm_fp16`
5. 文件末尾的 `KernelSpec` 注册

读 GEMM 时画出两个 tile：

```text
A tile: BLOCK_M x BLOCK_K
B tile: BLOCK_K x BLOCK_N
accumulator: BLOCK_M x BLOCK_N, FP32
```

重点看：

- program id 如何映射到 M/N tile；
- `GROUP_M` 如何改变 program ordering；
- K 维如何循环加载；
- `tl.dot` 为什么能够走 Tensor Core；
- 边界 mask 如何处理非整块 shape；
- accumulator 为什么使用 FP32；
- GELU 为什么放进 GEMM epilogue。

再到文件末尾看四组 variant：不同 `BLOCK_M/N/K`、warps 和 stages 如何进入 Kernel Pack。

读完后回答：

> 融合 GemmGelu 为什么主要节省 launch 和中间张量 DRAM 往返，而不是减少矩阵乘 FLOPs？

---

## 9. 第 6 站：C++ 如何加载并验证 Kernel Pack

打开：

- [`include/cutriton/backend/kernel_artifact.h`](../include/cutriton/backend/kernel_artifact.h)
- [`src/backend/kernel_artifact.cpp`](../src/backend/kernel_artifact.cpp)

先看：

```cpp
enum class GridExpressionKind;
struct GridExpression;
struct KernelArtifact;
class ArtifactRepository;
```

然后跟踪 manifest 解析过程，重点观察：

- ABI v1 和 v2 如何区分；
- v1 grid 如何转换成内部统一表达；
- AST 如何限制递归深度和字段；
- launch metadata 如何校验；
- PTX 路径如何防止逃逸 pack 目录；
- SHA256 不一致为什么必须拒绝。

这里是项目安全性和工程性的一个重要展示点。

---

## 10. 第 7 站：Backend 如何把 IR 变成候选序列

打开：

- [`src/backend/backend.cpp`](../src/backend/backend.cpp)

搜索：

```text
GemmGelu
SkipLayerNormalization
ArtifactRepository
ExecutionCandidate
```

重点看两种 Lowering：

- 融合候选：一个融合 Kernel invocation；
- 未融合候选：多个普通 Kernel invocation。

同时观察 artifact 查询为什么带上 dtype、layout、shape 约束和 variant 信息。

读完后回答：

> 为什么 Runtime 不应该在执行时根据 `op_type` 写一大串 if/else 来决定参数？

因为 Backend 已经把参数来源、Kernel、grid 和中间值绑定 Lower 成声明式 invocation；Runtime 应只执行计划。

---

## 11. 第 8 站：通用 CUDA Launcher 如何执行 AST

打开：

- [`src/backend/cuda_launcher.cpp`](../src/backend/cuda_launcher.cpp)

搜索：

```text
EvaluateGridExpression
ResolveKernelArguments
cuLaunchKernel
```

沿着调用顺序读：

1. 从 invocation 找输入/输出 Tensor；
2. 将 `input_dim`、`output_numel`、`meta` 等 AST 节点求值；
3. 得到 grid x/y/z；
4. 按 manifest 中的参数 ABI 构造 `void**`；
5. 从 Module Cache 取得 `CUfunction`；
6. 调用 `cuLaunchKernel`。

这一步能帮助你真正理解 AOT 的含义：Triton 在构建期已生成 PTX，生产运行时只做参数和 launch 解析。

---

## 12. 第 9 站：Runtime、候选调优与 CUDA Graph

打开：

- [`include/cutriton/runtime/engine.h`](../include/cutriton/runtime/engine.h)
- [`src/runtime/engine.cpp`](../src/runtime/engine.cpp)

先建立对象关系：

```text
Engine
  └─ EngineState：共享常量、Module Cache、不可变 plan
       └─ ExecutionContext：stream、bindings、workspace、CUDA Graph cache
```

重点找：

- candidate benchmark；
- FP16 Host 结果转换与容差比较；
- tuning cache key；
- Workspace 复用；
- 外部 stream；
- CUDA Graph capture/replay。

调优缓存键需要覆盖：shape、dtype、layout、GPU/SM、Driver、PTX、pack、variant 等身份。否则换 GPU 或换 Kernel 后可能错误复用旧决策。

---

## 13. 第 10 站：用测试反向验证设计

先读 CPU/编译器测试：

- [`tests/cpp/test_core.cpp`](../tests/cpp/test_core.cpp)

重点看：

- FP16 `TensorDesc` 字节数；
- v1/v2 pack 兼容；
- 非法 AST 拒绝；
- 两个融合 Pass 的正向场景；
- 中间值可观察时拒绝融合。

再读 GPU 测试：

- [`tests/cpp/test_cuda.cpp`](../tests/cpp/test_cuda.cpp)

搜索 `GemmGelu` 和 `SkipLayerNormalization`，观察测试如何：

1. 构造 `tokens=17` 的非整块 shape；
2. 上传 FP16 输入和常量；
3. 运行真实 CUDA Kernel；
4. 与 CPU reference 比较；
5. 验证候选数量与 CUDA Graph。

最后读 SDK 测试：

- [`tests/python/test_kernel_sdk.py`](../tests/python/test_kernel_sdk.py)

这些测试是理解 ABI 边界最快的方式。

---

## 14. 第 11 站：读 FFN benchmark

打开：

- [`benchmarks/transformer_ffn_native.cpp`](../benchmarks/transformer_ffn_native.cpp)
- [`benchmarks/transformer_ffn_compare.py`](../benchmarks/transformer_ffn_compare.py)

### 14.1 C++ benchmark

按以下函数读：

```text
MakeInputs
BuildModel
CreateRunner
Benchmark
main
```

注意 benchmark 如何使用同一套权重分别构造融合和未融合 Engine，并用 CUDA Event 排除 H2D/D2H。

### 14.2 Python 对照

观察它如何构造语义一致的：

- PyTorch eager；
- ONNX Runtime CUDA 子图；
- CUTriton native executable。

最终报告执行 50 次预热、200 次测量、5 轮，输出每轮 p50/p95、软件栈和误差。

当前 RTX 4060 五轮结果：

| 实现 | p50 | p95 |
| --- | ---: | ---: |
| CUTriton AOT fused | 0.033728 ms | 0.038912 ms |
| CUTriton AOT unfused | 0.038848 ms | 0.044704 ms |
| ORT CUDA subgraph | 0.113728 ms | 0.143360 ms |
| PyTorch eager | 0.030816 ms | 0.031712 ms |

结论必须准确表述为：

- AOT fused 相对 ORT 子图约 `3.372x`；
- 融合相对未融合约为 `1.154x`；
- PyTorch eager 仍略快于当前 AOT；
- 这不是 BERT 整图加速。

---

# 第二部分：跟读 Python JIT/ORT Runtime

JIT worktree 位于 `G:\CUTriton\CUTriton-jit`。从当前文档目录出发，下面的相对链接会进入该 worktree。

## 15. 第 1 站：配置和安全策略

打开：

- [`python/cutriton/config.py`](../../CUTriton-jit/python/cutriton/config.py)

重点读：

```text
FallbackPolicy
ShapeRange
ShapeProfile
CompileOptions
BatchingOptions
ServiceOptions
```

特别注意：

```python
performance_margin = 0.05
qualification_min_requests = 256
```

前者要求 Triton 子图至少比 ORT 子图快 5%；BERT benchmark 会显式提高到 20%。后者避免服务中大量一次性动态 batch shape 触发编译风暴。

---

## 16. 第 2 站：ONNX 加载和图分区

打开：

- [`python/cutriton/graph.py`](../../CUTriton-jit/python/cutriton/graph.py)
- [`python/cutriton/registry.py`](../../CUTriton-jit/python/cutriton/registry.py)
- [`python/cutriton/triton_backend.py`](../../CUTriton-jit/python/cutriton/triton_backend.py)

先看 `load_model()`：

- trusted model root；
- 模型大小限制；
- external data 路径逃逸防护；
- ONNX checker 和 shape inference；
- model SHA256。

再看 `partition_graph()` 如何根据 Registry capability 划分连续 ORT/Triton segment。

最后浏览 `create_default_registry()`，理解哪些 op 会成为 Triton 候选，哪些情况会明确交给 ORT。

---

## 17. 第 3 站：Engine 的三层性能门禁

打开：

- [`python/cutriton/api.py`](../../CUTriton-jit/python/cutriton/api.py)

这是 JIT 主线最重要的文件，建议分三次读。

### 17.1 第一次：读 Engine 生命周期

搜索：

```text
class Engine
warmup
create_context
close
```

理解 Engine 保存不可变 graph、initializers、decision cache 和单线程编译器；每个 ExecutionContext 独占 CUDA stream 和 ORT Session。

### 17.2 第二次：读子图资格验证

搜索：

```text
_qualify
_run_triton
_median_cuda_ms
_compare_outputs
```

`_qualify()` 做两道门禁：

1. Triton 输出与 ORT segment 数值一致；
2. Triton 延迟满足 `performance_margin`。

失败结果也会写入 negative cache，避免每次请求重复编译。

### 17.3 第三次：读整图门禁

搜索：

```text
_qualify_plan
hybrid
whole_plan_gate
```

这里把已通过的 Triton segment 与 ORT segment 组合成 hybrid plan，再与 whole-model ORT 比较。若 hybrid 更慢，整个签名被标记为 disabled。

读完后回答：

> 为什么“每个子图都更快”仍不代表“整图更快”？

因为 segment 边界、额外 launch、同步、数据依赖和 ORT/Triton 切换成本可能吃掉子图收益。

---

## 18. 第 4 站：冷 shape 为什么先走整图 ORT

仍在：

- [`python/cutriton/api.py`](../../CUTriton-jit/python/cutriton/api.py)

搜索：

```text
_submit_cold_plan_preparation
_prepare_cold_plan
_cold_signature_is_hot
_plan_decision
```

真实 KServe 压测曾暴露一个问题：动态 batching 会形成 batch 2、3、5、13、27 等 shape。如果每个新 shape 都立即构造几十个分段 ORT Session，p95 会达到秒级。

修复后的策略：

1. 首次未知 shape 直接用已存在的 whole-model ORT Session 返回；
2. 同一 shape 达到热度阈值后，才在单线程编译器后台准备候选；
3. request thread 不再承担分段 Session 初始化；
4. Engine shutdown 会等待已提交的准备任务。

这是一个很适合在面试中讲的“从压测发现系统问题并修复”的案例。

---

## 19. 第 5 站：decision cache

打开：

- [`python/cutriton/cache.py`](../../CUTriton-jit/python/cutriton/cache.py)

观察两层缓存：

- `_memory`：进程内热路径，不重复访问 SQLite；
- `decisions.sqlite3`：跨进程持久化 enabled/disabled/failed 决策。

需要理解：

- 为什么失败也需要缓存；
- SQLite WAL 和原子事务解决什么问题；
- LRU prune 如何限制磁盘占用；
- 为什么 cache key 必须包含模型、shape、stride、dtype、Kernel 源码和软件/设备身份。

---

## 20. 第 6 站：ORT CUDA I/O Binding 与 bool 修复

打开：

- [`python/cutriton/ort_backend.py`](../../CUTriton-jit/python/cutriton/ort_backend.py)

搜索：

```text
OrtSegmentRunner
_from_dlpack
_normalize_output_dtype
run_with_iobinding
```

重点看：

- ORT 如何绑定到 ExecutionContext 的用户 CUDA stream；
- CUDA Tensor 如何通过 DLPack 或原始指针零拷贝绑定；
- 输出如何转换回 torch Tensor；
- ORT 1.28 与 PyTorch 2.11 的 bool DLPack 差异如何处理。

真实 BERT 图中 `Equal` 等节点会产生 bool 中间张量。未修复时 ORT 会看到 `uint8`，随后拒绝下一个 segment 的输入。

---

## 21. 第 7 站：动态 batching 与 KServe 服务

打开：

- [`python/cutriton/batching.py`](../../CUTriton-jit/python/cutriton/batching.py)
- [`python/cutriton/service.py`](../../CUTriton-jit/python/cutriton/service.py)

先读 `DynamicBatcher`：

```text
infer
_schedule
_preferred_ready
_take_compatible
_execute
```

重点理解请求只有在以下信息一致时才能合并：

- 输入名；
- dtype；
- device；
- 非 batch 维度。

`_execute()` 会：

1. `torch.cat` 合并输入；
2. 从 context pool 取独占 context；
3. 执行一次推理；
4. 按原请求 batch size 拆分输出；
5. 记录实际 batch size、请求数和 queue time。

再读 `InferenceService.ModelInfer()`，跟踪 KServe raw bytes 如何变为 pinned Host Tensor，再异步复制到 GPU。

---

## 22. 第 8 站：BERT-tiny 导出与 manifest

打开：

- [`tools/export_bert_tiny.py`](../../CUTriton-jit/tools/export_bert_tiny.py)
- [`models/bert-tiny.manifest.json`](../../CUTriton-jit/models/bert-tiny.manifest.json)

重点观察导出脚本如何：

- 拒绝 `main/latest` 等移动 revision；
- 固定完整模型 commit；
- 只导出 `last_hidden_state`；
- 声明动态 batch/sequence；
- 使用 opset 18；
- 运行 ONNX checker；
- 对每个产物写入 SHA256 和字节数。

模型权重没有提交到 Git。当前外部模型目录是：

```text
G:\CUTriton\models\bert-tiny-onnx
```

---

## 23. 第 9 站：BERT 正确性与性能报告

打开：

- [`benchmarks/benchmark_bert_tiny.py`](../../CUTriton-jit/benchmarks/benchmark_bert_tiny.py)

按顺序读：

```text
_load_manifest
_inputs
_measure_pair
main
```

`_measure_pair()` 会逐轮交错 CUTriton 与 ORT，并在奇偶轮交换顺序，降低温度、频率和调度漂移造成的偏差。

六组 shape 最终全部自动选择整图 ORT，p50 相对 ORT 比值为：

```text
0.977 - 1.014
```

因此正确结论是：

> 当前 BERT hybrid plan 没有整图加速；性能门禁正确禁用了候选，回退路径满足不慢于 ORT 2% 的要求。

不要把它写成“BERT 加速 3.372x”。`3.372x` 只属于 AOT FFN 子图。

---

## 24. 第 10 站：KServe 压测

打开：

- [`benchmarks/benchmark_kserve_bert.py`](../../CUTriton-jit/benchmarks/benchmark_kserve_bert.py)

它会启动本地真实 gRPC server，通过 KServe V2 client 发请求，而不是直接调用 Python 函数。

最终 RTX 4060、sequence 128、每档 200 请求的结果：

| 并发 | 吞吐 | p50 | p95 |
| ---: | ---: | ---: | ---: |
| 1 | 188 req/s | 5.23 ms | 6.32 ms |
| 8 | 660 req/s | 11.94 ms | 13.49 ms |
| 32 | 1041 req/s | 27.28 ms | 37.19 ms |

报告还保留：

- 平均排队时间；
- 平均动态 batch；
- 每个实际 batch size 出现次数；
- 端到端延迟包含 H2D、D2H、编码和 gRPC response。

---

## 25. 第 11 站：JIT 测试与 CI

重点阅读：

- [`tests/python/test_runtime_gpu.py`](../../CUTriton-jit/tests/python/test_runtime_gpu.py)
- [`tests/python/test_batching.py`](../../CUTriton-jit/tests/python/test_batching.py)
- [`tests/python/test_cache.py`](../../CUTriton-jit/tests/python/test_cache.py)
- [`.github/workflows/ci.yml`](../../CUTriton-jit/.github/workflows/ci.yml)
- [`.github/workflows/nightly-gpu.yml`](../../CUTriton-jit/.github/workflows/nightly-gpu.yml)

特别关注这些回归场景：

- ORT CUDA Tensor I/O；
- bool 输出保持 ONNX dtype；
- 冷 shape 首请求走整图 ORT；
- FP16/BF16 容差；
- CUDA Graph 重复运行无明显显存增长；
- dynamic batch 保持请求边界；
- 热 decision 不重新打开 SQLite；
- nightly compute-sanitizer。

---

# 第三部分：边读边运行

## 26. 第一轮：只跑快速测试

### AOT CPU + SDK

```bash
source /root/.venvs/cutriton/bin/activate
cd /mnt/g/CUTriton/CUTriton

cmake --build build-aot-dev --parallel 8
ctest --test-dir build-aot-dev --output-on-failure
python -m pytest -q tests/python
```

### JIT 全量测试

```bash
source /root/.venvs/cutriton/bin/activate
cd /mnt/g/CUTriton/CUTriton-jit

PYTHONPATH=python python -m pytest -q tests/python
python -m ruff check python tests benchmarks demo tools
python -m compileall -q python tests benchmarks demo tools
python -m pip check
```

---

## 27. 第二轮：只跑 Transformer GPU 链路

```bash
source /root/.venvs/cutriton/bin/activate
cd /mnt/g/CUTriton/CUTriton

cmake --build build-aot-cuda-dev --parallel 8
ctest --test-dir build-aot-cuda-dev \
  -R 'transformer|cuda_tests' \
  --output-on-failure
```

运行完整五轮性能报告：

```bash
python benchmarks/transformer_ffn_compare.py \
  --executable build-aot-cuda-dev/cutriton_transformer_ffn_benchmark \
  --tokens 1024 \
  --warmup 50 \
  --iterations 200 \
  --rounds 5 \
  --json build-aot-cuda-dev/transformer_ffn_report.json
```

会生成：

```text
build-aot-cuda-dev/transformer_ffn_report.json
build-aot-cuda-dev/transformer_ffn_report.md
```

---

## 28. 第三轮：跑真实 BERT

```bash
source /root/.venvs/cutriton/bin/activate
cd /mnt/g/CUTriton/CUTriton-jit

PYTHONPATH=python python benchmarks/benchmark_bert_tiny.py \
  /mnt/g/CUTriton/models/bert-tiny-onnx/manifest.json \
  --output /mnt/g/CUTriton/models/bert-tiny-onnx/bert_tiny_report.json
```

然后跑 KServe：

```bash
PYTHONPATH=python python benchmarks/benchmark_kserve_bert.py \
  /mnt/g/CUTriton/models/bert-tiny-onnx/manifest.json \
  --output /mnt/g/CUTriton/models/bert-tiny-onnx/bert_kserve_report.json
```

---

## 29. 第四轮：Nsight

入口：

- [`tools/profile_transformer_ffn.sh`](../tools/profile_transformer_ffn.sh)

```bash
cd /mnt/g/CUTriton/CUTriton
bash tools/profile_transformer_ffn.sh \
  build-aot-cuda-dev/cutriton_transformer_ffn_benchmark \
  build-aot-cuda-dev/nsight-transformer \
  1024
```

当前环境限制：

- `nsys` 已安装，但当前 WSL 生成的 report 没有采集到 CUDA Kernel 数据；
- `ncu` 未安装，因此尚无真实 Nsight Compute 指标；
- 脚本已就绪，但不能把空报告或缺失指标写成已完成结果。

---

# 第四部分：如何把项目讲给面试官

## 30. 三分钟叙述模板

可以按下面顺序讲：

1. **目标**：实现一个生产运行时不依赖 Python 的 C++ AOT 推理引擎，同时用 Python JIT/ORT 支持真实 ONNX 部署。
2. **编译链**：自定义 IR 做 shape inference 和 Pattern Fusion，再生成融合/未融合候选。
3. **Kernel ABI**：Triton 只负责离线生成 PTX；版本化 manifest 描述参数、约束、安全 grid AST 和 variant metadata。
4. **Runtime**：C++ Loader 验证 pack，Backend Lower 成 invocation，通用 CUDA Launcher 解释 AST 并调用 Driver API。
5. **优化**：实现 FP16 Tensor Core GemmGelu 与 SkipLayerNorm，Runtime 依据真实 shape/GPU 调优，而非强制融合。
6. **真实结果**：BERT-tiny 尺寸 FFN 相对 ORT 子图约 3.372x，但仍略慢于 PyTorch eager。
7. **整图诚实回退**：BERT hybrid 没有整图收益，因此自动禁用，回退路径保持在 ORT 的 2% 内。
8. **系统问题**：KServe 压测发现冷动态 batch 初始化分段 Session 导致秒级 p95，改为 whole-model ORT fast path 和热度触发后台资格验证后解决。

---

## 31. 你真正应该掌握的八个问题

读完代码后，尝试不看文档回答：

1. AOT 与 JIT 的区别是什么？本项目为什么两者都保留？
2. 为什么 Kernel Pack 需要版本、PTX 哈希和受限 grid AST？
3. `Gemm + Gelu` 在什么情况下不能融合？
4. 为什么融合节点还需要保留未融合候选？
5. Runtime 调优缓存键为什么必须包含 GPU、Driver、shape 和 PTX？
6. 为什么子图更快不保证混合整图更快？
7. 冷动态 shape 为什么应该先走整图 ORT？
8. CUDA Event、端到端服务延迟和 Engine build time 分别测量了什么？

如果你能结合具体文件和一次真实 bug 回答这八个问题，这个项目就不再只是“会跑”，而是能用于推理引擎岗位面试。

---

## 32. 当前边界

以下能力没有实现，也不应在简历中宣称：

- INT8/FP8；
- Paged Attention、KV Cache；
- 完整 Attention/QKV 融合；
- 多 GPU；
- 完整原生 ONNX Importer；
- BERT 整图加速；
- 成功的 Nsight Compute 报告。

当前最准确的项目定位是：

> 一个具有版本化 Kernel ABI、图融合、候选调优、CUDA Runtime 和真实 ONNX/服务回退验证的推理引擎实验项目；已在 FP16 Transformer FFN 子图上达到相对 ORT 的明确加速，并对未达标整图保持自动回退。
