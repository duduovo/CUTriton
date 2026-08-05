#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "cutriton/backend/backend.h"
#include "cutriton/core/status.h"
#include "cutriton/runtime/executable_plan.h"
#include "cutriton/runtime/profiler.h"

namespace cutriton {

class ExecutionContext;
class EngineState;

// Engine 持有编译后的只读执行状态和共享常量资源，是创建执行上下文的工厂。
// ExecutionContext 通过 shared_ptr 共享 EngineState，因此可以晚于原 Engine 销毁。
class Engine {
 public:
  explicit Engine(ExecutablePlan plan);

  const ExecutablePlan& plan() const;
  // 每个 Context 独占 Stream、Workspace、中间 Tensor、Profiler 和 CUDA Graph。
  // 需要并行执行时，应为每个并发任务创建不同的 Context。
  std::unique_ptr<ExecutionContext> CreateExecutionContext() const;

 private:
  std::shared_ptr<EngineState> state_;
};

// 一次可复用的执行会话。单个 Context 非线程安全，并且同一时刻最多有一个未完成任务。
class ExecutionContext {
 public:
  explicit ExecutionContext(std::shared_ptr<EngineState> state);
  ~ExecutionContext();

  // 绑定 Graph 边界 Tensor。名称、shape、dtype、layout、设备和 Buffer 范围必须
  // 与 Plan 完全一致；常量和中间值不能通过这两个接口覆盖。
  // 重新绑定会使已经捕获的 CUDA Graph 失效，并在下次运行时重新捕获。
  Status BindInput(const std::string& name, Tensor tensor);
  Status BindOutput(const std::string& name, Tensor tensor);
  // 同步执行，语义等价于 RunAsync(nullptr) 后调用 Synchronize()。
  Status Run();
  // 将计算提交到 stream。CUDA 下传 nullptr 表示使用 Context 的内部 Stream；
  // 非空值必须是当前目标设备上的 CUstream。任务完成前不能再次运行或重新绑定。
  Status RunAsync(void* stream);
  // 等待当前任务完成，并在启用 profiling 时整理 GPU Event 计时结果。
  // 没有待完成任务时调用是安全的空操作。
  Status Synchronize();

  // CUDA profiling 结果只有在成功 Synchronize() 后才是本次执行的完整结果。
  const Profiler& profiler() const { return profiler_; }
  // 暴露当前 Tensor 视图用于诊断；其 Buffer 生命周期由 Context 管理。
  const std::unordered_map<std::string, Tensor>& tensors() const {
    return tensors_;
  }

 private:
  // 以下准备步骤采用惰性初始化，同一个 Context 后续执行会复用相关资源。
  Status PrepareKernels();
  Status PrepareTensors();
  Status ValidateBindings() const;
  void InvalidateCudaGraph();

  std::shared_ptr<EngineState> state_;
  bool kernels_prepared_{false};
  bool tensors_prepared_{false};
  bool run_pending_{false};
  std::vector<std::unique_ptr<OpKernel>> kernels_;
  std::unordered_map<std::string, Tensor> tensors_;
  std::shared_ptr<Buffer> workspace_;
  Profiler profiler_;
  // CUDA 私有状态的类型隐藏在实现文件中，避免公共头文件依赖 cuda.h。
  void* cuda_state_{nullptr};
};

}  // namespace cutriton
