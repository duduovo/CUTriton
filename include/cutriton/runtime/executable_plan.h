#pragma once

#include <string>
#include <unordered_map>
#include <vector>

#include "cutriton/backend/backend.h"
#include "cutriton/ir/graph.h"
#include "cutriton/runtime/memory_planner.h"

namespace cutriton {

// 一个拓扑有序的执行步骤，将优化后 Graph 节点绑定到具体后端。
struct PlanOp {
  // node_id 对应 ExecutablePlan::graph().nodes() 中的节点下标。
  int node_id{-1};
  std::string node_name;
  std::string op_type;
  std::string backend_name;
  std::vector<std::string> inputs;
  std::vector<std::string> outputs;
};

// Model 经 Graph Pass、后端选择和内存规划后得到的完整可执行快照。
// mutable_* 接口供 Compiler 构建 Plan 使用；交给 Engine 后应按只读对象对待。
class ExecutablePlan {
 public:
  // 优化后的 Graph；CUDA 编译会把逻辑 Tensor 描述改写为目标 CUDA 设备。
  const Graph& graph() const { return graph_; }
  Graph& mutable_graph() { return graph_; }
  // 按拓扑顺序排列的后端执行步骤。
  const std::vector<PlanOp>& ops() const { return ops_; }
  std::vector<PlanOp>& mutable_ops() { return ops_; }
  // 中间 Tensor 的 Workspace 分配及零拷贝 alias 关系。
  const MemoryPlan& memory_plan() const { return memory_plan_; }
  MemoryPlan& mutable_memory_plan() { return memory_plan_; }
  // 编译请求的首选后端名称，例如 "cuda_triton"。
  const std::string& target() const { return target_; }
  void set_target(std::string target) { target_ = std::move(target); }
  // 后端共享配置，包含目标设备和离线 Kernel 产物目录。
  const BackendOptions& backend_options() const { return backend_options_; }
  BackendOptions& mutable_backend_options() { return backend_options_; }
  // 常量仍保存为 Host Tensor；Engine 创建执行资源时一次性上传到目标设备。
  const std::unordered_map<std::string, Tensor>& constants() const {
    return constants_;
  }
  std::unordered_map<std::string, Tensor>& mutable_constants() {
    return constants_;
  }
  // CUDA Graph 开启后，Context 首次运行捕获，后续运行直接 replay。
  bool enable_cuda_graph() const { return enable_cuda_graph_; }
  void set_enable_cuda_graph(bool value) { enable_cuda_graph_ = value; }
  // 关闭 profiling 时不创建计时 Event，Profiler 的 events() 保持为空。
  bool enable_profiling() const { return enable_profiling_; }
  void set_enable_profiling(bool value) { enable_profiling_ = value; }

 private:
  Graph graph_;
  std::vector<PlanOp> ops_;
  MemoryPlan memory_plan_;
  std::string target_;
  BackendOptions backend_options_;
  std::unordered_map<std::string, Tensor> constants_;
  bool enable_cuda_graph_{false};
  bool enable_profiling_{false};
};

}  // namespace cutriton
