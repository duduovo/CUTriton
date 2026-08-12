#pragma once

#include <memory>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "cutriton/core/device.h"
#include "cutriton/core/status.h"
#include "cutriton/core/tensor.h"
#include "cutriton/ir/graph.h"
#include "cutriton/backend/kernel_artifact.h"
#include "cutriton/backend/kernel_invocation.h"

namespace cutriton {

class Profiler;
class CudaModuleCache;

// 后端在能力检查和 Lowering 阶段共享的只读配置。
// kernel_artifact_paths 可包含一个或多个 pack.json 或其父目录。
struct BackendOptions {
  Device device{};
  std::vector<std::string> kernel_artifact_paths;
  // Deprecated single-pack spelling kept as a source-compatible bridge.
  std::string kernel_artifact_dir;
};

struct LoweringContext {
  const Graph* graph{nullptr};
  const BackendOptions* options{nullptr};
  const ArtifactRepository* artifacts{nullptr};
  const KernelCatalog* catalog{nullptr};
};

// 单次算子执行所需的运行时对象。该结构及其指针只在 Compute() 调用期间有效。
// stream 是后端相关的非拥有句柄；CUDA 后端将其解释为 CUstream。
struct KernelContext {
  const Node* node{nullptr};
  std::unordered_map<std::string, Tensor>* tensors{nullptr};
  void* stream{nullptr};
  Profiler* profiler{nullptr};
};

// OpKernel 是一个已经针对特定后端完成 lowering 的可执行算子。
// Kernel 不拥有 Graph Tensor 和 Stream，它只通过 KernelContext 使用这些资源。
class OpKernel {
 public:
  virtual ~OpKernel() = default;
  virtual const char* backend_name() const = 0;
  virtual const char* op_type() const = 0;
  virtual Status Compute(KernelContext* context) = 0;
};

// Backend 在编译期把 IR 节点 Lower 成声明式 ExecutionCandidate。
class Backend {
 public:
  virtual ~Backend() = default;
  virtual const char* name() const = 0;
  // 完整检查 dtype、shape、layout、属性、设备能力和 Kernel 产物。
  // 不支持时应返回带具体原因的 Status，而不是创建空操作 Kernel。
  virtual Status CheckSupport(const Node& node, const Graph& graph,
                              const BackendOptions& options) const = 0;
  // Converts an IR node into artifact-backed executable candidates. The base
  // implementation creates a legacy candidate for non-artifact test backends.
  virtual Status Lower(const Node& node, const LoweringContext& context,
                       std::vector<ExecutionCandidate>* candidates) const;
  // 旧式非产物后端使用的 Kernel 工厂；CUDA 生产路径由 Invocation 工厂统一创建。
  virtual Status CreateKernel(const Node& node, const Graph& graph,
                              const BackendOptions& options,
                              std::unique_ptr<OpKernel>* kernel) const = 0;
};

// 进程级后端注册表。读写由 shared_mutex 保护，可供多个编译线程并发查询。
class KernelRegistry {
 public:
  static KernelRegistry& Global();

  // backend 不能为空且名称必须唯一；重复名称返回 kAlreadyExists。
  Status RegisterBackend(std::shared_ptr<Backend> backend);
  std::shared_ptr<Backend> GetBackend(const std::string& name) const;
  std::vector<std::string> AvailableBackends() const;
  // 仅供隔离单元测试状态使用，生产执行期间不应调用。
  void ClearForTests();

 private:
  mutable std::shared_mutex mutex_;
  std::unordered_map<std::string, std::shared_ptr<Backend>> backends_;
};

// 创建内置后端实例。当前 cpu_reference 对未实现的真实计算明确返回 Unsupported。
std::shared_ptr<Backend> CreateCpuReferenceBackend();
std::shared_ptr<Backend> CreateCudaTritonBackend();
// 幂等注册全部内置后端，已经存在的同名内置后端会被保留。
Status RegisterBuiltinBackends();

// Artifact-backed CUDA execution is intentionally separate from IR lowering.
Status CreateCudaModuleCache(int device_id,
                             std::shared_ptr<CudaModuleCache>* cache);
Status CreateInvocationKernel(const KernelInvocation& invocation, int device_id,
                              std::shared_ptr<CudaModuleCache> cache,
                              std::unique_ptr<OpKernel>* kernel);
std::size_t LoadedModuleCount(const std::shared_ptr<CudaModuleCache>& cache);

}  // namespace cutriton
