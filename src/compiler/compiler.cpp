#include "cutriton/compiler/compiler.h"

#include <utility>

#include "cutriton/backend/backend.h"
#include "cutriton/ir/pass.h"
#include "cutriton/runtime/memory_planner.h"

namespace cutriton {

Status Compiler::Compile(const Model& model, const CompileOptions& options,
                         ExecutablePlan* plan) const {
  if (plan == nullptr) {
    return Status::InvalidArgument("ExecutablePlan output pointer is null");
  }
  if (options.allow_cpu_fallback && options.target == "cuda_triton") {
    return Status::Unsupported(
        "CPU fallback in a CUDA graph requires device transfer nodes and is "
        "not implemented");
  }
  CUTRITON_RETURN_IF_ERROR(RegisterBuiltinBackends());

  ExecutablePlan result;
  result.mutable_graph() = model.graph();
  result.mutable_constants() = model.constants();
  result.set_target(options.target);
  result.set_enable_cuda_graph(options.enable_cuda_graph);
  result.set_enable_profiling(options.enable_profiling);
  result.mutable_backend_options().device =
      options.target == "cuda_triton" ? Device::CUDA(options.device_id)
                                       : Device::CPU();
  result.mutable_backend_options().kernel_artifact_dir =
      options.kernel_artifact_dir;

  for (const auto& entry : result.graph().values()) {
    if (entry.second.is_constant &&
        result.constants().find(entry.first) == result.constants().end()) {
      return Status::InvalidArgument(
          "Constant Value has no Tensor data: " + entry.first);
    }
  }

  auto passes = CreateDefaultCompilePasses();
  CUTRITON_RETURN_IF_ERROR(passes.Run(&result.mutable_graph()));

  if (options.target == "cuda_triton") {
    for (const auto& entry : result.graph().values()) {
      TensorDesc desc = entry.second.tensor;
      if (desc.dtype == DataType::kUnknown) {
        continue;
      }
      desc.device_type = DeviceType::kCUDA;
      desc.device_id = options.device_id;
      CUTRITON_RETURN_IF_ERROR(
          result.mutable_graph().SetValueDesc(entry.first, std::move(desc)));
    }
  }

  auto& registry = KernelRegistry::Global();
  auto target_backend = registry.GetBackend(options.target);
  if (target_backend == nullptr) {
    return Status::NotFound("Target backend is not registered: " +
                            options.target);
  }

  for (const auto& node : result.graph().nodes()) {
    Status supported = target_backend->CheckSupport(
        node, result.graph(), result.backend_options());
    if (!supported.ok()) {
      return Status::Unsupported("Node " + node.name() + " (" +
                                 node.op_type() + ") is not supported by " +
                                 options.target + ": " + supported.message());
    }
    result.mutable_ops().push_back(PlanOp{
        node.id(), node.name(), node.op_type(), target_backend->name(),
        node.inputs(), node.outputs()});
  }

  MemoryPlanner planner;
  CUTRITON_RETURN_IF_ERROR(
      planner.Plan(result.graph(), &result.mutable_memory_plan()));

  *plan = std::move(result);
  return Status::OK();
}

Status BuildEngine(const Model& model, const CompileOptions& options,
                   std::unique_ptr<Engine>* engine) {
  if (engine == nullptr) {
    return Status::InvalidArgument("Engine output pointer is null");
  }
  ExecutablePlan plan;
  Compiler compiler;
  CUTRITON_RETURN_IF_ERROR(compiler.Compile(model, options, &plan));
  *engine = std::make_unique<Engine>(std::move(plan));
  return Status::OK();
}

}  // namespace cutriton
