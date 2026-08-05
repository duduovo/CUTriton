#include "cutriton/compiler/compiler.h"

#include <algorithm>
#include <utility>

#include "cutriton/backend/backend.h"
#include "cutriton/ir/pass.h"
#include "cutriton/runtime/memory_planner.h"

namespace cutriton {
namespace {

enum class ProfilePoint { kMin, kOpt, kMax };

const std::vector<int64_t>& PointShape(const ShapeRange& range,
                                       ProfilePoint point) {
  if (point == ProfilePoint::kMin) return range.min;
  if (point == ProfilePoint::kMax) return range.max;
  return range.opt;
}

Status ValidateProfiles(const Graph& graph,
                        const std::vector<ShapeProfile>& profiles) {
  for (const auto& profile : profiles) {
    if (profile.name.empty()) {
      return Status::InvalidArgument("ShapeProfile name must not be empty");
    }
    for (const auto& input_name : graph.inputs()) {
      const auto range_it = profile.inputs.find(input_name);
      if (range_it == profile.inputs.end()) {
        return Status::InvalidArgument("ShapeProfile is missing input: " +
                                       input_name);
      }
      const auto& range = range_it->second;
      if (range.min.empty() || range.min.size() != range.opt.size() ||
          range.min.size() != range.max.size()) {
        return Status::ShapeError("ShapeProfile ranks differ for " + input_name);
      }
      for (std::size_t axis = 0; axis < range.min.size(); ++axis) {
        if (range.min[axis] <= 0 || range.min[axis] > range.opt[axis] ||
            range.opt[axis] > range.max[axis]) {
          return Status::ShapeError("ShapeProfile min/opt/max is invalid for " +
                                    input_name);
        }
      }
    }
  }
  return Status::OK();
}

Status ApplyProfilePoint(Graph* graph, const ShapeProfile& profile,
                         ProfilePoint point) {
  for (const auto& input_name : graph->inputs()) {
    const auto* value = graph->FindValue(input_name);
    if (value == nullptr) return Status::NotFound("Graph input is missing");
    TensorDesc desc = value->tensor;
    desc.shape = PointShape(profile.inputs.at(input_name), point);
    CUTRITON_RETURN_IF_ERROR(graph->SetValueDesc(input_name, std::move(desc)));
  }
  return Status::OK();
}

Status ResolveProfileGraph(const Graph& optimized_graph,
                           const ShapeProfile& profile, ProfilePoint point,
                           Graph* output) {
  *output = optimized_graph;
  CUTRITON_RETURN_IF_ERROR(ApplyProfilePoint(output, profile, point));
  auto inference = CreateShapeInferencePass();
  return inference->Run(output);
}

std::size_t AlignWorkspace(std::size_t value) {
  return (value + 255U) & ~std::size_t{255U};
}

Status AddCandidateTemporaries(const Graph& graph,
                               const std::vector<PlanOp>& ops,
                               MemoryPlan* plan) {
  std::vector<std::size_t> slot_sizes;
  for (const auto& op : ops) {
    for (const auto& candidate : op.candidates) {
      for (const auto& temporary : candidate.temporaries) {
        const auto* like = graph.FindValue(temporary.like_value);
        if (like == nullptr || like->tensor.ByteSize() == 0) {
          return Status::ShapeError("Candidate temporary has no concrete shape: " +
                                    temporary.name);
        }
        if (slot_sizes.size() <= temporary.slot) {
          slot_sizes.resize(temporary.slot + 1, 0);
        }
        slot_sizes[temporary.slot] =
            std::max(slot_sizes[temporary.slot], like->tensor.ByteSize());
      }
    }
  }
  std::vector<std::size_t> offsets(slot_sizes.size());
  std::size_t cursor = AlignWorkspace(plan->workspace_size_bytes);
  for (std::size_t slot = 0; slot < slot_sizes.size(); ++slot) {
    offsets[slot] = cursor;
    cursor += AlignWorkspace(slot_sizes[slot]);
  }
  for (const auto& op : ops) {
    for (const auto& candidate : op.candidates) {
      for (const auto& temporary : candidate.temporaries) {
        const auto duplicate = std::find_if(
            plan->allocations.begin(), plan->allocations.end(),
            [&](const Allocation& allocation) {
              return allocation.value_name == temporary.name;
            });
        if (duplicate != plan->allocations.end()) continue;
        const auto* like = graph.FindValue(temporary.like_value);
        plan->allocations.push_back(Allocation{
            temporary.name, like->tensor.ByteSize(), offsets[temporary.slot],
            true, ""});
      }
    }
  }
  plan->workspace_size_bytes = AlignWorkspace(cursor);
  return Status::OK();
}

}  // namespace

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
  CUTRITON_RETURN_IF_ERROR(
      ValidateProfiles(result.graph(), options.shape_profiles));
  if (!options.shape_profiles.empty()) {
    CUTRITON_RETURN_IF_ERROR(ApplyProfilePoint(
        &result.mutable_graph(), options.shape_profiles.front(),
        ProfilePoint::kOpt));
  }
  result.set_target(options.target);
  result.set_enable_cuda_graph(options.enable_cuda_graph);
  result.set_enable_profiling(options.enable_profiling);
  result.mutable_tuning_config().mode = options.tuning_mode;
  result.mutable_tuning_config().cache_dir = options.tuning_cache_dir;
  result.mutable_tuning_config().warmup_iterations =
      options.tuning_warmup_iterations;
  result.mutable_tuning_config().measurement_iterations =
      options.tuning_measurement_iterations;
  result.set_cuda_graph_cache_capacity(options.cuda_graph_cache_capacity);
  result.mutable_backend_options().device =
      options.target == "cuda_triton" ? Device::CUDA(options.device_id)
                                       : Device::CPU();
  result.mutable_backend_options().kernel_artifact_dir =
      options.kernel_artifact_dir;
  result.mutable_backend_options().kernel_artifact_paths =
      options.kernel_artifact_paths;
  if (result.mutable_backend_options().kernel_artifact_paths.empty() &&
      !options.kernel_artifact_dir.empty()) {
    result.mutable_backend_options().kernel_artifact_paths.push_back(
        options.kernel_artifact_dir);
  }
  if (result.mutable_tuning_config().cache_dir.empty() &&
      !result.mutable_backend_options().kernel_artifact_paths.empty()) {
    result.mutable_tuning_config().cache_dir =
        result.mutable_backend_options().kernel_artifact_paths.front() +
        "/tuning";
  }

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

  ArtifactRepository artifacts;
  const ArtifactRepository* artifact_repository = nullptr;
  if (options.target == "cuda_triton") {
    CUTRITON_RETURN_IF_ERROR(
        artifacts.Load(result.backend_options().kernel_artifact_paths));
    artifact_repository = &artifacts;
  }
  KernelCatalog catalog(artifact_repository);
  const LoweringContext lowering_context{
      &result.graph(), &result.backend_options(), artifact_repository,
      artifact_repository == nullptr ? nullptr : &catalog};

  for (const auto& node : result.graph().nodes()) {
    PlanOp plan_op{
        node.id(), node.name(), node.op_type(), target_backend->name(),
        node.inputs(), node.outputs(), {}, 0};
    Status lowered =
        target_backend->Lower(node, lowering_context, &plan_op.candidates);
    if (!lowered.ok()) {
      return Status::Unsupported("Node " + node.name() + " (" +
                                 node.op_type() + ") is not supported by " +
                                 options.target + ": " + lowered.message());
    }
    if (plan_op.candidates.empty()) {
      return Status::Internal("Backend produced no candidate for " +
                              node.name());
    }
    for (std::size_t index = 0; index < plan_op.candidates.size(); ++index) {
      if (plan_op.candidates[index].is_default) {
        plan_op.selected_candidate = index;
        break;
      }
    }
    result.mutable_ops().push_back(std::move(plan_op));
  }

  MemoryPlanner planner;
  if (options.shape_profiles.empty()) {
    CUTRITON_RETURN_IF_ERROR(
        planner.Plan(result.graph(), &result.mutable_memory_plan()));
    CUTRITON_RETURN_IF_ERROR(AddCandidateTemporaries(
        result.graph(), result.ops(), &result.mutable_memory_plan()));
  } else {
    for (const auto& profile : options.shape_profiles) {
      ProfilePlan profile_plan;
      profile_plan.profile = profile;
      CUTRITON_RETURN_IF_ERROR(ResolveProfileGraph(
          result.graph(), profile, ProfilePoint::kMin, &profile_plan.min_graph));
      CUTRITON_RETURN_IF_ERROR(ResolveProfileGraph(
          result.graph(), profile, ProfilePoint::kOpt, &profile_plan.opt_graph));
      CUTRITON_RETURN_IF_ERROR(ResolveProfileGraph(
          result.graph(), profile, ProfilePoint::kMax, &profile_plan.max_graph));
      CUTRITON_RETURN_IF_ERROR(planner.Plan(
          profile_plan.max_graph, &profile_plan.max_memory_plan));
      CUTRITON_RETURN_IF_ERROR(AddCandidateTemporaries(
          profile_plan.max_graph, result.ops(),
          &profile_plan.max_memory_plan));
      result.mutable_profile_plans().push_back(std::move(profile_plan));
    }
    result.mutable_memory_plan() =
        result.profile_plans().front().max_memory_plan;
  }

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
