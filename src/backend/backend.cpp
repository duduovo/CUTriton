#include "cutriton/backend/backend.h"

#include <algorithm>
#include <memory>
#include <mutex>
#include <utility>

#include "cutriton/ir/fusion.h"
#include "cutriton/ir/op_schema.h"

#ifndef CUTRITON_ENABLE_CUDA
#define CUTRITON_ENABLE_CUDA 0
#endif

#if CUTRITON_ENABLE_CUDA
#include <cuda.h>
#endif

namespace cutriton {
namespace {

class UnsupportedCpuBackend final : public Backend {
 public:
  const char* name() const override { return "cpu_reference"; }
  Status CheckSupport(const Node&, const Graph&,
                      const BackendOptions&) const override {
    return Status::Unsupported(
        "the production CPU reference kernels are not implemented");
  }
  Status CreateKernel(const Node&, const Graph&, const BackendOptions&,
                      std::unique_ptr<OpKernel>*) const override {
    return Status::Unsupported(
        "the production CPU reference kernels are not implemented");
  }
};

class ViewKernel final : public OpKernel {
 public:
  const char* backend_name() const override { return "cuda_triton"; }
  const char* op_type() const override { return "Flatten"; }
  Status Compute(KernelContext* context) override {
    if (context == nullptr || context->node == nullptr ||
        context->tensors == nullptr) {
      return Status::InvalidArgument("Flatten KernelContext is invalid");
    }
    return Status::OK();
  }
};

[[maybe_unused]] Status CheckTensor(const Graph& graph, const std::string& name,
                                   const Device& device,
                                   const std::string& layout) {
  const auto* value = graph.FindValue(name);
  if (value == nullptr) {
    return Status::NotFound("Tensor description is missing: " + name);
  }
  CUTRITON_RETURN_IF_ERROR(value->tensor.Validate());
  if (value->tensor.dtype != DataType::kFloat32 &&
      value->tensor.dtype != DataType::kFloat16) {
    return Status::Unsupported("only float32/float16 are supported: " + name);
  }
  if (value->tensor.device_type != device.type ||
      value->tensor.device_id != device.id) {
    return Status::Unsupported("Tensor is assigned to the wrong device: " +
                               name);
  }
  if (!layout.empty() && value->tensor.layout != layout) {
    return Status::Unsupported("unsupported Tensor layout for " + name +
                               ": " + value->tensor.layout);
  }
  return Status::OK();
}

#if CUTRITON_ENABLE_CUDA
Status CudaStatus(CUresult result, const char* operation) {
  if (result == CUDA_SUCCESS) return Status::OK();
  const char* name = nullptr;
  const char* message = nullptr;
  cuGetErrorName(result, &name);
  cuGetErrorString(result, &message);
  return Status::RuntimeError(std::string(operation) + " failed: " +
                              (name == nullptr ? "CUDA_ERROR" : name) + " (" +
                              (message == nullptr ? "unknown" : message) + ")");
}

Status ComputeCapability(int device_id, int* capability) {
  CUTRITON_RETURN_IF_ERROR(CudaStatus(cuInit(0), "cuInit"));
  CUdevice device{};
  CUTRITON_RETURN_IF_ERROR(CudaStatus(cuDeviceGet(&device, device_id), "cuDeviceGet"));
  int major = 0;
  int minor = 0;
  CUTRITON_RETURN_IF_ERROR(CudaStatus(
      cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
                           device), "cuDeviceGetAttribute"));
  CUTRITON_RETURN_IF_ERROR(CudaStatus(
      cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
                           device), "cuDeviceGetAttribute"));
  *capability = major * 10 + minor;
  return Status::OK();
}
#else
Status ComputeCapability(int, int*) {
  return Status::Unsupported("CUTriton was built without CUDA support");
}
#endif

bool ArtifactMatches(const KernelArtifact& artifact, const Node& node,
                     const Graph& graph, int capability) {
  if (capability < artifact.min_compute_capability) {
    return false;
  }
  const std::string reference_name = !node.outputs().empty()
                                         ? node.outputs().front()
                                         : node.inputs().front();
  const auto* reference = graph.FindValue(reference_name);
  if (reference == nullptr ||
      artifact.dtype != DataTypeName(reference->tensor.dtype)) {
    return false;
  }
  auto layout_matches = [&](const std::string& name) {
    const auto* value = graph.FindValue(name);
    return value != nullptr &&
           (artifact.layout.empty() || value->tensor.layout == artifact.layout);
  };
  if (!node.inputs().empty() && !layout_matches(node.inputs().front())) {
    return false;
  }
  for (const auto& output : node.outputs()) {
    if (!layout_matches(output)) return false;
  }
  for (const auto& constraint : artifact.constraints) {
    if (constraint.kind == KernelConstraintKind::kInputCountEquals) {
      if (node.inputs().size() !=
          static_cast<std::size_t>(constraint.expected)) return false;
      continue;
    }
    if (constraint.kind == KernelConstraintKind::kAttributeInt64Equals) {
      if (GetIntAttribute(node, constraint.attribute_name,
                          constraint.default_value) != constraint.expected) {
        return false;
      }
      continue;
    }
    const auto& names = constraint.tensor_is_output ? node.outputs()
                                                    : node.inputs();
    if (constraint.tensor_index < 0 ||
        static_cast<std::size_t>(constraint.tensor_index) >= names.size()) {
      return false;
    }
    const auto* value = graph.FindValue(names[constraint.tensor_index]);
    if (value == nullptr || value->tensor.shape.size() !=
                                static_cast<std::size_t>(constraint.expected)) {
      return false;
    }
  }
  return true;
}

Status CheckCudaNode(const Node& node, const Graph& graph,
                     const BackendOptions& options,
                     const ArtifactRepository& repository) {
#if !CUTRITON_ENABLE_CUDA
  (void)node; (void)graph; (void)options; (void)repository;
  return Status::Unsupported("CUTriton was built without CUDA support");
#else
  if (options.device.type != DeviceType::kCUDA) {
    return Status::Unsupported("cuda_triton requires a CUDA device");
  }
  const auto schema = OpSchemaRegistry::Global().Find(node.op_type());
  if (schema == nullptr) {
    return Status::Unsupported("no OpSchema is registered for " +
                               node.op_type());
  }
  if (node.inputs().size() < schema->min_inputs ||
      node.inputs().size() > schema->max_inputs ||
      node.outputs().size() != schema->outputs) {
    return Status::Unsupported(node.op_type() +
                               " has an invalid input/output count");
  }
  for (const auto& input : node.inputs()) {
    CUTRITON_RETURN_IF_ERROR(CheckTensor(graph, input, options.device, ""));
  }
  for (const auto& output : node.outputs()) {
    CUTRITON_RETURN_IF_ERROR(CheckTensor(graph, output, options.device, ""));
  }
  if (node.op_type() == "Flatten") {
    if (std::find(graph.outputs().begin(), graph.outputs().end(),
                  node.outputs().front()) != graph.outputs().end()) {
      return Status::Unsupported(
          "Flatten graph outputs require an explicit copy kernel");
    }
    return Status::OK();
  }
  const auto* artifacts = repository.Find(node.op_type());
  if (artifacts == nullptr || artifacts->empty()) {
    return Status::Unsupported("No Kernel Pack entry for " + node.op_type());
  }
  int capability = 0;
  CUTRITON_RETURN_IF_ERROR(ComputeCapability(options.device.id, &capability));
  for (const auto& artifact : *artifacts) {
    if (ArtifactMatches(artifact, node, graph, capability)) return Status::OK();
  }
  return Status::Unsupported("No Kernel Pack variant matches dtype/layout/SM");
#endif
}

const KernelArtifact* DefaultArtifact(const ArtifactRepository& repository,
                                      const std::string& op_type,
                                      const std::string& dtype) {
  const auto* artifacts = repository.Find(op_type);
  if (artifacts == nullptr) return nullptr;
  const auto found = std::find_if(
      artifacts->begin(), artifacts->end(),
      [&](const KernelArtifact& artifact) {
        return artifact.is_default && artifact.dtype == dtype;
      });
  return found == artifacts->end() ? nullptr : &*found;
}

class CudaTritonBackend final : public Backend {
 public:
  const char* name() const override { return "cuda_triton"; }

  Status CheckSupport(const Node& node, const Graph& graph,
                      const BackendOptions& options) const override {
    ArtifactRepository repository;
    std::vector<std::string> paths = options.kernel_artifact_paths;
    if (paths.empty() && !options.kernel_artifact_dir.empty()) {
      paths.push_back(options.kernel_artifact_dir);
    }
    CUTRITON_RETURN_IF_ERROR(repository.Load(paths));
    return CheckCudaNode(node, graph, options, repository);
  }

  Status Lower(const Node& node, const LoweringContext& context,
               std::vector<ExecutionCandidate>* candidates) const override {
    if (candidates == nullptr || context.graph == nullptr ||
        context.options == nullptr || context.artifacts == nullptr ||
        context.catalog == nullptr) {
      return Status::InvalidArgument("CUDA lowering context is incomplete");
    }
    CUTRITON_RETURN_IF_ERROR(CheckCudaNode(
        node, *context.graph, *context.options, *context.artifacts));
    candidates->clear();
    if (node.op_type() == "Flatten") {
      ExecutionCandidate candidate;
      candidate.candidate_id = "view:flatten";
      candidate.is_default = true;
      candidate.steps.push_back(ViewInvocation{node.inputs().front(),
                                               node.outputs().front()});
      candidates->push_back(std::move(candidate));
      return Status::OK();
    }
    int capability = 0;
    CUTRITON_RETURN_IF_ERROR(
        ComputeCapability(context.options->device.id, &capability));
    const auto* output_value =
        context.graph->FindValue(node.outputs().front());
    if (output_value == nullptr) {
      return Status::NotFound("Lowered node output description is missing");
    }
    const std::string dtype = DataTypeName(output_value->tensor.dtype);
    const auto artifacts = context.catalog->Query(
        node.op_type(), dtype,
        node.inputs().empty()
            ? std::string{}
            : context.graph->FindValue(node.inputs().front())->tensor.layout,
        capability);
    for (const auto* artifact_ptr : artifacts) {
      const auto& artifact = *artifact_ptr;
      if (!ArtifactMatches(artifact, node, *context.graph, capability)) continue;
      ExecutionCandidate candidate;
      candidate.candidate_id = artifact.identity();
      candidate.is_default = artifact.is_default;
      candidate.steps.push_back(
          KernelInvocation{artifact, node.inputs(), node.outputs()});
      candidates->push_back(std::move(candidate));
    }
    CUTRITON_RETURN_IF_ERROR(RegisterBuiltinFusionAlternatives());
    for (const auto& alternative :
         FusionRegistry::Global().Find(node.op_type())) {
      ExecutionCandidate candidate;
      candidate.candidate_id = alternative.candidate_id;
      if (node.op_type() == "FusedConvBatchNormRelu" ||
          node.op_type() == "FusedConvBatchNorm") {
        const auto* conv_artifact =
            DefaultArtifact(*context.artifacts, "Conv", dtype);
        const auto* bn_artifact =
            DefaultArtifact(*context.artifacts, "BatchNormalization", dtype);
        const auto* relu_artifact =
            DefaultArtifact(*context.artifacts, "Relu", dtype);
        if (conv_artifact == nullptr || bn_artifact == nullptr ||
            (node.op_type() == "FusedConvBatchNormRelu" &&
             relu_artifact == nullptr)) continue;
        const std::string conv_output = node.name() + "::__conv";
        candidate.temporaries.push_back(
            CandidateTemporary{conv_output, node.outputs().front(), 0});
        candidate.steps.push_back(KernelInvocation{
            *conv_artifact, {node.inputs()[0], node.inputs()[1]}, {conv_output}});
        std::string bn_output = node.outputs().front();
        if (node.op_type() == "FusedConvBatchNormRelu") {
          bn_output = node.name() + "::__bn";
          candidate.temporaries.push_back(
              CandidateTemporary{bn_output, node.outputs().front(), 1});
        }
        candidate.steps.push_back(KernelInvocation{
            *bn_artifact,
            {conv_output, node.inputs()[2], node.inputs()[3], node.inputs()[4],
             node.inputs()[5]},
            {bn_output}});
        if (node.op_type() == "FusedConvBatchNormRelu") {
          candidate.steps.push_back(KernelInvocation{
              *relu_artifact, {bn_output}, {node.outputs().front()}});
        }
      } else if (node.op_type() == "AddRelu") {
        const auto* add_artifact =
            DefaultArtifact(*context.artifacts, "Add", dtype);
        const auto* relu_artifact =
            DefaultArtifact(*context.artifacts, "Relu", dtype);
        if (add_artifact == nullptr || relu_artifact == nullptr) continue;
        const std::string sum = node.name() + "::__add";
        candidate.temporaries.push_back(
            CandidateTemporary{sum, node.outputs().front(), 0});
        candidate.steps.push_back(KernelInvocation{
            *add_artifact, node.inputs(), {sum}});
        candidate.steps.push_back(KernelInvocation{
            *relu_artifact, {sum}, {node.outputs().front()}});
      } else if (node.op_type() == "GemmGelu") {
        const auto* gemm_artifact =
            DefaultArtifact(*context.artifacts, "Gemm", dtype);
        const auto* gelu_artifact =
            DefaultArtifact(*context.artifacts, "Gelu", dtype);
        if (gemm_artifact == nullptr || gelu_artifact == nullptr) continue;
        const std::string projected = node.name() + "::__gemm";
        candidate.temporaries.push_back(
            CandidateTemporary{projected, node.outputs().front(), 0});
        candidate.steps.push_back(KernelInvocation{
            *gemm_artifact, node.inputs(), {projected}});
        candidate.steps.push_back(KernelInvocation{
            *gelu_artifact, {projected}, {node.outputs().front()}});
      } else if (node.op_type() == "SkipLayerNormalization") {
        const auto* add_artifact =
            DefaultArtifact(*context.artifacts, "Add", dtype);
        const auto* norm_artifact =
            DefaultArtifact(*context.artifacts, "LayerNormalization", dtype);
        if (add_artifact == nullptr || norm_artifact == nullptr) continue;
        const std::string sum = node.name() + "::__add";
        candidate.temporaries.push_back(
            CandidateTemporary{sum, node.outputs().front(), 0});
        candidate.steps.push_back(KernelInvocation{
            *add_artifact, {node.inputs()[0], node.inputs()[1]}, {sum}});
        candidate.steps.push_back(KernelInvocation{
            *norm_artifact,
            {sum, node.inputs()[2], node.inputs()[3]},
            {node.outputs().front()}});
      }
      if (!candidate.steps.empty()) candidates->push_back(std::move(candidate));
    }
    return candidates->empty()
               ? Status::Unsupported("No compatible artifact for " + node.op_type())
               : Status::OK();
  }

  Status CreateKernel(const Node& node, const Graph&, const BackendOptions&,
                      std::unique_ptr<OpKernel>* kernel) const override {
    if (kernel == nullptr) {
      return Status::InvalidArgument("OpKernel output pointer is null");
    }
    if (node.op_type() == "Flatten") {
      *kernel = std::make_unique<ViewKernel>();
      return Status::OK();
    }
    return Status::Unsupported(
        "artifact-backed CUDA kernels are instantiated from KernelInvocation");
  }
};

Status RegisterBuiltin(std::shared_ptr<Backend> backend) {
  auto& registry = KernelRegistry::Global();
  if (registry.GetBackend(backend->name()) != nullptr) return Status::OK();
  Status status = registry.RegisterBackend(std::move(backend));
  return status.code() == ErrorCode::kAlreadyExists ? Status::OK() : status;
}

}  // namespace

Status Backend::Lower(const Node& node, const LoweringContext& context,
                      std::vector<ExecutionCandidate>* candidates) const {
  if (candidates == nullptr || context.graph == nullptr ||
      context.options == nullptr) {
    return Status::InvalidArgument("lowering context is incomplete");
  }
  CUTRITON_RETURN_IF_ERROR(
      CheckSupport(node, *context.graph, *context.options));
  ExecutionCandidate candidate;
  candidate.candidate_id = std::string(name()) + ":legacy:" + node.op_type();
  candidate.is_default = true;
  candidates->assign(1, std::move(candidate));
  return Status::OK();
}

KernelRegistry& KernelRegistry::Global() {
  static KernelRegistry registry;
  return registry;
}

Status KernelRegistry::RegisterBackend(std::shared_ptr<Backend> backend) {
  if (backend == nullptr) {
    return Status::InvalidArgument("Backend must not be null");
  }
  const std::string name = backend->name();
  if (name.empty()) return Status::InvalidArgument("Backend name must not be empty");
  std::unique_lock<std::shared_mutex> lock(mutex_);
  if (backends_.find(name) != backends_.end()) {
    return Status::AlreadyExists("Backend is already registered: " + name);
  }
  backends_.emplace(name, std::move(backend));
  return Status::OK();
}

std::shared_ptr<Backend> KernelRegistry::GetBackend(
    const std::string& name) const {
  std::shared_lock<std::shared_mutex> lock(mutex_);
  const auto it = backends_.find(name);
  return it == backends_.end() ? nullptr : it->second;
}

std::vector<std::string> KernelRegistry::AvailableBackends() const {
  std::shared_lock<std::shared_mutex> lock(mutex_);
  std::vector<std::string> names;
  for (const auto& entry : backends_) names.push_back(entry.first);
  std::sort(names.begin(), names.end());
  return names;
}

void KernelRegistry::ClearForTests() {
  std::unique_lock<std::shared_mutex> lock(mutex_);
  backends_.clear();
}

std::shared_ptr<Backend> CreateCpuReferenceBackend() {
  return std::make_shared<UnsupportedCpuBackend>();
}

std::shared_ptr<Backend> CreateCudaTritonBackend() {
  return std::make_shared<CudaTritonBackend>();
}

Status RegisterBuiltinBackends() {
  CUTRITON_RETURN_IF_ERROR(RegisterBuiltin(CreateCpuReferenceBackend()));
  CUTRITON_RETURN_IF_ERROR(RegisterBuiltin(CreateCudaTritonBackend()));
  return Status::OK();
}

}  // namespace cutriton
