#include "cutriton/runtime/engine.h"
#include "cutriton/core/buffer.h"
#include "cutriton/ir/pass.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <mutex>
#include <optional>
#include <sstream>
#include <type_traits>
#include <utility>
#include <variant>

#ifndef CUTRITON_ENABLE_CUDA
#define CUTRITON_ENABLE_CUDA 0
#endif

#if CUTRITON_ENABLE_CUDA
#include <cuda.h>
#endif

namespace cutriton {
namespace {

bool Contains(const std::vector<std::string>& values, const std::string& name) {
  return std::find(values.begin(), values.end(), name) != values.end();
}

Status ValidateBoundTensor(const std::string& name, const TensorDesc& expected,
                           const Tensor& tensor) {
  CUTRITON_RETURN_IF_ERROR(tensor.desc().Validate());
  if (!tensor.defined() || tensor.buffer()->empty()) {
    return Status::InvalidArgument("Tensor Buffer is not defined: " + name);
  }
  const auto& actual = tensor.desc();
  if (actual.shape != expected.shape || actual.dtype != expected.dtype ||
      actual.layout != expected.layout ||
      actual.device_type != expected.device_type ||
      actual.device_id != expected.device_id) {
    return Status::InvalidArgument("Tensor description does not match plan: " +
                                   name);
  }
  if (tensor.buffer()->device_type() != expected.device_type ||
      tensor.buffer()->device_id() != expected.device_id) {
    return Status::InvalidArgument("Tensor Buffer is on the wrong device: " +
                                   name);
  }
  if (tensor.byte_offset() > tensor.buffer()->size_bytes() ||
      expected.ByteSize() >
          tensor.buffer()->size_bytes() - tensor.byte_offset()) {
    return Status::InvalidArgument("Tensor Buffer is too small: " + name);
  }
  return Status::OK();
}

#if CUTRITON_ENABLE_CUDA
Status CudaStatus(CUresult result, const char* operation) {
  if (result == CUDA_SUCCESS) {
    return Status::OK();
  }
  const char* name = nullptr;
  const char* message = nullptr;
  cuGetErrorName(result, &name);
  cuGetErrorString(result, &message);
  return Status::RuntimeError(std::string(operation) + " failed: " +
                              (name == nullptr ? "CUDA_ERROR" : name) +
                              " (" + (message == nullptr ? "unknown" : message) +
                              ")");
}

struct EventPair {
  CUevent start{nullptr};
  CUevent end{nullptr};
  bool timed{false};
};

struct CudaGraphEntry {
  std::string key;
  CUgraph graph{nullptr};
  CUgraphExec graph_exec{nullptr};
  std::uint64_t last_used{0};
};

struct CudaExecutionState {
  CUdevice device{};
  CUcontext context{nullptr};
  CUstream owned_stream{nullptr};
  CUstream active_stream{nullptr};
  std::vector<CudaGraphEntry> graphs;
  std::uint64_t graph_clock{0};
  std::vector<EventPair> events;
};

Status InitializeCuda(CudaExecutionState* state, int device_id,
                      std::size_t op_count, bool profiling) {
  CUTRITON_RETURN_IF_ERROR(CudaStatus(cuInit(0), "cuInit"));
  CUTRITON_RETURN_IF_ERROR(
      CudaStatus(cuDeviceGet(&state->device, device_id), "cuDeviceGet"));
  CUTRITON_RETURN_IF_ERROR(CudaStatus(
      cuDevicePrimaryCtxRetain(&state->context, state->device),
      "cuDevicePrimaryCtxRetain"));
  CUTRITON_RETURN_IF_ERROR(
      CudaStatus(cuCtxSetCurrent(state->context), "cuCtxSetCurrent"));
  CUTRITON_RETURN_IF_ERROR(CudaStatus(
      cuStreamCreate(&state->owned_stream, CU_STREAM_NON_BLOCKING),
      "cuStreamCreate"));
  state->events.resize(op_count);
  if (profiling) {
    for (auto& pair : state->events) {
      CUTRITON_RETURN_IF_ERROR(
          CudaStatus(cuEventCreate(&pair.start, CU_EVENT_DEFAULT),
                     "cuEventCreate"));
      CUTRITON_RETURN_IF_ERROR(
          CudaStatus(cuEventCreate(&pair.end, CU_EVENT_DEFAULT),
                     "cuEventCreate"));
    }
  }
  return Status::OK();
}

void DestroyCudaGraph(CudaExecutionState* state) {
  for (auto& entry : state->graphs) {
    if (entry.graph_exec != nullptr) cuGraphExecDestroy(entry.graph_exec);
    if (entry.graph != nullptr) cuGraphDestroy(entry.graph);
  }
  state->graphs.clear();
}

void DestroyCuda(CudaExecutionState* state) {
  if (state == nullptr) {
    return;
  }
  cuCtxSetCurrent(state->context);
  DestroyCudaGraph(state);
  for (auto& pair : state->events) {
    if (pair.start != nullptr) {
      cuEventDestroy(pair.start);
    }
    if (pair.end != nullptr) {
      cuEventDestroy(pair.end);
    }
  }
  if (state->owned_stream != nullptr) {
    cuStreamDestroy(state->owned_stream);
  }
  if (state->context != nullptr) {
    cuDevicePrimaryCtxRelease(state->device);
  }
  delete state;
}

std::string Hex64(std::uint64_t value) {
  std::ostringstream output;
  output << std::hex << std::setw(16) << std::setfill('0') << value;
  return output.str();
}

std::string TuningKey(const PlanOp& op, const Node& node, const Graph& graph,
                      int device_id) {
  std::ostringstream material;
  material << op.op_type << '|' << device_id;
  auto append_tensors = [&](const char* role,
                            const std::vector<std::string>& names) {
    for (const auto& name : names) {
      const auto* value = graph.FindValue(name);
      material << '|' << role << ':' << name;
      if (value != nullptr) {
        material << ':' << static_cast<int>(value->tensor.dtype) << ':'
                 << value->tensor.layout;
        for (const auto dimension : value->tensor.shape) {
          material << ':' << dimension;
        }
      }
    }
  };
  append_tensors("input", op.inputs);
  append_tensors("output", op.outputs);
  std::vector<std::string> attribute_names;
  attribute_names.reserve(node.attributes().size());
  for (const auto& entry : node.attributes()) {
    attribute_names.push_back(entry.first);
  }
  std::sort(attribute_names.begin(), attribute_names.end());
  for (const auto& name : attribute_names) {
    const auto& attribute = node.attributes().at(name);
    material << "|attribute:" << name << ':' << attribute.index();
    std::visit([&](const auto& value) {
      using T = std::decay_t<decltype(value)>;
      if constexpr (std::is_same_v<T, std::vector<int64_t>> ||
                    std::is_same_v<T, std::vector<double>>) {
        for (const auto item : value) material << ':' << item;
      } else {
        material << ':' << value;
      }
    }, attribute);
  }
  for (const auto& candidate : op.candidates) {
    material << '|' << candidate.candidate_id;
    for (const auto& step : candidate.steps) {
      if (const auto* invocation = std::get_if<KernelInvocation>(&step)) {
        const auto& artifact = invocation->artifact;
        material << "|pack:" << artifact.pack_name << ':'
                 << artifact.pack_version << "|kernel:"
                 << artifact.kernel_id << ':' << artifact.kernel_version
                 << ':' << artifact.variant_id << "|abi:"
                 << artifact.abi_schema_version << "|generator:"
                 << artifact.generator_version << "|triton:"
                 << artifact.triton_version << "|ptx:" << artifact.sha256;
      } else {
        material << "|view";
      }
    }
  }
  CUdevice device{};
  CUuuid uuid{};
  int driver = 0;
  int major = 0;
  int minor = 0;
  if (cuDeviceGet(&device, device_id) == CUDA_SUCCESS) {
    cuDeviceGetUuid(&uuid, device);
    cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
                         device);
    cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
                         device);
    material << "|sm:" << major << minor << "|uuid:";
    for (const char byte : uuid.bytes) {
      material << std::hex << static_cast<int>(static_cast<unsigned char>(byte));
    }
  }
  cuDriverGetVersion(&driver);
  material << "|driver:" << driver;
  std::uint64_t hash = 1469598103934665603ULL;
  for (const unsigned char byte : material.str()) {
    hash ^= byte;
    hash *= 1099511628211ULL;
  }
  return Hex64(hash);
}

std::optional<std::string> ReadTuningChoice(
    const std::filesystem::path& path) {
  std::ifstream input(path);
  if (!input) return std::nullopt;
  std::ostringstream contents; contents << input.rdbuf();
  const std::string marker = "\"candidate_id\": \"";
  const auto begin = contents.str().find(marker);
  if (begin == std::string::npos) return std::nullopt;
  const auto value_begin = begin + marker.size();
  const auto end = contents.str().find('"', value_begin);
  if (end == std::string::npos) return std::nullopt;
  return contents.str().substr(value_begin, end - value_begin);
}

Status WriteTuningChoice(const std::filesystem::path& path,
                         const std::string& key,
                         const std::string& candidate_id,
                         double median_ms) {
  std::error_code error;
  std::filesystem::create_directories(path.parent_path(), error);
  if (error) return Status::RuntimeError("Cannot create tuning cache directory: " + error.message());
  const auto temporary = path.string() + ".tmp";
  {
    std::ofstream output(temporary, std::ios::trunc);
    output << "{\n  \"schema_version\": 1,\n"
           << "  \"tuning_key\": \"" << key << "\",\n"
           << "  \"candidate_id\": \"" << candidate_id << "\",\n"
           << "  \"median_ms\": " << median_ms << "\n}\n";
    if (!output) return Status::RuntimeError("Cannot write tuning cache");
  }
  std::filesystem::rename(temporary, path, error);
  if (error) {
    std::filesystem::remove(path, error);
    error.clear();
    std::filesystem::rename(temporary, path, error);
  }
  return error ? Status::RuntimeError("Cannot publish tuning cache: " + error.message())
               : Status::OK();
}
#endif

}  // namespace

class EngineState {
 public:
  explicit EngineState(ExecutablePlan executable_plan)
      : plan(std::move(executable_plan)) {}

  Status InitializeConstants() {
    std::call_once(constants_once, [&]() { constants_status = UploadConstants(); });
    return constants_status;
  }

  Status InitializeModuleCache() {
    if (plan.backend_options().device.type != DeviceType::kCUDA) {
      return Status::OK();
    }
    std::call_once(module_cache_once, [&]() {
      module_cache_status = CreateCudaModuleCache(
          plan.backend_options().device.id, &module_cache);
    });
    return module_cache_status;
  }

  ExecutablePlan plan;
  std::unordered_map<std::string, Tensor> device_constants;
  std::shared_ptr<CudaModuleCache> module_cache;

 private:
  Status UploadConstants() {
    for (const auto& entry : plan.constants()) {
      const auto* value = plan.graph().FindValue(entry.first);
      if (value == nullptr || !value->is_constant) {
        return Status::Internal("Plan constant is missing from Graph: " +
                                entry.first);
      }
      if (plan.backend_options().device.type == DeviceType::kCPU) {
        device_constants.emplace(entry.first, entry.second);
        continue;
      }
      std::shared_ptr<Buffer> buffer;
      CUTRITON_RETURN_IF_ERROR(Buffer::AllocateCuda(
          value->tensor.ByteSize(), plan.backend_options().device.id, &buffer));
      const auto& source = entry.second;
      const auto* source_bytes = static_cast<const std::uint8_t*>(
          source.buffer()->data()) + source.byte_offset();
      CUTRITON_RETURN_IF_ERROR(
          buffer->CopyFromHost(source_bytes, value->tensor.ByteSize()));
      device_constants.emplace(entry.first,
                               Tensor(value->tensor, std::move(buffer)));
    }
    return Status::OK();
  }

  std::once_flag constants_once;
  Status constants_status;
  std::once_flag module_cache_once;
  Status module_cache_status;
};

Engine::Engine(ExecutablePlan plan)
    : state_(std::make_shared<EngineState>(std::move(plan))) {}

const ExecutablePlan& Engine::plan() const { return state_->plan; }

std::unique_ptr<ExecutionContext> Engine::CreateExecutionContext() const {
  return std::make_unique<ExecutionContext>(state_);
}

ExecutionContext::ExecutionContext(std::shared_ptr<EngineState> state)
    : state_(std::move(state)) {
  if (state_ != nullptr && state_->plan.profile_plans().size() == 1) {
    selected_profile_ = 0;
    const auto& profile_plan = state_->plan.profile_plans().front();
    resolved_graph_ = profile_plan.opt_graph;
    shapes_resolved_ = true;
    for (const auto& entry : profile_plan.profile.inputs) {
      input_shapes_[entry.first] = entry.second.opt;
    }
  }
}

ExecutionContext::~ExecutionContext() {
  (void)Synchronize();
#if CUTRITON_ENABLE_CUDA
  DestroyCuda(static_cast<CudaExecutionState*>(cuda_state_));
#endif
}

Status ExecutionContext::BindInput(const std::string& name, Tensor tensor) {
  if (state_ == nullptr) {
    return Status::InvalidArgument("ExecutionContext has no EngineState");
  }
  if (run_pending_) {
    return Status::RuntimeError("Cannot bind a Tensor while a run is pending");
  }
  if (!state_->plan.profile_plans().empty() && !shapes_resolved_) {
    return Status::RuntimeError("ResolveShapes must be called before binding");
  }
  const auto& graph = ActiveGraph();
  if (!Contains(graph.inputs(), name)) {
    return Status::NotFound("Name is not a graph input: " + name);
  }
  const auto* value = graph.FindValue(name);
  if (value == nullptr) {
    return Status::NotFound("Graph input description is missing: " + name);
  }
  CUTRITON_RETURN_IF_ERROR(ValidateBoundTensor(name, value->tensor, tensor));
  tensors_[name] = std::move(tensor);
  tensors_prepared_ = false;
  return Status::OK();
}

Status ExecutionContext::BindOutput(const std::string& name, Tensor tensor) {
  if (state_ == nullptr) {
    return Status::InvalidArgument("ExecutionContext has no EngineState");
  }
  if (run_pending_) {
    return Status::RuntimeError("Cannot bind a Tensor while a run is pending");
  }
  if (!state_->plan.profile_plans().empty() && !shapes_resolved_) {
    return Status::RuntimeError("ResolveShapes must be called before binding");
  }
  const auto& graph = ActiveGraph();
  if (!Contains(graph.outputs(), name)) {
    return Status::NotFound("Name is not a graph output: " + name);
  }
  const auto* value = graph.FindValue(name);
  if (value == nullptr) {
    return Status::NotFound("Graph output description is missing: " + name);
  }
  CUTRITON_RETURN_IF_ERROR(ValidateBoundTensor(name, value->tensor, tensor));
  tensors_[name] = std::move(tensor);
  tensors_prepared_ = false;
  return Status::OK();
}

const Graph& ExecutionContext::ActiveGraph() const {
  return shapes_resolved_ ? resolved_graph_ : state_->plan.graph();
}

const MemoryPlan& ExecutionContext::ActiveMemoryPlan() const {
  if (selected_profile_ >= 0) {
    return state_->plan.profile_plans()
        .at(static_cast<std::size_t>(selected_profile_)).max_memory_plan;
  }
  return state_->plan.memory_plan();
}

Status ExecutionContext::SelectShapeProfile(const std::string& name) {
  if (run_pending_) {
    return Status::RuntimeError("Cannot change profile while a run is pending");
  }
  const auto& profiles = state_->plan.profile_plans();
  const auto found = std::find_if(
      profiles.begin(), profiles.end(), [&](const ProfilePlan& item) {
        return item.profile.name == name;
      });
  if (found == profiles.end()) {
    return Status::NotFound("ShapeProfile is not defined: " + name);
  }
  selected_profile_ = static_cast<int>(std::distance(profiles.begin(), found));
  input_shapes_.clear();
  for (const auto& entry : found->profile.inputs) {
    input_shapes_[entry.first] = entry.second.opt;
  }
  resolved_graph_ = found->opt_graph;
  shapes_resolved_ = true;
  tensors_.clear();
  workspace_.reset();
  tensors_prepared_ = false;
  kernels_.clear();
  kernels_prepared_ = false;
  tuning_prepared_ = false;
  InvalidateCudaGraph();
  return Status::OK();
}

Status ExecutionContext::SetInputShape(const std::string& name,
                                       std::vector<int64_t> shape) {
  if (run_pending_) {
    return Status::RuntimeError("Cannot change shape while a run is pending");
  }
  if (selected_profile_ < 0) {
    return Status::RuntimeError("Select a ShapeProfile before setting shapes");
  }
  const auto& profile = state_->plan.profile_plans()
                            .at(static_cast<std::size_t>(selected_profile_))
                            .profile;
  const auto range_it = profile.inputs.find(name);
  if (range_it == profile.inputs.end()) {
    return Status::NotFound("Name is not a profiled graph input: " + name);
  }
  const auto& range = range_it->second;
  if (shape.size() != range.min.size()) {
    return Status::ShapeError("Input shape rank does not match profile: " + name);
  }
  for (std::size_t axis = 0; axis < shape.size(); ++axis) {
    if (shape[axis] < range.min[axis] || shape[axis] > range.max[axis]) {
      return Status::ShapeError("Input shape is outside profile range: " + name);
    }
  }
  input_shapes_[name] = std::move(shape);
  shapes_resolved_ = false;
  tensors_.clear();
  tensors_prepared_ = false;
  kernels_.clear();
  kernels_prepared_ = false;
  tuning_prepared_ = false;
  return Status::OK();
}

Status ExecutionContext::ResolveShapes() {
  if (selected_profile_ < 0) {
    return Status::RuntimeError("Select a ShapeProfile before resolving shapes");
  }
  const auto& profile_plan = state_->plan.profile_plans().at(
      static_cast<std::size_t>(selected_profile_));
  resolved_graph_ = profile_plan.opt_graph;
  for (const auto& input_name : resolved_graph_.inputs()) {
    const auto shape_it = input_shapes_.find(input_name);
    if (shape_it == input_shapes_.end()) {
      return Status::InvalidArgument("Input shape is not set: " + input_name);
    }
    const auto* value = resolved_graph_.FindValue(input_name);
    TensorDesc desc = value->tensor;
    desc.shape = shape_it->second;
    CUTRITON_RETURN_IF_ERROR(
        resolved_graph_.SetValueDesc(input_name, std::move(desc)));
  }
  auto inference = CreateShapeInferencePass();
  CUTRITON_RETURN_IF_ERROR(inference->Run(&resolved_graph_));
  shapes_resolved_ = true;
  return Status::OK();
}

const TensorDesc* ExecutionContext::GetResolvedTensorDesc(
    const std::string& name) const {
  if (!shapes_resolved_) return nullptr;
  const auto* value = resolved_graph_.FindValue(name);
  return value == nullptr ? nullptr : &value->tensor;
}

std::size_t ExecutionContext::cached_cuda_graph_count() const {
#if CUTRITON_ENABLE_CUDA
  const auto* cuda = static_cast<const CudaExecutionState*>(cuda_state_);
  return cuda == nullptr ? 0 : cuda->graphs.size();
#else
  return 0;
#endif
}

std::size_t ExecutionContext::loaded_cuda_module_count() const {
  return state_ == nullptr ? 0 : LoadedModuleCount(state_->module_cache);
}

Status ExecutionContext::PrepareTensors() {
  if (tensors_prepared_) {
    return Status::OK();
  }
  if (state_ == nullptr) {
    return Status::InvalidArgument("ExecutionContext has no EngineState");
  }
  CUTRITON_RETURN_IF_ERROR(state_->InitializeConstants());
  for (const auto& entry : state_->device_constants) {
    tensors_[entry.first] = entry.second;
  }
  const auto& memory_plan = ActiveMemoryPlan();
  if (memory_plan.workspace_size_bytes != 0 &&
      (workspace_ == nullptr ||
       workspace_->size_bytes() < memory_plan.workspace_size_bytes)) {
    if (state_->plan.backend_options().device.type == DeviceType::kCUDA) {
      CUTRITON_RETURN_IF_ERROR(Buffer::AllocateCuda(
          memory_plan.workspace_size_bytes,
          state_->plan.backend_options().device.id, &workspace_));
    } else {
      workspace_ = Buffer::AllocateHost(memory_plan.workspace_size_bytes);
    }
  }
  for (const auto& allocation : memory_plan.allocations) {
    const auto* value = ActiveGraph().FindValue(allocation.value_name);
    if (value == nullptr) {
      const ValueDesc* like_value = nullptr;
      for (const auto& op : state_->plan.ops()) {
        for (const auto& candidate : op.candidates) {
          const auto found = std::find_if(
              candidate.temporaries.begin(), candidate.temporaries.end(),
              [&](const CandidateTemporary& temporary) {
                return temporary.name == allocation.value_name;
              });
          if (found != candidate.temporaries.end()) {
            like_value = ActiveGraph().FindValue(found->like_value);
            break;
          }
        }
        if (like_value != nullptr) break;
      }
      if (like_value == nullptr) {
        return Status::Internal("Allocation references an unknown Value: " +
                                allocation.value_name);
      }
      tensors_[allocation.value_name] = Tensor(
          like_value->tensor, workspace_, allocation.offset);
      continue;
    }
    if (!allocation.alias_of.empty()) {
      const auto source = tensors_.find(allocation.alias_of);
      if (source == tensors_.end()) {
        return Status::Internal("Allocation alias source Tensor is missing: " +
                                allocation.alias_of);
      }
      if (value->tensor.ByteSize() != source->second.desc().ByteSize()) {
        return Status::ShapeError("Allocation alias changes byte size: " +
                                  allocation.value_name);
      }
      tensors_[allocation.value_name] =
          Tensor(value->tensor, source->second.buffer(),
                 source->second.byte_offset());
      continue;
    }
    if (workspace_ == nullptr ||
        allocation.offset > workspace_->size_bytes() ||
        allocation.size_bytes > workspace_->size_bytes() - allocation.offset) {
      return Status::Internal("Allocation exceeds Workspace: " +
                              allocation.value_name);
    }
    tensors_[allocation.value_name] =
        Tensor(value->tensor, workspace_, allocation.offset);
  }
  tensors_prepared_ = true;
  return Status::OK();
}

Status ExecutionContext::PrepareKernels() {
  if (kernels_prepared_) {
    return Status::OK();
  }
  if (state_ == nullptr) {
    return Status::InvalidArgument("ExecutionContext has no EngineState");
  }
  CUTRITON_RETURN_IF_ERROR(RegisterBuiltinBackends());
  CUTRITON_RETURN_IF_ERROR(state_->InitializeModuleCache());
  auto& registry = KernelRegistry::Global();
  if (selected_candidates_.size() != state_->plan.ops().size()) {
    return Status::Internal("Tuning selections are not prepared");
  }
  for (std::size_t op_index = 0; op_index < state_->plan.ops().size(); ++op_index) {
    const auto& op = state_->plan.ops()[op_index];
    auto backend = registry.GetBackend(op.backend_name);
    if (backend == nullptr) {
      return Status::NotFound("Backend is not registered: " + op.backend_name);
    }
    const auto& node = ActiveGraph().nodes().at(
        static_cast<std::size_t>(op.node_id));
    const std::size_t selected = selected_candidates_[op_index];
    if (selected >= op.candidates.size()) {
      return Status::Internal("Selected execution candidate is out of range");
    }
    const auto& candidate = op.candidates[selected];
    std::vector<std::unique_ptr<OpKernel>> group;
    for (const auto& step : candidate.steps) {
      std::unique_ptr<OpKernel> kernel;
      if (std::holds_alternative<KernelInvocation>(step)) {
        CUTRITON_RETURN_IF_ERROR(CreateInvocationKernel(
            std::get<KernelInvocation>(step),
            state_->plan.backend_options().device.id, state_->module_cache,
            &kernel));
      } else {
        CUTRITON_RETURN_IF_ERROR(backend->CreateKernel(
            node, ActiveGraph(), state_->plan.backend_options(), &kernel));
      }
      if (kernel == nullptr) {
        return Status::Internal("Backend returned a null OpKernel");
      }
      group.push_back(std::move(kernel));
    }
    if (group.empty()) {
      std::unique_ptr<OpKernel> kernel;
      CUTRITON_RETURN_IF_ERROR(backend->CreateKernel(
          node, ActiveGraph(), state_->plan.backend_options(), &kernel));
      group.push_back(std::move(kernel));
    }
    kernels_.push_back(std::move(group));
  }
  kernels_prepared_ = true;
  return Status::OK();
}

Status ExecutionContext::PrepareTuning() {
  if (tuning_prepared_) return Status::OK();
  selected_candidates_.clear();
  selected_candidates_.reserve(state_->plan.ops().size());
  for (const auto& op : state_->plan.ops()) {
    selected_candidates_.push_back(op.selected_candidate);
  }
  if (state_->plan.backend_options().device.type != DeviceType::kCUDA ||
      state_->plan.tuning_config().mode == TuningMode::kDisabled) {
    tuning_prepared_ = true;
    return Status::OK();
  }
#if CUTRITON_ENABLE_CUDA
  CUTRITON_RETURN_IF_ERROR(state_->InitializeModuleCache());
  const auto& config = state_->plan.tuning_config();
  for (std::size_t op_index = 0; op_index < state_->plan.ops().size(); ++op_index) {
    const auto& op = state_->plan.ops()[op_index];
    if (op.candidates.size() < 2) continue;
    const auto& node = ActiveGraph().nodes().at(
        static_cast<std::size_t>(op.node_id));
    const std::string key = TuningKey(
        op, node, ActiveGraph(), state_->plan.backend_options().device.id);
    const std::filesystem::path cache_path =
        std::filesystem::path(config.cache_dir) /
        (key + ".json");
    if (config.mode != TuningMode::kForceRetune) {
      if (const auto cached = ReadTuningChoice(cache_path)) {
        const auto found = std::find_if(
            op.candidates.begin(), op.candidates.end(),
            [&](const ExecutionCandidate& item) {
              return item.candidate_id == *cached;
            });
        if (found != op.candidates.end()) {
          selected_candidates_[op_index] = static_cast<std::size_t>(
              std::distance(op.candidates.begin(), found));
          continue;
        }
      }
      if (config.mode == TuningMode::kUseCache && selected_profile_ >= 0) {
        const auto& opt_graph = state_->plan.profile_plans()
                                    .at(static_cast<std::size_t>(selected_profile_))
                                    .opt_graph;
        const auto& opt_node = opt_graph.nodes().at(
            static_cast<std::size_t>(op.node_id));
        const std::string opt_key = TuningKey(
            op, opt_node, opt_graph,
            state_->plan.backend_options().device.id);
        const auto opt_choice = ReadTuningChoice(
            std::filesystem::path(config.cache_dir) / (opt_key + ".json"));
        if (opt_choice) {
          const auto found = std::find_if(
              op.candidates.begin(), op.candidates.end(),
              [&](const ExecutionCandidate& item) {
                return item.candidate_id == *opt_choice;
              });
          if (found != op.candidates.end()) {
            selected_candidates_[op_index] = static_cast<std::size_t>(
                std::distance(op.candidates.begin(), found));
            continue;
          }
        }
      }
    }
    if (config.mode == TuningMode::kUseCache) continue;

    CUstream stream{};
    CUevent start{};
    CUevent end{};
    CUTRITON_RETURN_IF_ERROR(CudaStatus(
        cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING), "cuStreamCreate"));
    CUTRITON_RETURN_IF_ERROR(CudaStatus(cuEventCreate(&start, CU_EVENT_DEFAULT),
                                       "cuEventCreate"));
    CUTRITON_RETURN_IF_ERROR(CudaStatus(cuEventCreate(&end, CU_EVENT_DEFAULT),
                                       "cuEventCreate"));
    double best_ms = std::numeric_limits<double>::infinity();
    std::size_t best_index = op.selected_candidate;
    std::vector<float> reference_output;
    for (std::size_t candidate_index = 0;
         candidate_index < op.candidates.size(); ++candidate_index) {
      const auto& candidate = op.candidates[candidate_index];
      if (candidate.steps.empty()) {
        continue;
      }
      std::vector<std::unique_ptr<OpKernel>> candidate_kernels;
      for (const auto& step : candidate.steps) {
        if (!std::holds_alternative<KernelInvocation>(step)) continue;
        std::unique_ptr<OpKernel> kernel;
        CUTRITON_RETURN_IF_ERROR(CreateInvocationKernel(
            std::get<KernelInvocation>(step),
            state_->plan.backend_options().device.id, state_->module_cache,
            &kernel));
        candidate_kernels.push_back(std::move(kernel));
      }
      if (candidate_kernels.empty()) continue;
      KernelContext kernel_context{&node, &tensors_, stream, nullptr};
      for (int iteration = 0; iteration < config.warmup_iterations; ++iteration) {
        for (const auto& kernel : candidate_kernels) {
          CUTRITON_RETURN_IF_ERROR(kernel->Compute(&kernel_context));
        }
      }
      CUTRITON_RETURN_IF_ERROR(CudaStatus(cuStreamSynchronize(stream),
                                         "cuStreamSynchronize"));
      const Tensor& candidate_output = tensors_.at(op.outputs.front());
      std::vector<float> host_output(
          static_cast<std::size_t>(candidate_output.desc().NumElements()));
      CUTRITON_RETURN_IF_ERROR(candidate_output.buffer()->CopyToHost(
          host_output.data(), candidate_output.desc().ByteSize(),
          candidate_output.byte_offset()));
      if (reference_output.empty()) {
        reference_output = host_output;
      } else {
        bool matches = true;
        for (std::size_t element = 0; element < host_output.size(); ++element) {
          const float expected = reference_output[element];
          const float actual = host_output[element];
          const float tolerance = 1e-4F + 1e-4F * std::abs(expected);
          if (!std::isfinite(actual) || std::abs(actual - expected) > tolerance) {
            matches = false;
            break;
          }
        }
        if (!matches) continue;
      }
      std::vector<float> samples;
      samples.reserve(static_cast<std::size_t>(config.measurement_iterations));
      for (int iteration = 0; iteration < config.measurement_iterations;
           ++iteration) {
        CUTRITON_RETURN_IF_ERROR(CudaStatus(cuEventRecord(start, stream),
                                           "cuEventRecord"));
        for (const auto& kernel : candidate_kernels) {
          CUTRITON_RETURN_IF_ERROR(kernel->Compute(&kernel_context));
        }
        CUTRITON_RETURN_IF_ERROR(CudaStatus(cuEventRecord(end, stream),
                                           "cuEventRecord"));
        CUTRITON_RETURN_IF_ERROR(CudaStatus(cuEventSynchronize(end),
                                           "cuEventSynchronize"));
        float elapsed = 0.0F;
        CUTRITON_RETURN_IF_ERROR(CudaStatus(
            cuEventElapsedTime(&elapsed, start, end), "cuEventElapsedTime"));
        samples.push_back(elapsed);
      }
      std::sort(samples.begin(), samples.end());
      const double median = samples[samples.size() / 2];
      if (median < best_ms) {
        best_ms = median;
        best_index = candidate_index;
      }
    }
    cuEventDestroy(start);
    cuEventDestroy(end);
    cuStreamDestroy(stream);
    if (!std::isfinite(best_ms)) {
      return Status::RuntimeError("No numerically valid tuning candidate for " +
                                  op.node_name);
    }
    selected_candidates_[op_index] = best_index;
    CUTRITON_RETURN_IF_ERROR(WriteTuningChoice(
        cache_path, key, op.candidates[best_index].candidate_id, best_ms));
  }
  tuning_prepared_ = true;
  return Status::OK();
#else
  return Status::Unsupported("CUTriton was built without CUDA support");
#endif
}

Status ExecutionContext::ValidateBindings() const {
  const auto& graph = ActiveGraph();
  for (const auto& name : graph.inputs()) {
    if (tensors_.find(name) == tensors_.end()) {
      return Status::InvalidArgument("Graph input is not bound: " + name);
    }
  }
  for (const auto& name : graph.outputs()) {
    if (tensors_.find(name) == tensors_.end()) {
      return Status::InvalidArgument("Graph output is not bound: " + name);
    }
  }
  for (const auto& node : graph.nodes()) {
    for (const auto& name : node.inputs()) {
      if (tensors_.find(name) == tensors_.end()) {
        return Status::InvalidArgument("Node input Tensor is missing: " + name);
      }
    }
    for (const auto& name : node.outputs()) {
      if (tensors_.find(name) == tensors_.end()) {
        return Status::InvalidArgument("Node output Tensor is missing: " + name);
      }
    }
  }
  return Status::OK();
}

Status ExecutionContext::Run() {
  CUTRITON_RETURN_IF_ERROR(RunAsync(nullptr));
  return Synchronize();
}

Status ExecutionContext::RunAsync(void* stream) {
  if (run_pending_) {
    return Status::RuntimeError("A run is already pending");
  }
  CUTRITON_RETURN_IF_ERROR(PrepareTensors());
  CUTRITON_RETURN_IF_ERROR(ValidateBindings());
  CUTRITON_RETURN_IF_ERROR(PrepareTuning());
  CUTRITON_RETURN_IF_ERROR(PrepareKernels());

  const bool use_cuda =
      state_->plan.backend_options().device.type == DeviceType::kCUDA;
  if (!use_cuda) {
    profiler_.Clear();
    for (std::size_t i = 0; i < kernels_.size(); ++i) {
      const auto& op = state_->plan.ops()[i];
      const auto& node = ActiveGraph().nodes().at(
          static_cast<std::size_t>(op.node_id));
      KernelContext context{&node, &tensors_, stream,
                            state_->plan.enable_profiling() ? &profiler_ : nullptr};
      for (const auto& kernel : kernels_[i]) {
        CUTRITON_RETURN_IF_ERROR(kernel->Compute(&context));
      }
    }
    run_pending_ = true;
    return Status::OK();
  }

#if CUTRITON_ENABLE_CUDA
  auto* cuda = static_cast<CudaExecutionState*>(cuda_state_);
  if (cuda == nullptr) {
    cuda = new CudaExecutionState();
    Status status = InitializeCuda(cuda, state_->plan.backend_options().device.id,
                                   kernels_.size(),
                                   state_->plan.enable_profiling());
    if (!status.ok()) {
      DestroyCuda(cuda);
      return status;
    }
    cuda_state_ = cuda;
  }
  CUTRITON_RETURN_IF_ERROR(
      CudaStatus(cuCtxSetCurrent(cuda->context), "cuCtxSetCurrent"));
  cuda->active_stream =
      stream == nullptr ? cuda->owned_stream : reinterpret_cast<CUstream>(stream);

  auto enqueue_sequence = [&]() -> Status {
    auto record_event = [&](CUevent event) -> Status {
      if (state_->plan.enable_cuda_graph()) {
        return CudaStatus(
            cuEventRecordWithFlags(event, cuda->active_stream,
                                   CU_EVENT_RECORD_EXTERNAL),
            "cuEventRecordWithFlags");
      }
      return CudaStatus(cuEventRecord(event, cuda->active_stream),
                        "cuEventRecord");
    };
    for (std::size_t i = 0; i < kernels_.size(); ++i) {
      const auto& op = state_->plan.ops()[i];
      const auto& node = ActiveGraph().nodes().at(
          static_cast<std::size_t>(op.node_id));
      auto& events = cuda->events[i];
      events.timed = state_->plan.enable_profiling() &&
                     node.op_type() != "Flatten";
      if (events.timed) {
        CUTRITON_RETURN_IF_ERROR(record_event(events.start));
      }
      KernelContext context{&node, &tensors_, cuda->active_stream, nullptr};
      for (const auto& kernel : kernels_[i]) {
        CUTRITON_RETURN_IF_ERROR(kernel->Compute(&context));
      }
      if (events.timed) {
        CUTRITON_RETURN_IF_ERROR(record_event(events.end));
      }
    }
    return Status::OK();
  };

  if (state_->plan.enable_cuda_graph()) {
    std::ostringstream graph_key_stream;
    graph_key_stream << "profile:" << selected_profile_ << "|workspace:"
                     << (workspace_ == nullptr ? nullptr : workspace_->data());
    for (const auto& name : ActiveGraph().inputs()) {
      const auto& tensor = tensors_.at(name);
      graph_key_stream << "|in:" << name << ':' << tensor.buffer()->data()
                       << ':' << tensor.byte_offset();
      for (const auto dimension : tensor.desc().shape) {
        graph_key_stream << ':' << dimension;
      }
    }
    for (const auto& name : ActiveGraph().outputs()) {
      const auto& tensor = tensors_.at(name);
      graph_key_stream << "|out:" << name << ':' << tensor.buffer()->data()
                       << ':' << tensor.byte_offset();
    }
    for (std::size_t index = 0; index < selected_candidates_.size(); ++index) {
      graph_key_stream << "|candidate:"
                       << state_->plan.ops()[index]
                              .candidates[selected_candidates_[index]]
                              .candidate_id;
    }
    const std::string graph_key = graph_key_stream.str();
    auto graph_entry = std::find_if(
        cuda->graphs.begin(), cuda->graphs.end(),
        [&](const CudaGraphEntry& entry) { return entry.key == graph_key; });
    if (graph_entry == cuda->graphs.end()) {
      CudaGraphEntry entry;
      entry.key = graph_key;
      CUTRITON_RETURN_IF_ERROR(CudaStatus(
          cuStreamBeginCapture(cuda->active_stream, CU_STREAM_CAPTURE_MODE_GLOBAL),
          "cuStreamBeginCapture"));
      Status sequence_status = enqueue_sequence();
      Status end_status = CudaStatus(
          cuStreamEndCapture(cuda->active_stream, &entry.graph),
          "cuStreamEndCapture");
      if (!sequence_status.ok()) {
        if (entry.graph != nullptr) cuGraphDestroy(entry.graph);
        return sequence_status;
      }
      if (!end_status.ok()) {
        if (entry.graph != nullptr) cuGraphDestroy(entry.graph);
        return end_status;
      }
      CUTRITON_RETURN_IF_ERROR(CudaStatus(
          cuGraphInstantiate(&entry.graph_exec, entry.graph, 0),
          "cuGraphInstantiate"));
      entry.last_used = ++cuda->graph_clock;
      const std::size_t capacity =
          std::max<std::size_t>(1, state_->plan.cuda_graph_cache_capacity());
      if (cuda->graphs.size() >= capacity) {
        auto oldest = std::min_element(
            cuda->graphs.begin(), cuda->graphs.end(),
            [](const CudaGraphEntry& left, const CudaGraphEntry& right) {
              return left.last_used < right.last_used;
            });
        cuGraphExecDestroy(oldest->graph_exec);
        cuGraphDestroy(oldest->graph);
        cuda->graphs.erase(oldest);
      }
      cuda->graphs.push_back(std::move(entry));
      graph_entry = std::prev(cuda->graphs.end());
    }
    graph_entry->last_used = ++cuda->graph_clock;
    CUTRITON_RETURN_IF_ERROR(CudaStatus(
        cuGraphLaunch(graph_entry->graph_exec, cuda->active_stream),
        "cuGraphLaunch"));
  } else {
    CUTRITON_RETURN_IF_ERROR(enqueue_sequence());
  }
  run_pending_ = true;
  return Status::OK();
#else
  (void)stream;
  return Status::Unsupported("CUTriton was built without CUDA support");
#endif
}

Status ExecutionContext::Synchronize() {
  if (!run_pending_) {
    return Status::OK();
  }
  if (state_->plan.backend_options().device.type != DeviceType::kCUDA) {
    run_pending_ = false;
    return Status::OK();
  }
#if CUTRITON_ENABLE_CUDA
  auto* cuda = static_cast<CudaExecutionState*>(cuda_state_);
  CUTRITON_RETURN_IF_ERROR(
      CudaStatus(cuCtxSetCurrent(cuda->context), "cuCtxSetCurrent"));
  Status status = CudaStatus(cuStreamSynchronize(cuda->active_stream),
                             "cuStreamSynchronize");
  if (!status.ok()) {
    run_pending_ = false;
    return status;
  }
  profiler_.Clear();
  if (state_->plan.enable_profiling()) {
    for (std::size_t i = 0; i < state_->plan.ops().size(); ++i) {
      const auto& op = state_->plan.ops()[i];
      const auto& pair = cuda->events[i];
      float duration_ms = 0.0F;
      if (pair.timed) {
        CUTRITON_RETURN_IF_ERROR(CudaStatus(
            cuEventElapsedTime(&duration_ms, pair.start, pair.end),
            "cuEventElapsedTime"));
      }
      profiler_.Record(op.node_name, op.backend_name,
                       static_cast<double>(duration_ms));
    }
  }
  run_pending_ = false;
  return Status::OK();
#else
  return Status::Unsupported("CUTriton was built without CUDA support");
#endif
}

void ExecutionContext::InvalidateCudaGraph() {
#if CUTRITON_ENABLE_CUDA
  auto* cuda = static_cast<CudaExecutionState*>(cuda_state_);
  if (cuda != nullptr) {
    cuCtxSetCurrent(cuda->context);
    DestroyCudaGraph(cuda);
  }
#endif
}

}  // namespace cutriton
