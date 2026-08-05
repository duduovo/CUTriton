#include "cutriton/runtime/engine.h"
#include "cutriton/core/buffer.h"

#include <algorithm>
#include <cstdint>
#include <mutex>
#include <utility>

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

struct CudaExecutionState {
  CUdevice device{};
  CUcontext context{nullptr};
  CUstream owned_stream{nullptr};
  CUstream active_stream{nullptr};
  CUgraph graph{nullptr};
  CUgraphExec graph_exec{nullptr};
  bool graph_valid{false};
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
  if (state->graph_exec != nullptr) {
    cuGraphExecDestroy(state->graph_exec);
    state->graph_exec = nullptr;
  }
  if (state->graph != nullptr) {
    cuGraphDestroy(state->graph);
    state->graph = nullptr;
  }
  state->graph_valid = false;
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

  ExecutablePlan plan;
  std::unordered_map<std::string, Tensor> device_constants;

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
};

Engine::Engine(ExecutablePlan plan)
    : state_(std::make_shared<EngineState>(std::move(plan))) {}

const ExecutablePlan& Engine::plan() const { return state_->plan; }

std::unique_ptr<ExecutionContext> Engine::CreateExecutionContext() const {
  return std::make_unique<ExecutionContext>(state_);
}

ExecutionContext::ExecutionContext(std::shared_ptr<EngineState> state)
    : state_(std::move(state)) {}

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
  const auto& graph = state_->plan.graph();
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
  InvalidateCudaGraph();
  return Status::OK();
}

Status ExecutionContext::BindOutput(const std::string& name, Tensor tensor) {
  if (state_ == nullptr) {
    return Status::InvalidArgument("ExecutionContext has no EngineState");
  }
  if (run_pending_) {
    return Status::RuntimeError("Cannot bind a Tensor while a run is pending");
  }
  const auto& graph = state_->plan.graph();
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
  InvalidateCudaGraph();
  return Status::OK();
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
  const auto& memory_plan = state_->plan.memory_plan();
  if (memory_plan.workspace_size_bytes != 0) {
    if (state_->plan.backend_options().device.type == DeviceType::kCUDA) {
      CUTRITON_RETURN_IF_ERROR(Buffer::AllocateCuda(
          memory_plan.workspace_size_bytes,
          state_->plan.backend_options().device.id, &workspace_));
    } else {
      workspace_ = Buffer::AllocateHost(memory_plan.workspace_size_bytes);
    }
  }
  for (const auto& allocation : memory_plan.allocations) {
    const auto* value = state_->plan.graph().FindValue(allocation.value_name);
    if (value == nullptr) {
      return Status::Internal("Allocation references an unknown Value: " +
                              allocation.value_name);
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
  auto& registry = KernelRegistry::Global();
  for (const auto& op : state_->plan.ops()) {
    auto backend = registry.GetBackend(op.backend_name);
    if (backend == nullptr) {
      return Status::NotFound("Backend is not registered: " + op.backend_name);
    }
    const auto& node = state_->plan.graph().nodes().at(
        static_cast<std::size_t>(op.node_id));
    std::unique_ptr<OpKernel> kernel;
    CUTRITON_RETURN_IF_ERROR(backend->CreateKernel(
        node, state_->plan.graph(), state_->plan.backend_options(), &kernel));
    if (kernel == nullptr) {
      return Status::Internal("Backend returned a null OpKernel");
    }
    kernels_.push_back(std::move(kernel));
  }
  kernels_prepared_ = true;
  return Status::OK();
}

Status ExecutionContext::ValidateBindings() const {
  const auto& graph = state_->plan.graph();
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
  CUTRITON_RETURN_IF_ERROR(PrepareKernels());
  CUTRITON_RETURN_IF_ERROR(ValidateBindings());

  const bool use_cuda =
      state_->plan.backend_options().device.type == DeviceType::kCUDA;
  if (!use_cuda) {
    profiler_.Clear();
    for (std::size_t i = 0; i < kernels_.size(); ++i) {
      const auto& op = state_->plan.ops()[i];
      const auto& node = state_->plan.graph().nodes().at(
          static_cast<std::size_t>(op.node_id));
      KernelContext context{&node, &tensors_, stream,
                            state_->plan.enable_profiling() ? &profiler_ : nullptr};
      CUTRITON_RETURN_IF_ERROR(kernels_[i]->Compute(&context));
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
    for (std::size_t i = 0; i < kernels_.size(); ++i) {
      const auto& op = state_->plan.ops()[i];
      const auto& node = state_->plan.graph().nodes().at(
          static_cast<std::size_t>(op.node_id));
      auto& events = cuda->events[i];
      events.timed = state_->plan.enable_profiling() &&
                     node.op_type() != "Flatten";
      if (events.timed) {
        CUTRITON_RETURN_IF_ERROR(CudaStatus(
            cuEventRecord(events.start, cuda->active_stream), "cuEventRecord"));
      }
      KernelContext context{&node, &tensors_, cuda->active_stream, nullptr};
      CUTRITON_RETURN_IF_ERROR(kernels_[i]->Compute(&context));
      if (events.timed) {
        CUTRITON_RETURN_IF_ERROR(CudaStatus(
            cuEventRecord(events.end, cuda->active_stream), "cuEventRecord"));
      }
    }
    return Status::OK();
  };

  if (state_->plan.enable_cuda_graph()) {
    if (!cuda->graph_valid) {
      DestroyCudaGraph(cuda);
      CUTRITON_RETURN_IF_ERROR(CudaStatus(
          cuStreamBeginCapture(cuda->active_stream, CU_STREAM_CAPTURE_MODE_GLOBAL),
          "cuStreamBeginCapture"));
      Status sequence_status = enqueue_sequence();
      Status end_status = CudaStatus(
          cuStreamEndCapture(cuda->active_stream, &cuda->graph),
          "cuStreamEndCapture");
      if (!sequence_status.ok()) {
        DestroyCudaGraph(cuda);
        return sequence_status;
      }
      if (!end_status.ok()) {
        DestroyCudaGraph(cuda);
        return end_status;
      }
      CUTRITON_RETURN_IF_ERROR(CudaStatus(
          cuGraphInstantiate(&cuda->graph_exec, cuda->graph, 0),
          "cuGraphInstantiate"));
      cuda->graph_valid = true;
    }
    CUTRITON_RETURN_IF_ERROR(CudaStatus(
        cuGraphLaunch(cuda->graph_exec, cuda->active_stream), "cuGraphLaunch"));
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
