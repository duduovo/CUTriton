#include "cutriton/backend/backend.h"

#include <array>
#include <cstdint>
#include <fstream>
#include <limits>
#include <mutex>
#include <sstream>
#include <unordered_map>
#include <utility>

#include "cutriton/core/buffer.h"

#ifndef CUTRITON_ENABLE_CUDA
#define CUTRITON_ENABLE_CUDA 0
#endif

#if CUTRITON_ENABLE_CUDA
#include <cuda.h>
#endif

namespace cutriton {

#if CUTRITON_ENABLE_CUDA
namespace {

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

CUdeviceptr TensorPointer(const Tensor& tensor) {
  return static_cast<CUdeviceptr>(
             reinterpret_cast<std::uintptr_t>(tensor.buffer()->data())) +
         tensor.byte_offset();
}

}  // namespace
#endif

class CudaModuleCache {
 public:
  explicit CudaModuleCache(int device_id) : device_id_(device_id) {}
  ~CudaModuleCache() {
#if CUTRITON_ENABLE_CUDA
    std::lock_guard<std::mutex> lock(mutex_);
    if (context_ != nullptr) cuCtxSetCurrent(context_);
    for (const auto& entry : modules_) cuModuleUnload(entry.second);
    if (context_ != nullptr) cuDevicePrimaryCtxRelease(device_);
#endif
  }

#if CUTRITON_ENABLE_CUDA
  Status Initialize() {
    CUTRITON_RETURN_IF_ERROR(CudaStatus(cuInit(0), "cuInit"));
    CUTRITON_RETURN_IF_ERROR(CudaStatus(cuDeviceGet(&device_, device_id_), "cuDeviceGet"));
    CUTRITON_RETURN_IF_ERROR(CudaStatus(
        cuDevicePrimaryCtxRetain(&context_, device_), "cuDevicePrimaryCtxRetain"));
    return Status::OK();
  }

  Status GetFunction(const KernelArtifact& artifact, CUfunction* function) {
    if (function == nullptr) return Status::InvalidArgument("CUfunction output is null");
    std::lock_guard<std::mutex> lock(mutex_);
    CUTRITON_RETURN_IF_ERROR(CudaStatus(cuCtxSetCurrent(context_), "cuCtxSetCurrent"));
    const std::string module_key = artifact.sha256;
    auto module_it = modules_.find(module_key);
    if (module_it == modules_.end()) {
      std::ifstream input(artifact.binary_path, std::ios::binary);
      if (!input) return Status::NotFound("Kernel binary not found: " + artifact.binary_path.string());
      std::ostringstream contents; contents << input.rdbuf();
      CUmodule module{};
      CUTRITON_RETURN_IF_ERROR(CudaStatus(
          cuModuleLoadDataEx(&module, contents.str().c_str(), 0, nullptr, nullptr),
          "cuModuleLoadDataEx"));
      module_it = modules_.emplace(module_key, module).first;
    }
    const std::string function_key = module_key + ":" + artifact.symbol;
    const auto function_it = functions_.find(function_key);
    if (function_it != functions_.end()) {
      *function = function_it->second;
      return Status::OK();
    }
    CUfunction loaded{};
    CUTRITON_RETURN_IF_ERROR(CudaStatus(
        cuModuleGetFunction(&loaded, module_it->second, artifact.symbol.c_str()),
        "cuModuleGetFunction"));
    functions_.emplace(function_key, loaded);
    *function = loaded;
    return Status::OK();
  }
#endif

  std::size_t size() const {
    std::lock_guard<std::mutex> lock(mutex_);
#if CUTRITON_ENABLE_CUDA
    return modules_.size();
#else
    return 0;
#endif
  }

 private:
  int device_id_{0};
  mutable std::mutex mutex_;
#if CUTRITON_ENABLE_CUDA
  CUdevice device_{};
  CUcontext context_{};
  std::unordered_map<std::string, CUmodule> modules_;
  std::unordered_map<std::string, CUfunction> functions_;
#endif
};

#if CUTRITON_ENABLE_CUDA
union BoundArgument {
  CUdeviceptr pointer;
  int32_t i32;
  float fp32;
};

class ArgumentBinder {
 public:
  Status Bind(const KernelInvocation& invocation, KernelContext* context,
              std::vector<BoundArgument>* storage,
              std::vector<void*>* parameters) const {
    if (context == nullptr || context->node == nullptr ||
        context->tensors == nullptr || storage == nullptr ||
        parameters == nullptr) {
      return Status::InvalidArgument("invalid ArgumentBinder context");
    }
    storage->clear();
    storage->reserve(invocation.artifact.arguments.size());
    const Node& node = *context->node;
    for (const auto& argument : invocation.artifact.arguments) {
      BoundArgument value{};
      const auto& source = argument.source;
      switch (source.kind) {
        case KernelArgumentSourceKind::kInput:
          CUTRITON_RETURN_IF_ERROR(BindTensor(
              invocation.inputs, source.index, context, &value.pointer));
          break;
        case KernelArgumentSourceKind::kOutput:
          CUTRITON_RETURN_IF_ERROR(BindTensor(
              invocation.outputs, source.index, context, &value.pointer));
          break;
        case KernelArgumentSourceKind::kInputDim:
          CUTRITON_RETURN_IF_ERROR(BindDimension(
              invocation.inputs, source.index, source.axis, context,
              &value.i32));
          break;
        case KernelArgumentSourceKind::kOutputDim:
          CUTRITON_RETURN_IF_ERROR(BindDimension(
              invocation.outputs, source.index, source.axis, context,
              &value.i32));
          break;
        case KernelArgumentSourceKind::kInputDimProduct:
          CUTRITON_RETURN_IF_ERROR(BindDimensionProduct(
              invocation.inputs, source.index, source.begin_axis, context,
              &value.i32));
          break;
        case KernelArgumentSourceKind::kOutputNumElements:
          CUTRITON_RETURN_IF_ERROR(BindNumElements(
              invocation.outputs, source.index, context, &value.i32));
          break;
        case KernelArgumentSourceKind::kAttributeInt64:
          value.i32 = static_cast<int32_t>(GetAttributeInt(node, source));
          break;
        case KernelArgumentSourceKind::kAttributeFloat64:
          if (const auto primary =
                  node.GetAttribute<double>(source.attribute_name)) {
            value.fp32 = static_cast<float>(*primary);
          } else if (!source.alternate_attribute_name.empty()) {
            value.fp32 = static_cast<float>(
                node.GetAttribute<double>(source.alternate_attribute_name)
                    .value_or(source.default_float));
          } else {
            value.fp32 = static_cast<float>(source.default_float);
          }
          break;
        case KernelArgumentSourceKind::kLiteralInt32:
          value.i32 = static_cast<int32_t>(source.default_int);
          break;
        case KernelArgumentSourceKind::kRuntimeReserved:
          value.pointer = 0;
          break;
      }
      storage->push_back(value);
    }
    parameters->clear();
    parameters->reserve(storage->size());
    for (auto& value : *storage) parameters->push_back(&value);
    return Status::OK();
  }

 private:
  static Status BindTensor(const std::vector<std::string>& names, int index,
                           KernelContext* context, CUdeviceptr* pointer) {
    if (index < 0 || index >= static_cast<int>(names.size())) {
      return Status::InvalidArgument("Kernel Tensor argument index is invalid");
    }
    const auto it = context->tensors->find(names[static_cast<std::size_t>(index)]);
    if (it == context->tensors->end() || !it->second.defined()) {
      return Status::NotFound("Kernel Tensor argument is missing");
    }
    *pointer = TensorPointer(it->second);
    return Status::OK();
  }

  static Status BindDimension(const std::vector<std::string>& names, int index,
                              int axis, KernelContext* context,
                              int32_t* output) {
    if (index < 0 || index >= static_cast<int>(names.size())) {
      return Status::InvalidArgument("Kernel dimension Tensor index is invalid");
    }
    const auto& shape = context->tensors
                            ->at(names[static_cast<std::size_t>(index)])
                            .desc().shape;
    if (axis < 0 || axis >= static_cast<int>(shape.size()) ||
        shape[static_cast<std::size_t>(axis)] >
            std::numeric_limits<int32_t>::max()) {
      return Status::ShapeError("Kernel dimension is invalid");
    }
    *output = static_cast<int32_t>(shape[static_cast<std::size_t>(axis)]);
    return Status::OK();
  }

  static Status BindDimensionProduct(const std::vector<std::string>& names,
                                     int index, int begin_axis,
                                     KernelContext* context,
                                     int32_t* output) {
    if (index < 0 || index >= static_cast<int>(names.size())) {
      return Status::InvalidArgument("Kernel product Tensor index is invalid");
    }
    const auto& shape = context->tensors
                            ->at(names[static_cast<std::size_t>(index)])
                            .desc().shape;
    if (begin_axis < 0 || begin_axis > static_cast<int>(shape.size())) {
      return Status::ShapeError("Kernel dimension product axis is invalid");
    }
    int64_t value = 1;
    for (std::size_t axis = static_cast<std::size_t>(begin_axis);
         axis < shape.size(); ++axis) {
      value *= shape[axis];
    }
    if (value <= 0 || value > std::numeric_limits<int32_t>::max()) {
      return Status::ShapeError("Kernel dimension product is invalid");
    }
    *output = static_cast<int32_t>(value);
    return Status::OK();
  }

  static Status BindNumElements(const std::vector<std::string>& names,
                                int index, KernelContext* context,
                                int32_t* output) {
    if (index < 0 || index >= static_cast<int>(names.size())) {
      return Status::InvalidArgument("Kernel numel Tensor index is invalid");
    }
    const auto value = context->tensors
                           ->at(names[static_cast<std::size_t>(index)])
                           .desc().NumElements();
    if (value <= 0 || value > std::numeric_limits<int32_t>::max()) {
      return Status::ShapeError("Kernel element count is invalid");
    }
    *output = static_cast<int32_t>(value);
    return Status::OK();
  }

  static int64_t GetAttributeInt(const Node& node,
                                 const KernelArgumentSource& source) {
    if (const auto list =
            node.GetAttribute<std::vector<int64_t>>(source.attribute_name)) {
      if (source.attribute_index >= 0 &&
          source.attribute_index < static_cast<int>(list->size())) {
        return (*list)[static_cast<std::size_t>(source.attribute_index)];
      }
    }
    if (const auto scalar =
            node.GetAttribute<int64_t>(source.attribute_name)) {
      return *scalar;
    }
    return source.default_int;
  }
};

Status EvaluateGridExpression(const GridExpression& expression,
                              const KernelInvocation& invocation,
                              KernelContext* context, int64_t* output) {
  if (context == nullptr || context->tensors == nullptr || output == nullptr) {
    return Status::InvalidArgument("invalid launch-grid evaluation context");
  }
  auto tensor_value = [&](const std::vector<std::string>& names,
                          int index, const Tensor** tensor) -> Status {
    if (index < 0 || index >= static_cast<int>(names.size())) {
      return Status::InvalidArgument("launch-grid Tensor index is invalid");
    }
    const auto found = context->tensors->find(names[static_cast<std::size_t>(index)]);
    if (found == context->tensors->end()) {
      return Status::NotFound("launch-grid Tensor is not bound");
    }
    *tensor = &found->second;
    return Status::OK();
  };
  switch (expression.kind) {
    case GridExpressionKind::kLiteral:
      *output = expression.literal;
      break;
    case GridExpressionKind::kVariantMeta: {
      const auto found = invocation.artifact.launch_meta.find(expression.meta_name);
      if (found == invocation.artifact.launch_meta.end()) {
        return Status::InvalidArgument("launch-grid references missing variant meta: " +
                                       expression.meta_name);
      }
      *output = found->second;
      break;
    }
    case GridExpressionKind::kInputDim:
    case GridExpressionKind::kOutputDim: {
      const auto& names = expression.kind == GridExpressionKind::kInputDim
                              ? invocation.inputs
                              : invocation.outputs;
      const Tensor* tensor = nullptr;
      CUTRITON_RETURN_IF_ERROR(
          tensor_value(names, expression.tensor_index, &tensor));
      const auto& shape = tensor->desc().shape;
      if (expression.axis < 0 ||
          expression.axis >= static_cast<int>(shape.size())) {
        return Status::ShapeError("launch-grid Tensor axis is invalid");
      }
      *output = shape[static_cast<std::size_t>(expression.axis)];
      break;
    }
    case GridExpressionKind::kInputNumElements:
    case GridExpressionKind::kOutputNumElements: {
      const auto& names =
          expression.kind == GridExpressionKind::kInputNumElements
              ? invocation.inputs
              : invocation.outputs;
      const Tensor* tensor = nullptr;
      CUTRITON_RETURN_IF_ERROR(
          tensor_value(names, expression.tensor_index, &tensor));
      *output = tensor->desc().NumElements();
      break;
    }
    case GridExpressionKind::kCeilDiv:
    case GridExpressionKind::kMultiply: {
      if (expression.operands.size() != 2) {
        return Status::InvalidArgument("launch-grid binary expression is malformed");
      }
      int64_t left = 0;
      int64_t right = 0;
      CUTRITON_RETURN_IF_ERROR(EvaluateGridExpression(
          expression.operands[0], invocation, context, &left));
      CUTRITON_RETURN_IF_ERROR(EvaluateGridExpression(
          expression.operands[1], invocation, context, &right));
      if (left <= 0 || right <= 0) {
        return Status::ShapeError("launch-grid operands must be positive");
      }
      if (expression.kind == GridExpressionKind::kCeilDiv) {
        *output = 1 + (left - 1) / right;
      } else {
        if (left > std::numeric_limits<int64_t>::max() / right) {
          return Status::ShapeError("launch-grid multiplication overflow");
        }
        *output = left * right;
      }
      break;
    }
  }
  return *output > 0 ? Status::OK()
                     : Status::ShapeError("launch-grid dimension is not positive");
}

class InvocationKernel final : public OpKernel {
 public:
  static Status Create(KernelInvocation invocation, int device_id,
                       std::shared_ptr<CudaModuleCache> cache,
                       std::unique_ptr<OpKernel>* output) {
    CUfunction function{};
    CUTRITON_RETURN_IF_ERROR(cache->GetFunction(invocation.artifact, &function));
    *output = std::unique_ptr<OpKernel>(new InvocationKernel(
        std::move(invocation), device_id, std::move(cache), function));
    return Status::OK();
  }

  const char* backend_name() const override { return "cuda_triton"; }
  const char* op_type() const override { return invocation_.artifact.op_type.c_str(); }

  Status Compute(KernelContext* context) override {
    if (context == nullptr || context->node == nullptr || context->tensors == nullptr) {
      return Status::InvalidArgument("invalid artifact KernelContext");
    }
    std::vector<BoundArgument> storage;
    std::vector<void*> parameters;
    CUTRITON_RETURN_IF_ERROR(
        ArgumentBinder{}.Bind(invocation_, context, &storage, &parameters));
    std::array<unsigned int, 3> grid{};
    for (std::size_t axis = 0; axis < grid.size(); ++axis) {
      int64_t value = 0;
      CUTRITON_RETURN_IF_ERROR(EvaluateGridExpression(
          invocation_.artifact.launch_grid[axis], invocation_, context,
          &value));
      if (value > std::numeric_limits<unsigned int>::max()) {
        return Status::ShapeError("launch-grid dimension exceeds CUDA limits");
      }
      grid[axis] = static_cast<unsigned int>(value);
    }
    CUcontext cuda_context{};
    CUTRITON_RETURN_IF_ERROR(CudaStatus(cuCtxGetCurrent(&cuda_context), "cuCtxGetCurrent"));
    return CudaStatus(
        cuLaunchKernel(function_, grid[0], grid[1], grid[2],
                       static_cast<unsigned int>(invocation_.artifact.num_warps * 32),
                       1, 1,
                       static_cast<unsigned int>(invocation_.artifact.shared_memory_bytes),
                       reinterpret_cast<CUstream>(context->stream),
                       parameters.data(), nullptr),
        "cuLaunchKernel");
  }

 private:
  InvocationKernel(KernelInvocation invocation, int device_id,
                   std::shared_ptr<CudaModuleCache> cache, CUfunction function)
      : invocation_(std::move(invocation)), device_id_(device_id),
        cache_(std::move(cache)), function_(function) {}

  KernelInvocation invocation_;
  int device_id_{0};
  std::shared_ptr<CudaModuleCache> cache_;
  CUfunction function_{};
};
#endif

Status CreateCudaModuleCache(int device_id,
                             std::shared_ptr<CudaModuleCache>* cache) {
  if (cache == nullptr) return Status::InvalidArgument("module cache output is null");
#if CUTRITON_ENABLE_CUDA
  auto result = std::make_shared<CudaModuleCache>(device_id);
  CUTRITON_RETURN_IF_ERROR(result->Initialize());
  *cache = std::move(result);
  return Status::OK();
#else
  (void)device_id;
  return Status::Unsupported("CUTriton was built without CUDA support");
#endif
}

Status CreateInvocationKernel(const KernelInvocation& invocation, int device_id,
                              std::shared_ptr<CudaModuleCache> cache,
                              std::unique_ptr<OpKernel>* kernel) {
  if (kernel == nullptr || cache == nullptr) {
    return Status::InvalidArgument("invocation kernel arguments are invalid");
  }
#if CUTRITON_ENABLE_CUDA
  return InvocationKernel::Create(invocation, device_id, std::move(cache), kernel);
#else
  (void)invocation; (void)device_id;
  return Status::Unsupported("CUTriton was built without CUDA support");
#endif
}

std::size_t LoadedModuleCount(const std::shared_ptr<CudaModuleCache>& cache) {
  return cache == nullptr ? 0 : cache->size();
}

}  // namespace cutriton
