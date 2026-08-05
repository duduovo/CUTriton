#include "cutriton/backend/backend.h"
#include "cutriton/core/buffer.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <mutex>
#include <regex>
#include <sstream>
#include <utility>

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

#if CUTRITON_ENABLE_CUDA
struct KernelArtifact {
  std::string op_type;
  std::string symbol;
  std::filesystem::path ptx_path;
  std::string dtype;
  std::string layout;
  std::string sha256;
  int min_compute_capability{80};
  int num_warps{4};
  int shared_memory_bytes{0};
};

std::uint32_t RotateRight(std::uint32_t value, unsigned int count) {
  return (value >> count) | (value << (32U - count));
}

std::string Sha256(const std::string& input) {
  static constexpr std::array<std::uint32_t, 64> constants = {
      0x428a2f98U, 0x71374491U, 0xb5c0fbcfU, 0xe9b5dba5U, 0x3956c25bU,
      0x59f111f1U, 0x923f82a4U, 0xab1c5ed5U, 0xd807aa98U, 0x12835b01U,
      0x243185beU, 0x550c7dc3U, 0x72be5d74U, 0x80deb1feU, 0x9bdc06a7U,
      0xc19bf174U, 0xe49b69c1U, 0xefbe4786U, 0x0fc19dc6U, 0x240ca1ccU,
      0x2de92c6fU, 0x4a7484aaU, 0x5cb0a9dcU, 0x76f988daU, 0x983e5152U,
      0xa831c66dU, 0xb00327c8U, 0xbf597fc7U, 0xc6e00bf3U, 0xd5a79147U,
      0x06ca6351U, 0x14292967U, 0x27b70a85U, 0x2e1b2138U, 0x4d2c6dfcU,
      0x53380d13U, 0x650a7354U, 0x766a0abbU, 0x81c2c92eU, 0x92722c85U,
      0xa2bfe8a1U, 0xa81a664bU, 0xc24b8b70U, 0xc76c51a3U, 0xd192e819U,
      0xd6990624U, 0xf40e3585U, 0x106aa070U, 0x19a4c116U, 0x1e376c08U,
      0x2748774cU, 0x34b0bcb5U, 0x391c0cb3U, 0x4ed8aa4aU, 0x5b9cca4fU,
      0x682e6ff3U, 0x748f82eeU, 0x78a5636fU, 0x84c87814U, 0x8cc70208U,
      0x90befffaU, 0xa4506cebU, 0xbef9a3f7U, 0xc67178f2U};
  std::vector<std::uint8_t> data(input.begin(), input.end());
  const std::uint64_t bit_length = static_cast<std::uint64_t>(data.size()) * 8U;
  data.push_back(0x80U);
  while (data.size() % 64U != 56U) {
    data.push_back(0U);
  }
  for (int shift = 56; shift >= 0; shift -= 8) {
    data.push_back(static_cast<std::uint8_t>(bit_length >> shift));
  }
  std::array<std::uint32_t, 8> hash = {
      0x6a09e667U, 0xbb67ae85U, 0x3c6ef372U, 0xa54ff53aU,
      0x510e527fU, 0x9b05688cU, 0x1f83d9abU, 0x5be0cd19U};
  for (std::size_t offset = 0; offset < data.size(); offset += 64U) {
    std::array<std::uint32_t, 64> words{};
    for (std::size_t i = 0; i < 16U; ++i) {
      const std::size_t base = offset + i * 4U;
      words[i] = (static_cast<std::uint32_t>(data[base]) << 24U) |
                 (static_cast<std::uint32_t>(data[base + 1U]) << 16U) |
                 (static_cast<std::uint32_t>(data[base + 2U]) << 8U) |
                 static_cast<std::uint32_t>(data[base + 3U]);
    }
    for (std::size_t i = 16U; i < words.size(); ++i) {
      const auto s0 = RotateRight(words[i - 15U], 7U) ^
                      RotateRight(words[i - 15U], 18U) ^
                      (words[i - 15U] >> 3U);
      const auto s1 = RotateRight(words[i - 2U], 17U) ^
                      RotateRight(words[i - 2U], 19U) ^
                      (words[i - 2U] >> 10U);
      words[i] = words[i - 16U] + s0 + words[i - 7U] + s1;
    }
    auto a = hash[0];
    auto b = hash[1];
    auto c = hash[2];
    auto d = hash[3];
    auto e = hash[4];
    auto f = hash[5];
    auto g = hash[6];
    auto h = hash[7];
    for (std::size_t i = 0; i < words.size(); ++i) {
      const auto sum1 = RotateRight(e, 6U) ^ RotateRight(e, 11U) ^
                        RotateRight(e, 25U);
      const auto choice = (e & f) ^ ((~e) & g);
      const auto temp1 = h + sum1 + choice + constants[i] + words[i];
      const auto sum0 = RotateRight(a, 2U) ^ RotateRight(a, 13U) ^
                        RotateRight(a, 22U);
      const auto majority = (a & b) ^ (a & c) ^ (b & c);
      const auto temp2 = sum0 + majority;
      h = g;
      g = f;
      f = e;
      e = d + temp1;
      d = c;
      c = b;
      b = a;
      a = temp1 + temp2;
    }
    hash[0] += a;
    hash[1] += b;
    hash[2] += c;
    hash[3] += d;
    hash[4] += e;
    hash[5] += f;
    hash[6] += g;
    hash[7] += h;
  }
  std::ostringstream output;
  output << std::hex << std::setfill('0');
  for (const auto value : hash) {
    output << std::setw(8) << value;
  }
  return output.str();
}

std::string JsonString(const std::string& object, const std::string& key) {
  const std::regex expression("\\\"" + key +
                              "\\\"\\s*:\\s*\\\"([^\\\"]*)\\\"");
  std::smatch match;
  return std::regex_search(object, match, expression) ? match[1].str() : "";
}

int JsonInt(const std::string& object, const std::string& key, int fallback) {
  const std::regex expression("\\\"" + key +
                              "\\\"\\s*:\\s*(-?[0-9]+)");
  std::smatch match;
  return std::regex_search(object, match, expression)
             ? std::stoi(match[1].str())
             : fallback;
}

Status LoadArtifact(const BackendOptions& options, const std::string& op_type,
                    KernelArtifact* artifact) {
  if (artifact == nullptr) {
    return Status::InvalidArgument("KernelArtifact output pointer is null");
  }
  if (options.kernel_artifact_dir.empty()) {
    return Status::NotFound("kernel_artifact_dir is empty");
  }
  const std::filesystem::path root(options.kernel_artifact_dir);
  const auto manifest_path = root / "manifest.json";
  std::ifstream input(manifest_path);
  if (!input) {
    return Status::NotFound("Triton manifest not found: " +
                            manifest_path.string());
  }
  std::ostringstream contents;
  contents << input.rdbuf();
  const std::string json = contents.str();
  if (JsonInt(json, "schema_version", -1) != 1) {
    return Status::InvalidArgument("Unsupported Triton manifest schema");
  }
  if (JsonString(json, "triton_version") != "3.6.0") {
    return Status::InvalidArgument(
        "Triton manifest was not produced by Triton 3.6.0");
  }

  const std::regex object_expression("\\{[^{}]*\\\"op_type\\\"[^{}]*\\}");
  for (std::sregex_iterator it(json.begin(), json.end(), object_expression), end;
       it != end; ++it) {
    const std::string object = it->str();
    if (JsonString(object, "op_type") != op_type) {
      continue;
    }
    KernelArtifact result;
    result.op_type = op_type;
    result.symbol = JsonString(object, "symbol");
    result.ptx_path = root / JsonString(object, "ptx");
    result.dtype = JsonString(object, "dtype");
    result.layout = JsonString(object, "layout");
    result.sha256 = JsonString(object, "sha256");
    result.min_compute_capability =
        JsonInt(object, "min_compute_capability", 80);
    result.num_warps = JsonInt(object, "num_warps", 4);
    result.shared_memory_bytes = JsonInt(object, "shared_memory_bytes", 0);
    if (result.symbol.empty() || result.ptx_path.filename().empty() ||
        result.sha256.empty() || object.find("\"abi\"") == std::string::npos) {
      return Status::InvalidArgument("Incomplete manifest entry for " + op_type);
    }
    if (!std::filesystem::exists(result.ptx_path)) {
      return Status::NotFound("PTX artifact not found: " +
                              result.ptx_path.string());
    }
    std::ifstream ptx_input(result.ptx_path, std::ios::binary);
    std::ostringstream ptx_contents;
    ptx_contents << ptx_input.rdbuf();
    if (Sha256(ptx_contents.str()) != result.sha256) {
      return Status::InvalidArgument("PTX SHA-256 mismatch: " +
                                     result.ptx_path.string());
    }
    *artifact = std::move(result);
    return Status::OK();
  }
  return Status::NotFound("No Triton artifact for op: " + op_type);
}

Status CheckTensor(const Graph& graph, const std::string& name,
                   DeviceType device_type, int device_id,
                   const std::string& required_layout = "") {
  const auto* value = graph.FindValue(name);
  if (value == nullptr) {
    return Status::NotFound("Tensor description is missing: " + name);
  }
  CUTRITON_RETURN_IF_ERROR(value->tensor.Validate());
  if (value->tensor.dtype != DataType::kFloat32) {
    return Status::Unsupported("only float32 is supported: " + name);
  }
  if (value->tensor.device_type != device_type ||
      value->tensor.device_id != device_id) {
    return Status::Unsupported("Tensor is assigned to the wrong device: " +
                               name);
  }
  if (!required_layout.empty() && value->tensor.layout != required_layout) {
    return Status::Unsupported("unsupported Tensor layout for " + name +
                               ": " + value->tensor.layout);
  }
  return Status::OK();
}
#endif

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

Status SetDevice(int device_id, CUcontext* context, CUdevice* device) {
  CUTRITON_RETURN_IF_ERROR(CudaStatus(cuInit(0), "cuInit"));
  CUTRITON_RETURN_IF_ERROR(
      CudaStatus(cuDeviceGet(device, device_id), "cuDeviceGet"));
  CUTRITON_RETURN_IF_ERROR(CudaStatus(
      cuDevicePrimaryCtxRetain(context, *device), "cuDevicePrimaryCtxRetain"));
  CUTRITON_RETURN_IF_ERROR(CudaStatus(cuCtxSetCurrent(*context), "cuCtxSetCurrent"));
  return Status::OK();
}

CUdeviceptr TensorPointer(const Tensor& tensor) {
  return static_cast<CUdeviceptr>(
             reinterpret_cast<std::uintptr_t>(tensor.buffer()->data())) +
         tensor.byte_offset();
}

class CudaTritonKernel final : public OpKernel {
 public:
  ~CudaTritonKernel() override {
    if (module_ != nullptr) {
      CUcontext context{};
      CUdevice device{};
      if (SetDevice(device_id_, &context, &device).ok()) {
        cuModuleUnload(module_);
        cuDevicePrimaryCtxRelease(device);
      }
    }
  }

  static Status Create(const Node& node, const KernelArtifact& artifact,
                       int device_id,
                       std::unique_ptr<OpKernel>* output) {
    CUcontext context{};
    CUdevice device{};
    CUTRITON_RETURN_IF_ERROR(SetDevice(device_id, &context, &device));
    std::ifstream input(artifact.ptx_path, std::ios::binary);
    std::ostringstream contents;
    contents << input.rdbuf();
    auto kernel = std::unique_ptr<CudaTritonKernel>(new CudaTritonKernel(
        node.op_type(), device_id, artifact.num_warps,
        artifact.shared_memory_bytes));
    Status status = CudaStatus(
        cuModuleLoadDataEx(&kernel->module_, contents.str().c_str(), 0, nullptr,
                           nullptr),
        "cuModuleLoadDataEx");
    if (status.ok()) {
      status = CudaStatus(cuModuleGetFunction(&kernel->function_, kernel->module_,
                                              artifact.symbol.c_str()),
                          "cuModuleGetFunction");
    }
    cuDevicePrimaryCtxRelease(device);
    if (!status.ok()) {
      return status;
    }
    *output = std::move(kernel);
    return Status::OK();
  }

  const char* backend_name() const override { return "cuda_triton"; }
  const char* op_type() const override { return op_type_.c_str(); }

  Status Compute(KernelContext* context) override {
    if (context == nullptr || context->node == nullptr ||
        context->tensors == nullptr) {
      return Status::InvalidArgument("CUDA KernelContext is invalid");
    }
    CUcontext cuda_context{};
    CUdevice device{};
    CUTRITON_RETURN_IF_ERROR(SetDevice(device_id_, &cuda_context, &device));
    const auto& node = *context->node;
    Status status;
    if (op_type_ == "FusedConvBatchNormRelu") {
      status = LaunchFused(node, context);
    } else if (op_type_ == "GlobalAveragePool") {
      status = LaunchGlobalAveragePool(node, context);
    } else if (op_type_ == "Gemm") {
      status = LaunchGemm(node, context);
    } else {
      status = Status::Unsupported("Unknown CUDA op: " + op_type_);
    }
    cuDevicePrimaryCtxRelease(device);
    return status;
  }

 private:
  CudaTritonKernel(std::string op_type, int device_id, int num_warps,
                   int shared_memory_bytes)
      : op_type_(std::move(op_type)),
        device_id_(device_id),
        block_size_(num_warps * 32),
        shared_memory_bytes_(shared_memory_bytes) {}

  Tensor& TensorAt(KernelContext* context, const std::string& name) const {
    return context->tensors->at(name);
  }

  Status Launch(CUstream stream, unsigned int grid_x,
                std::vector<void*> arguments) {
    return CudaStatus(
        cuLaunchKernel(function_, grid_x, 1, 1,
                       static_cast<unsigned int>(block_size_), 1, 1,
                       static_cast<unsigned int>(shared_memory_bytes_), stream,
                       arguments.data(), nullptr),
        "cuLaunchKernel");
  }

  Status LaunchFused(const Node& node, KernelContext* context) {
    if (node.inputs().size() != 6 || node.outputs().size() != 1) {
      return Status::InvalidArgument(
          "FusedConvBatchNormRelu expects 6 inputs and 1 output");
    }
    auto x = TensorPointer(TensorAt(context, node.inputs()[0]));
    auto w = TensorPointer(TensorAt(context, node.inputs()[1]));
    auto scale = TensorPointer(TensorAt(context, node.inputs()[2]));
    auto bias = TensorPointer(TensorAt(context, node.inputs()[3]));
    auto mean = TensorPointer(TensorAt(context, node.inputs()[4]));
    auto variance = TensorPointer(TensorAt(context, node.inputs()[5]));
    auto y = TensorPointer(TensorAt(context, node.outputs()[0]));
    const auto& xd = TensorAt(context, node.inputs()[0]).desc().shape;
    const auto& wd = TensorAt(context, node.inputs()[1]).desc().shape;
    const auto& yd = TensorAt(context, node.outputs()[0]).desc().shape;
    int n = static_cast<int>(xd[0]);
    int c = static_cast<int>(xd[1]);
    int h = static_cast<int>(xd[2]);
    int width = static_cast<int>(xd[3]);
    int k = static_cast<int>(wd[0]);
    int r = static_cast<int>(wd[2]);
    int s = static_cast<int>(wd[3]);
    int oh = static_cast<int>(yd[2]);
    int ow = static_cast<int>(yd[3]);
    auto strides = GetIntListAttribute(node, "strides", {1, 1});
    auto pads = GetIntListAttribute(node, "pads", {0, 0, 0, 0});
    auto dilations = GetIntListAttribute(node, "dilations", {1, 1});
    int stride_h = static_cast<int>(strides[0]);
    int stride_w = static_cast<int>(strides[1]);
    int pad_h = static_cast<int>(pads[0]);
    int pad_w = static_cast<int>(pads[1]);
    int dilation_h = static_cast<int>(dilations[0]);
    int dilation_w = static_cast<int>(dilations[1]);
    float epsilon = static_cast<float>(
        node.GetAttribute<double>("batchnorm_epsilon").value_or(1e-5));
    std::vector<void*> args{&x, &w, &scale, &bias, &mean, &variance, &y,
                            &n, &c, &h, &width, &k, &r, &s, &oh, &ow,
                            &stride_h, &stride_w, &pad_h, &pad_w,
                            &dilation_h, &dilation_w, &epsilon};
    const auto count = static_cast<unsigned int>(n * k * oh * ow);
    return Launch(reinterpret_cast<CUstream>(context->stream),
                  (count + 255U) / 256U, std::move(args));
  }

  Status LaunchGlobalAveragePool(const Node& node, KernelContext* context) {
    auto x = TensorPointer(TensorAt(context, node.inputs()[0]));
    auto y = TensorPointer(TensorAt(context, node.outputs()[0]));
    const auto& dims = TensorAt(context, node.inputs()[0]).desc().shape;
    int n = static_cast<int>(dims[0]);
    int c = static_cast<int>(dims[1]);
    int h = static_cast<int>(dims[2]);
    int w = static_cast<int>(dims[3]);
    std::vector<void*> args{&x, &y, &n, &c, &h, &w};
    return Launch(reinterpret_cast<CUstream>(context->stream),
                  static_cast<unsigned int>(n * c), std::move(args));
  }

  Status LaunchGemm(const Node& node, KernelContext* context) {
    auto a = TensorPointer(TensorAt(context, node.inputs()[0]));
    auto b = TensorPointer(TensorAt(context, node.inputs()[1]));
    auto y = TensorPointer(TensorAt(context, node.outputs()[0]));
    const auto& ad = TensorAt(context, node.inputs()[0]).desc().shape;
    const auto& bd = TensorAt(context, node.inputs()[1]).desc().shape;
    int m = static_cast<int>(ad[0]);
    int k = static_cast<int>(ad[1]);
    int n = static_cast<int>(bd[1]);
    std::vector<void*> args{&a, &b, &y, &m, &n, &k};
    return Launch(reinterpret_cast<CUstream>(context->stream),
                  static_cast<unsigned int>(m * n), std::move(args));
  }

  std::string op_type_;
  int device_id_{0};
  int block_size_{128};
  int shared_memory_bytes_{0};
  CUmodule module_{nullptr};
  CUfunction function_{nullptr};
};
#endif

class CudaTritonBackend final : public Backend {
 public:
  const char* name() const override { return "cuda_triton"; }

  Status CheckSupport(const Node& node, const Graph& graph,
                      const BackendOptions& options) const override {
#if !CUTRITON_ENABLE_CUDA
    (void)node;
    (void)graph;
    (void)options;
    return Status::Unsupported("CUTriton was built without CUDA support");
#else
    if (options.device.type != DeviceType::kCUDA) {
      return Status::Unsupported("cuda_triton requires a CUDA device");
    }
    if (node.op_type() != "FusedConvBatchNormRelu" &&
        node.op_type() != "GlobalAveragePool" && node.op_type() != "Flatten" &&
        node.op_type() != "Gemm") {
      return Status::Unsupported("no FP32 Triton factory for " +
                                 node.op_type());
    }
    const std::string layout =
        (node.op_type() == "FusedConvBatchNormRelu" ||
         node.op_type() == "GlobalAveragePool")
            ? "NCHW"
            : "";
    for (const auto& input : node.inputs()) {
      CUTRITON_RETURN_IF_ERROR(CheckTensor(
          graph, input, DeviceType::kCUDA, options.device.id,
          input == node.inputs().front() ? layout : ""));
    }
    for (const auto& output : node.outputs()) {
      CUTRITON_RETURN_IF_ERROR(CheckTensor(
          graph, output, DeviceType::kCUDA, options.device.id, layout));
    }
    if (node.op_type() == "Flatten") {
      if (node.outputs().size() != 1 ||
          std::find(graph.outputs().begin(), graph.outputs().end(),
                    node.outputs()[0]) != graph.outputs().end()) {
        return Status::Unsupported(
            "Flatten graph outputs require an explicit copy kernel");
      }
      return Status::OK();
    }
    KernelArtifact artifact;
    CUTRITON_RETURN_IF_ERROR(LoadArtifact(options, node.op_type(), &artifact));
    if (artifact.dtype != "float32" ||
        (!layout.empty() && artifact.layout != layout)) {
      return Status::Unsupported("Triton artifact metadata does not match " +
                                 node.op_type());
    }
    CUcontext context{};
    CUdevice device{};
    CUTRITON_RETURN_IF_ERROR(SetDevice(options.device.id, &context, &device));
    int major = 0;
    int minor = 0;
    Status status = CudaStatus(
        cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
                             device),
        "cuDeviceGetAttribute");
    if (status.ok()) {
      status = CudaStatus(
          cuDeviceGetAttribute(&minor,
                               CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
                               device),
          "cuDeviceGetAttribute");
    }
    cuDevicePrimaryCtxRelease(device);
    if (!status.ok()) {
      return status;
    }
    if (major * 10 + minor < artifact.min_compute_capability) {
      return Status::Unsupported("GPU compute capability is below artifact "
                                 "minimum");
    }
    return Status::OK();
#endif
  }

  Status CreateKernel(const Node& node, const Graph& graph,
                      const BackendOptions& options,
                      std::unique_ptr<OpKernel>* kernel) const override {
    if (kernel == nullptr) {
      return Status::InvalidArgument("OpKernel output pointer is null");
    }
    CUTRITON_RETURN_IF_ERROR(CheckSupport(node, graph, options));
    if (node.op_type() == "Flatten") {
      *kernel = std::make_unique<ViewKernel>();
      return Status::OK();
    }
#if CUTRITON_ENABLE_CUDA
    KernelArtifact artifact;
    CUTRITON_RETURN_IF_ERROR(LoadArtifact(options, node.op_type(), &artifact));
    return CudaTritonKernel::Create(node, artifact, options.device.id, kernel);
#else
    return Status::Unsupported("CUTriton was built without CUDA support");
#endif
  }
};

Status RegisterBuiltin(std::shared_ptr<Backend> backend) {
  auto& registry = KernelRegistry::Global();
  if (registry.GetBackend(backend->name()) != nullptr) {
    return Status::OK();
  }
  Status status = registry.RegisterBackend(std::move(backend));
  return status.code() == ErrorCode::kAlreadyExists ? Status::OK() : status;
}

}  // namespace

KernelRegistry& KernelRegistry::Global() {
  static KernelRegistry registry;
  return registry;
}

Status KernelRegistry::RegisterBackend(std::shared_ptr<Backend> backend) {
  if (backend == nullptr) {
    return Status::InvalidArgument("Backend must not be null");
  }
  const std::string name = backend->name();
  if (name.empty()) {
    return Status::InvalidArgument("Backend name must not be empty");
  }
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
  names.reserve(backends_.size());
  for (const auto& entry : backends_) {
    names.push_back(entry.first);
  }
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
