#include <cuda.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "cutriton/cutriton.h"

namespace {

using namespace cutriton;
constexpr int kHidden = 128;
constexpr int kIntermediate = 512;

void RequireOk(const Status& status) {
  if (!status.ok()) {
    std::cerr << status.ToString() << std::endl;
    std::exit(1);
  }
}

void RequireCuda(CUresult result) {
  if (result == CUDA_SUCCESS) return;
  const char* name = nullptr;
  cuGetErrorName(result, &name);
  std::cerr << "CUDA error: " << (name == nullptr ? "unknown" : name)
            << std::endl;
  std::exit(1);
}

std::uint16_t FloatToHalfBits(float value) {
  std::uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  const std::uint32_t sign = (bits >> 16U) & 0x8000U;
  int exponent = static_cast<int>((bits >> 23U) & 0xFFU) - 112;
  std::uint32_t mantissa = bits & 0x7FFFFFU;
  if (exponent <= 0) {
    if (exponent < -10) return static_cast<std::uint16_t>(sign);
    mantissa = (mantissa | 0x800000U) >> static_cast<unsigned int>(1 - exponent);
    return static_cast<std::uint16_t>(sign | ((mantissa + 0x1000U) >> 13U));
  }
  if (exponent >= 31) return static_cast<std::uint16_t>(sign | 0x7C00U);
  mantissa += 0x1000U;
  if ((mantissa & 0x800000U) != 0) {
    mantissa = 0;
    if (++exponent >= 31) return static_cast<std::uint16_t>(sign | 0x7C00U);
  }
  return static_cast<std::uint16_t>(
      sign | (static_cast<std::uint32_t>(exponent) << 10U) |
      (mantissa >> 13U));
}

std::vector<std::uint16_t> ToHalf(const std::vector<float>& values) {
  std::vector<std::uint16_t> result(values.size());
  std::transform(values.begin(), values.end(), result.begin(), FloatToHalfBits);
  return result;
}

Tensor HostHalfTensor(std::vector<int64_t> shape,
                      const std::vector<float>& data) {
  TensorDesc desc(std::move(shape), DataType::kFloat16,
                  DeviceType::kCPU, 0, "");
  auto buffer = Buffer::AllocateHost(desc.ByteSize());
  const auto half = ToHalf(data);
  RequireOk(buffer->CopyFromHost(half.data(), desc.ByteSize()));
  return Tensor(desc, std::move(buffer));
}

struct Inputs {
  std::vector<float> x;
  std::vector<float> residual;
  std::vector<float> weight;
  std::vector<float> scale;
  std::vector<float> bias;
};

Inputs MakeInputs(int tokens) {
  Inputs values;
  values.x.resize(static_cast<std::size_t>(tokens * kHidden));
  values.residual.resize(static_cast<std::size_t>(tokens * kIntermediate));
  values.weight.resize(static_cast<std::size_t>(kHidden * kIntermediate));
  values.scale.resize(kIntermediate);
  values.bias.resize(kIntermediate);
  for (std::size_t index = 0; index < values.x.size(); ++index) {
    values.x[index] = static_cast<float>(static_cast<int>(index % 23) - 11) / 37.0F;
  }
  for (std::size_t index = 0; index < values.residual.size(); ++index) {
    values.residual[index] =
        static_cast<float>(static_cast<int>(index % 17) - 8) / 53.0F;
  }
  for (std::size_t index = 0; index < values.weight.size(); ++index) {
    values.weight[index] =
        static_cast<float>(static_cast<int>(index % 19) - 9) / 181.0F;
  }
  for (int index = 0; index < kIntermediate; ++index) {
    values.scale[static_cast<std::size_t>(index)] =
        0.9F + 0.001F * (index % 13);
    values.bias[static_cast<std::size_t>(index)] =
        static_cast<float>((index % 7) - 3) / 100.0F;
  }
  return values;
}

Model BuildModel(int tokens, const Inputs& values) {
  Model model;
  auto& graph = model.graph();
  RequireOk(graph.AddInput(
      "x", TensorDesc({tokens, kHidden}, DataType::kFloat16,
                      DeviceType::kCPU, 0, "")));
  RequireOk(graph.AddInput(
      "residual", TensorDesc({tokens, kIntermediate}, DataType::kFloat16,
                             DeviceType::kCPU, 0, "")));
  RequireOk(model.AddConstant(
      "weight", HostHalfTensor({kHidden, kIntermediate}, values.weight)));
  RequireOk(model.AddConstant(
      "scale", HostHalfTensor({kIntermediate}, values.scale)));
  RequireOk(model.AddConstant(
      "bias", HostHalfTensor({kIntermediate}, values.bias)));
  RequireOk(graph.AddNode(
      Node("projection", "Gemm", {"x", "weight"}, {"projected"})));
  RequireOk(graph.AddNode(
      Node("activation", "Gelu", {"projected"}, {"activated"})));
  RequireOk(graph.AddNode(Node("residual_add", "Add",
                               {"activated", "residual"}, {"sum"})));
  Node norm("output_norm", "LayerNormalization",
            {"sum", "scale", "bias"}, {"output"});
  norm.SetAttribute("axis", static_cast<int64_t>(-1));
  norm.SetAttribute("epsilon", 1e-5);
  RequireOk(graph.AddNode(std::move(norm)));
  RequireOk(graph.AddOutput("output"));
  return model;
}

struct Runner {
  std::unique_ptr<Engine> engine;
  std::unique_ptr<ExecutionContext> context;
  std::shared_ptr<Buffer> input;
  std::shared_ptr<Buffer> residual;
  std::shared_ptr<Buffer> output;
};

Runner CreateRunner(int tokens, const Inputs& values, bool fused) {
  CompileOptions options;
  options.target = "cuda_triton";
  options.device_id = 0;
  options.kernel_artifact_paths = {CUTRITON_BENCHMARK_KERNEL_DIR};
  options.enable_cuda_graph = true;
  options.enable_profiling = false;
  options.enable_transformer_fusions = fused;
  options.tuning_mode = TuningMode::kDisabled;
  Runner runner;
  RequireOk(BuildEngine(BuildModel(tokens, values), options, &runner.engine));
  runner.context = runner.engine->CreateExecutionContext();
  const auto input_desc = runner.engine->plan().graph().FindValue("x")->tensor;
  const auto residual_desc =
      runner.engine->plan().graph().FindValue("residual")->tensor;
  const auto output_desc =
      runner.engine->plan().graph().FindValue("output")->tensor;
  RequireOk(Buffer::AllocateCuda(input_desc.ByteSize(), 0, &runner.input));
  RequireOk(Buffer::AllocateCuda(residual_desc.ByteSize(), 0, &runner.residual));
  RequireOk(Buffer::AllocateCuda(output_desc.ByteSize(), 0, &runner.output));
  const auto input_half = ToHalf(values.x);
  const auto residual_half = ToHalf(values.residual);
  RequireOk(runner.input->CopyFromHost(input_half.data(), input_desc.ByteSize()));
  RequireOk(runner.residual->CopyFromHost(
      residual_half.data(), residual_desc.ByteSize()));
  RequireOk(runner.context->BindInput("x", Tensor(input_desc, runner.input)));
  RequireOk(runner.context->BindInput(
      "residual", Tensor(residual_desc, runner.residual)));
  RequireOk(runner.context->BindOutput(
      "output", Tensor(output_desc, runner.output)));
  return runner;
}

struct Timing {
  double median_ms = 0.0;
  double p95_ms = 0.0;
};

Timing Benchmark(Runner* runner, int warmup, int iterations) {
  CUstream stream{};
  CUevent start{};
  CUevent end{};
  RequireCuda(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING));
  RequireCuda(cuEventCreate(&start, CU_EVENT_DEFAULT));
  RequireCuda(cuEventCreate(&end, CU_EVENT_DEFAULT));
  for (int iteration = 0; iteration < warmup; ++iteration) {
    RequireOk(runner->context->RunAsync(reinterpret_cast<void*>(stream)));
    RequireOk(runner->context->Synchronize());
  }
  std::vector<float> samples;
  samples.reserve(static_cast<std::size_t>(iterations));
  for (int iteration = 0; iteration < iterations; ++iteration) {
    RequireCuda(cuEventRecord(start, stream));
    RequireOk(runner->context->RunAsync(reinterpret_cast<void*>(stream)));
    RequireCuda(cuEventRecord(end, stream));
    RequireOk(runner->context->Synchronize());
    float elapsed = 0.0F;
    RequireCuda(cuEventElapsedTime(&elapsed, start, end));
    samples.push_back(elapsed);
  }
  RequireCuda(cuEventDestroy(start));
  RequireCuda(cuEventDestroy(end));
  RequireCuda(cuStreamDestroy(stream));
  std::sort(samples.begin(), samples.end());
  const std::size_t p95_index = std::min(
      samples.size() - 1,
      static_cast<std::size_t>(std::ceil(samples.size() * 0.95)) - 1);
  return {samples[samples.size() / 2], samples[p95_index]};
}

}  // namespace

int main(int argc, char** argv) {
  int tokens = 1024;
  int warmup = 50;
  int iterations = 200;
  std::string output_path;
  for (int index = 1; index < argc; ++index) {
    const std::string argument(argv[index]);
    if (argument == "--tokens" && index + 1 < argc) tokens = std::stoi(argv[++index]);
    else if (argument == "--warmup" && index + 1 < argc) warmup = std::stoi(argv[++index]);
    else if (argument == "--iterations" && index + 1 < argc) iterations = std::stoi(argv[++index]);
    else if (argument == "--output" && index + 1 < argc) output_path = argv[++index];
    else {
      std::cerr << "Unknown or incomplete argument: " << argument << std::endl;
      return 2;
    }
  }
  if (tokens <= 0 || warmup <= 0 || iterations <= 0) return 2;
  RequireCuda(cuInit(0));
  const Inputs values = MakeInputs(tokens);
  const auto compile_started = std::chrono::steady_clock::now();
  Runner fused = CreateRunner(tokens, values, true);
  Runner unfused = CreateRunner(tokens, values, false);
  const double compile_ms = std::chrono::duration<double, std::milli>(
                                std::chrono::steady_clock::now() - compile_started)
                                .count();
  const Timing fused_timing = Benchmark(&fused, warmup, iterations);
  const Timing unfused_timing = Benchmark(&unfused, warmup, iterations);
  if (!output_path.empty()) {
    const auto* desc = fused.engine->plan().graph().FindValue("output");
    std::vector<std::uint16_t> output(
        static_cast<std::size_t>(desc->tensor.NumElements()));
    RequireOk(fused.output->CopyToHost(output.data(), desc->tensor.ByteSize()));
    std::ofstream file(output_path, std::ios::binary);
    file.write(reinterpret_cast<const char*>(output.data()),
               static_cast<std::streamsize>(desc->tensor.ByteSize()));
    if (!file) return 1;
  }
  std::cout << "CUTRITON_TRANSFORMER_FFN tokens=" << tokens
            << " hidden=" << kHidden << " intermediate=" << kIntermediate
            << " dtype=fp16 warmup=" << warmup
            << " iterations=" << iterations
            << " compile_ms=" << compile_ms
            << " fused_ms=" << fused_timing.median_ms
            << " fused_p95_ms=" << fused_timing.p95_ms
            << " unfused_ms=" << unfused_timing.median_ms
            << " unfused_p95_ms=" << unfused_timing.p95_ms
            << " fusion_speedup="
            << unfused_timing.median_ms / fused_timing.median_ms << std::endl;
  return 0;
}
