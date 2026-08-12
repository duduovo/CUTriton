#include <cuda.h>

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "cutriton/cutriton.h"

namespace {

using cutriton::Buffer;
using cutriton::CompileOptions;
using cutriton::DataType;
using cutriton::Engine;
using cutriton::Model;
using cutriton::Node;
using cutriton::Status;
using cutriton::Tensor;
using cutriton::TensorDesc;

void RequireOk(const Status& status) {
  if (!status.ok()) {
    std::cerr << status.ToString() << std::endl;
    std::exit(1);
  }
}

void RequireCuda(CUresult result, const char* operation) {
  if (result == CUDA_SUCCESS) {
    return;
  }
  const char* name = nullptr;
  const char* message = nullptr;
  cuGetErrorName(result, &name);
  cuGetErrorString(result, &message);
  std::cerr << operation << " failed: "
            << (name == nullptr ? "CUDA_ERROR" : name) << " ("
            << (message == nullptr ? "unknown" : message) << ")\n";
  std::exit(1);
}

Tensor HostTensor(std::vector<int64_t> shape, const std::vector<float>& data,
                  std::string layout = "NCHW") {
  TensorDesc desc(std::move(shape), DataType::kFloat32,
                  cutriton::DeviceType::kCPU, 0, std::move(layout));
  if (static_cast<std::size_t>(desc.NumElements()) != data.size()) {
    std::cerr << "Host Tensor element count mismatch\n";
    std::exit(1);
  }
  auto buffer = Buffer::AllocateHost(desc.ByteSize());
  RequireOk(buffer->CopyFromHost(data.data(), desc.ByteSize()));
  return Tensor(std::move(desc), std::move(buffer));
}

class ResNet50Builder {
 public:
  Model Build() {
    RequireOk(model_.graph().AddInput(
        "input", TensorDesc({1, 3, 224, 224}, DataType::kFloat32)));

    std::string value = ConvBn("input", 3, 64, 7, 2, 3, true, "stem");
    Node pool("stem_maxpool", "MaxPool", {value}, {"stem_pool"});
    pool.SetAttribute("kernel_shape", std::vector<int64_t>{3, 3});
    pool.SetAttribute("strides", std::vector<int64_t>{2, 2});
    pool.SetAttribute("pads", std::vector<int64_t>{1, 1, 1, 1});
    pool.SetAttribute("dilations", std::vector<int64_t>{1, 1});
    RequireOk(model_.graph().AddNode(std::move(pool)));
    value = "stem_pool";

    int channels = 64;
    const int blocks[4] = {3, 4, 6, 3};
    const int planes[4] = {64, 128, 256, 512};
    for (int stage = 0; stage < 4; ++stage) {
      for (int block = 0; block < blocks[stage]; ++block) {
        const int stride = stage != 0 && block == 0 ? 2 : 1;
        value = Bottleneck(value, channels, planes[stage], stride, stage + 1,
                           block + 1);
        channels = planes[stage] * 4;
      }
    }

    RequireOk(model_.graph().AddNode(
        Node("global_average_pool", "GlobalAveragePool", {value}, {"gap"})));
    RequireOk(model_.graph().AddNode(
        Node("flatten", "Flatten", {"gap"}, {"flattened"})));
    AddWeight("fc_weight", {2048, 1000}, 2048, "");
    RequireOk(model_.graph().AddNode(
        Node("classifier", "Gemm", {"flattened", "fc_weight"}, {"output"})));
    RequireOk(model_.graph().AddOutput("output"));
    return std::move(model_);
  }

 private:
  void AddConstant(const std::string& name, std::vector<int64_t> shape,
                   const std::vector<float>& data,
                   const std::string& layout = "NCHW") {
    RequireOk(model_.AddConstant(
        name, HostTensor(std::move(shape), data, layout)));
  }

  void AddWeight(const std::string& name, std::vector<int64_t> shape,
                 int fan_in, const std::string& layout = "OIHW") {
    std::size_t count = 1;
    for (const int64_t dimension : shape) {
      count *= static_cast<std::size_t>(dimension);
    }
    // A small deterministic scale keeps this synthetic, untrained ResNet
    // numerically well conditioned across all 50 layers.
    const float scale = 0.02F / std::sqrt(static_cast<float>(fan_in));
    std::vector<float> data(count);
    for (std::size_t index = 0; index < count; ++index) {
      const int value = static_cast<int>(
                            (index * 17U + static_cast<std::size_t>(seed_) * 13U) %
                            19U) -
                        9;
      data[index] = static_cast<float>(value) * scale;
    }
    ++seed_;
    AddConstant(name, std::move(shape), data, layout);
  }

  std::string ConvBn(const std::string& input, int in_channels,
                     int out_channels, int kernel, int stride, int padding,
                     bool relu, const std::string& prefix) {
    const std::string weight = prefix + "_weight";
    const std::string conv_output = prefix + "_conv";
    const std::string bn_output = prefix + "_bn";
    AddWeight(weight,
              {out_channels, in_channels, kernel, kernel},
              in_channels * kernel * kernel);
    AddConstant(prefix + "_scale", {out_channels},
                std::vector<float>(out_channels, 1.0F), "");
    AddConstant(prefix + "_bias", {out_channels},
                std::vector<float>(out_channels, 0.01F), "");
    AddConstant(prefix + "_mean", {out_channels},
                std::vector<float>(out_channels, 0.0F), "");
    AddConstant(prefix + "_variance", {out_channels},
                std::vector<float>(out_channels, 0.99999F), "");

    Node conv(prefix + "_conv_node", "Conv", {input, weight}, {conv_output});
    conv.SetAttribute("strides", std::vector<int64_t>{stride, stride});
    conv.SetAttribute("pads",
                      std::vector<int64_t>{padding, padding, padding, padding});
    conv.SetAttribute("dilations", std::vector<int64_t>{1, 1});
    RequireOk(model_.graph().AddNode(std::move(conv)));
    Node bn(prefix + "_bn_node", "BatchNormalization",
            {conv_output, prefix + "_scale", prefix + "_bias",
             prefix + "_mean", prefix + "_variance"},
            {bn_output});
    bn.SetAttribute("epsilon", 1e-5);
    RequireOk(model_.graph().AddNode(std::move(bn)));
    if (!relu) {
      return bn_output;
    }
    const std::string output = prefix + "_relu";
    RequireOk(model_.graph().AddNode(
        Node(prefix + "_relu_node", "Relu", {bn_output}, {output})));
    return output;
  }

  std::string Bottleneck(const std::string& input, int in_channels, int planes,
                         int stride, int stage, int block) {
    const std::string prefix = "layer" + std::to_string(stage) + "_block" +
                               std::to_string(block);
    std::string value =
        ConvBn(input, in_channels, planes, 1, 1, 0, true, prefix + "_conv1");
    value = ConvBn(value, planes, planes, 3, stride, 1, true,
                   prefix + "_conv2");
    value = ConvBn(value, planes, planes * 4, 1, 1, 0, false,
                   prefix + "_conv3");
    std::string identity = input;
    if (stride != 1 || in_channels != planes * 4) {
      identity = ConvBn(input, in_channels, planes * 4, 1, stride, 0, false,
                        prefix + "_downsample");
    }
    const std::string sum = prefix + "_sum";
    const std::string output = prefix + "_output";
    RequireOk(model_.graph().AddNode(
        Node(prefix + "_add", "Add", {value, identity}, {sum})));
    RequireOk(model_.graph().AddNode(
        Node(prefix + "_relu", "Relu", {sum}, {output})));
    return output;
  }

  Model model_;
  int seed_{1};
};

struct Arguments {
  std::string output_path{"cutriton_resnet50_output.bin"};
  int warmup{2};
  int iterations{10};
  bool cuda_graph{true};
};

Arguments ParseArguments(int argc, char** argv) {
  Arguments result;
  for (int i = 1; i < argc; ++i) {
    const std::string argument = argv[i];
    if (argument == "--output" && i + 1 < argc) {
      result.output_path = argv[++i];
    } else if (argument == "--warmup" && i + 1 < argc) {
      result.warmup = std::stoi(argv[++i]);
    } else if (argument == "--iterations" && i + 1 < argc) {
      result.iterations = std::stoi(argv[++i]);
    } else if (argument == "--cuda-graph" && i + 1 < argc) {
      result.cuda_graph = std::stoi(argv[++i]) != 0;
    } else {
      std::cerr << "Unknown or incomplete argument: " << argument << '\n';
      std::exit(2);
    }
  }
  if (result.warmup < 1 || result.iterations < 1) {
    std::cerr << "warmup and iterations must be positive\n";
    std::exit(2);
  }
  return result;
}

}  // namespace

int main(int argc, char** argv) {
  const Arguments arguments = ParseArguments(argc, argv);
  std::vector<float> input(3 * 224 * 224);
  for (std::size_t index = 0; index < input.size(); ++index) {
    input[index] =
        static_cast<float>(static_cast<int>(index % 29U) - 14) / 29.0F;
  }

  Model model = ResNet50Builder{}.Build();
  CompileOptions options;
  options.target = "cuda_triton";
  options.device_id = 0;
  options.kernel_artifact_paths = {CUTRITON_BENCHMARK_KERNEL_DIR};
  options.enable_cuda_graph = arguments.cuda_graph;
  options.enable_profiling = false;

  const auto compile_start = std::chrono::steady_clock::now();
  std::unique_ptr<Engine> engine;
  RequireOk(cutriton::BuildEngine(model, options, &engine));
  const double compile_ms = std::chrono::duration<double, std::milli>(
                                std::chrono::steady_clock::now() - compile_start)
                                .count();
  auto context = engine->CreateExecutionContext();
  const TensorDesc input_desc =
      engine->plan().graph().FindValue("input")->tensor;
  const TensorDesc output_desc =
      engine->plan().graph().FindValue("output")->tensor;
  std::shared_ptr<Buffer> input_buffer;
  std::shared_ptr<Buffer> output_buffer;
  RequireOk(Buffer::AllocateCuda(input_desc.ByteSize(), 0, &input_buffer));
  RequireOk(Buffer::AllocateCuda(output_desc.ByteSize(), 0, &output_buffer));
  RequireOk(input_buffer->CopyFromHost(input.data(), input_desc.ByteSize()));
  RequireOk(context->BindInput("input", Tensor(input_desc, input_buffer)));
  RequireOk(context->BindOutput("output", Tensor(output_desc, output_buffer)));

  for (int i = 0; i < arguments.warmup; ++i) {
    RequireOk(context->Run());
  }

  CUdevice device{};
  CUcontext cuda_context{};
  CUstream stream{};
  CUevent start{};
  CUevent end{};
  RequireCuda(cuInit(0), "cuInit");
  RequireCuda(cuDeviceGet(&device, 0), "cuDeviceGet");
  RequireCuda(cuDevicePrimaryCtxRetain(&cuda_context, device),
              "cuDevicePrimaryCtxRetain");
  RequireCuda(cuCtxSetCurrent(cuda_context), "cuCtxSetCurrent");
  RequireCuda(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING), "cuStreamCreate");
  RequireCuda(cuEventCreate(&start, CU_EVENT_DEFAULT), "cuEventCreate(start)");
  RequireCuda(cuEventCreate(&end, CU_EVENT_DEFAULT), "cuEventCreate(end)");

  double total_ms = 0.0;
  for (int i = 0; i < arguments.iterations; ++i) {
    RequireCuda(cuEventRecord(start, stream), "cuEventRecord(start)");
    RequireOk(context->RunAsync(reinterpret_cast<void*>(stream)));
    RequireCuda(cuEventRecord(end, stream), "cuEventRecord(end)");
    RequireOk(context->Synchronize());
    float elapsed = 0.0F;
    RequireCuda(cuEventElapsedTime(&elapsed, start, end),
                "cuEventElapsedTime");
    total_ms += elapsed;
  }

  std::vector<float> output(1000);
  RequireOk(output_buffer->CopyToHost(output.data(), output_desc.ByteSize()));
  std::ofstream output_file(arguments.output_path, std::ios::binary);
  output_file.write(reinterpret_cast<const char*>(output.data()),
                    static_cast<std::streamsize>(output_desc.ByteSize()));
  if (!output_file) {
    std::cerr << "Failed to write output: " << arguments.output_path << '\n';
    return 1;
  }

  RequireCuda(cuEventDestroy(start), "cuEventDestroy(start)");
  RequireCuda(cuEventDestroy(end), "cuEventDestroy(end)");
  RequireCuda(cuStreamDestroy(stream), "cuStreamDestroy");
  RequireCuda(cuDevicePrimaryCtxRelease(device),
              "cuDevicePrimaryCtxRelease");

  std::cout << "CUTRITON_RESNET50 compile_ms=" << compile_ms
            << " latency_ms=" << total_ms / arguments.iterations
            << " warmup=" << arguments.warmup
            << " iterations=" << arguments.iterations
            << " cuda_graph=" << (arguments.cuda_graph ? 1 : 0) << '\n';
  return 0;
}
