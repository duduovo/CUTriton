#include <cuda.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <filesystem>
#include <iostream>
#include <memory>
#include <map>
#include <string>
#include <vector>

#include "cutriton/cutriton.h"

namespace {

using namespace cutriton;

void RequireOk(const Status& status) {
  if (!status.ok()) {
    std::cerr << status.ToString() << std::endl;
    std::abort();
  }
}

void RequireCuda(CUresult result) {
  if (result == CUDA_SUCCESS) {
    return;
  }
  const char* name = nullptr;
  cuGetErrorName(result, &name);
  std::cerr << "CUDA Driver error: "
            << (name == nullptr ? "unknown" : name) << std::endl;
  std::abort();
}

Tensor HostTensor(std::vector<int64_t> shape, const std::vector<float>& data) {
  TensorDesc desc(std::move(shape), DataType::kFloat32);
  assert(static_cast<std::size_t>(desc.NumElements()) == data.size());
  auto buffer = Buffer::AllocateHost(desc.ByteSize());
  RequireOk(buffer->CopyFromHost(data.data(), desc.ByteSize()));
  return Tensor(desc, std::move(buffer));
}

void AddConstant(Model* model, const std::string& name,
                 std::vector<int64_t> shape, const std::vector<float>& data) {
  RequireOk(model->AddConstant(name,
                               HostTensor(std::move(shape), data)));
}

Model BuildModel(const std::vector<float>& conv_weights,
                 const std::vector<float>& fc_weights,
                 std::vector<int64_t> input_shape = {1, 3, 224, 224}) {
  Model model;
  auto& graph = model.graph();
  RequireOk(graph.AddInput(
      "input", TensorDesc(std::move(input_shape), DataType::kFloat32)));
  AddConstant(&model, "conv_w", {64, 3, 7, 7}, conv_weights);
  AddConstant(&model, "bn_scale", {64}, std::vector<float>(64, 1.0F));
  AddConstant(&model, "bn_bias", {64}, std::vector<float>(64, 0.0F));
  AddConstant(&model, "bn_mean", {64}, std::vector<float>(64, 0.0F));
  AddConstant(&model, "bn_var", {64}, std::vector<float>(64, 0.99999F));
  AddConstant(&model, "fc_w", {64, 1000}, fc_weights);

  Node conv("conv", "Conv", {"input", "conv_w"}, {"conv_y"});
  conv.SetAttribute("pads", std::vector<int64_t>{3, 3, 3, 3});
  conv.SetAttribute("strides", std::vector<int64_t>{2, 2});
  conv.SetAttribute("dilations", std::vector<int64_t>{1, 1});
  RequireOk(graph.AddNode(std::move(conv)));
  Node bn("bn", "BatchNormalization",
          {"conv_y", "bn_scale", "bn_bias", "bn_mean", "bn_var"},
          {"bn_y"});
  bn.SetAttribute("epsilon", 1e-5);
  RequireOk(graph.AddNode(std::move(bn)));
  RequireOk(graph.AddNode(Node("relu", "Relu", {"bn_y"}, {"relu_y"})));
  RequireOk(graph.AddNode(
      Node("gap", "GlobalAveragePool", {"relu_y"}, {"gap_y"})));
  RequireOk(graph.AddNode(
      Node("flatten", "Flatten", {"gap_y"}, {"flat_y"})));
  RequireOk(graph.AddNode(
      Node("gemm", "Gemm", {"flat_y", "fc_w"}, {"output"})));
  RequireOk(graph.AddOutput("output"));
  return model;
}

void Verify(const std::vector<float>& actual,
            const std::vector<float>& expected) {
  assert(actual.size() == expected.size());
  for (std::size_t i = 0; i < actual.size(); ++i) {
    const float error = std::abs(actual[i] - expected[i]);
    const float tolerance = 1e-4F + 1e-4F * std::abs(expected[i]);
    if (error > tolerance) {
      std::cerr << "Mismatch at " << i << ": " << actual[i] << " vs "
                << expected[i] << std::endl;
      std::abort();
    }
  }
}

void TestDynamicShapeProfile(const std::vector<float>& conv_weights,
                             const std::vector<float>& fc_weights,
                             const CompileOptions& base_options) {
  CompileOptions options = base_options;
  options.enable_profiling = false;
  options.cuda_graph_cache_capacity = 2;
  ShapeProfile profile;
  profile.name = "images";
  profile.inputs["input"] = ShapeRange{
      {1, 3, 160, 160}, {1, 3, 224, 224}, {2, 3, 256, 256}};
  options.shape_profiles = {profile};
  Model model = BuildModel(conv_weights, fc_weights, {-1, 3, -1, -1});
  std::unique_ptr<Engine> engine;
  RequireOk(BuildEngine(model, options, &engine));
  auto context = engine->CreateExecutionContext();
  assert(context->GetResolvedTensorDesc("input") != nullptr);
  assert((context->GetResolvedTensorDesc("output")->shape ==
          std::vector<int64_t>{1, 1000}));
  assert(context->SetInputShape("input", {1, 3, 300, 300}).code() ==
         ErrorCode::kShapeError);

  auto run_shape = [&](const std::vector<int64_t>& shape,
                       std::shared_ptr<Buffer>* saved_input,
                       std::shared_ptr<Buffer>* saved_output) {
    RequireOk(context->SetInputShape("input", shape));
    RequireOk(context->ResolveShapes());
    const auto input_desc = *context->GetResolvedTensorDesc("input");
    const auto output_desc = *context->GetResolvedTensorDesc("output");
    std::vector<float> values(static_cast<std::size_t>(input_desc.NumElements()));
    for (std::size_t index = 0; index < values.size(); ++index) {
      values[index] =
          static_cast<float>(static_cast<int>(index % 29) - 14) / 29.0F;
    }
    std::shared_ptr<Buffer> input;
    std::shared_ptr<Buffer> output;
    RequireOk(Buffer::AllocateCuda(input_desc.ByteSize(), 0, &input));
    RequireOk(Buffer::AllocateCuda(output_desc.ByteSize(), 0, &output));
    RequireOk(input->CopyFromHost(values.data(), input_desc.ByteSize()));
    RequireOk(context->BindInput("input", Tensor(input_desc, input)));
    RequireOk(context->BindOutput("output", Tensor(output_desc, output)));
    RequireOk(context->Run());
    std::vector<float> result(static_cast<std::size_t>(output_desc.NumElements()));
    RequireOk(output->CopyToHost(result.data(), output_desc.ByteSize()));
    for (const float value : result) assert(std::isfinite(value));
    *saved_input = std::move(input);
    *saved_output = std::move(output);
  };

  std::shared_ptr<Buffer> opt_input;
  std::shared_ptr<Buffer> opt_output;
  run_shape({1, 3, 224, 224}, &opt_input, &opt_output);
  assert(context->cached_cuda_graph_count() == 1);
  assert(context->loaded_cuda_module_count() == 3);
  std::shared_ptr<Buffer> small_input;
  std::shared_ptr<Buffer> small_output;
  run_shape({1, 3, 160, 160}, &small_input, &small_output);
  assert(context->cached_cuda_graph_count() == 2);

  RequireOk(context->SetInputShape("input", {1, 3, 224, 224}));
  RequireOk(context->ResolveShapes());
  const auto input_desc = *context->GetResolvedTensorDesc("input");
  const auto output_desc = *context->GetResolvedTensorDesc("output");
  RequireOk(context->BindInput("input", Tensor(input_desc, opt_input)));
  RequireOk(context->BindOutput("output", Tensor(output_desc, opt_output)));
  RequireOk(context->Run());
  assert(context->cached_cuda_graph_count() == 2);
}

void TestAutotuningAndCache(const std::vector<float>& input,
                            const std::vector<float>& conv_weights,
                            const std::vector<float>& fc_weights,
                            const std::vector<float>& expected,
                            const CompileOptions& base_options) {
  const auto cache_directory =
      std::filesystem::temp_directory_path() /
      ("cutriton-tuning-" + std::to_string(
          std::chrono::steady_clock::now().time_since_epoch().count()));
  CompileOptions options = base_options;
  options.enable_cuda_graph = false;
  options.enable_profiling = false;
  options.tuning_mode = TuningMode::kTuneOnMiss;
  options.tuning_cache_dir = cache_directory.string();
  options.tuning_warmup_iterations = 1;
  options.tuning_measurement_iterations = 3;
  std::unique_ptr<Engine> engine;
  RequireOk(BuildEngine(BuildModel(conv_weights, fc_weights), options, &engine));

  auto run = [&]() {
    auto context = engine->CreateExecutionContext();
    const auto input_desc = engine->plan().graph().FindValue("input")->tensor;
    const auto output_desc = engine->plan().graph().FindValue("output")->tensor;
    std::shared_ptr<Buffer> input_buffer;
    std::shared_ptr<Buffer> output_buffer;
    RequireOk(Buffer::AllocateCuda(input_desc.ByteSize(), 0, &input_buffer));
    RequireOk(Buffer::AllocateCuda(output_desc.ByteSize(), 0, &output_buffer));
    RequireOk(input_buffer->CopyFromHost(input.data(), input_desc.ByteSize()));
    RequireOk(context->BindInput("input", Tensor(input_desc, input_buffer)));
    RequireOk(context->BindOutput("output", Tensor(output_desc, output_buffer)));
    RequireOk(context->Run());
    std::vector<float> actual(expected.size());
    RequireOk(output_buffer->CopyToHost(actual.data(), output_desc.ByteSize()));
    Verify(actual, expected);
  };

  run();
  std::map<std::string, std::filesystem::file_time_type> timestamps;
  for (const auto& entry : std::filesystem::directory_iterator(cache_directory)) {
    if (entry.path().extension() == ".json") {
      timestamps.emplace(entry.path().filename().string(),
                         std::filesystem::last_write_time(entry.path()));
    }
  }
  assert(!timestamps.empty());
  run();
  for (const auto& entry : std::filesystem::directory_iterator(cache_directory)) {
    if (entry.path().extension() == ".json") {
      assert(std::filesystem::last_write_time(entry.path()) ==
             timestamps.at(entry.path().filename().string()));
    }
  }
  std::filesystem::remove_all(cache_directory);
}

}  // namespace

int main(int argc, char** argv) {
  std::vector<float> input(3 * 224 * 224);
  for (std::size_t i = 0; i < input.size(); ++i) {
    input[i] = static_cast<float>(static_cast<int>(i % 29) - 14) / 29.0F;
  }
  std::vector<float> conv_weights(64 * 3 * 7 * 7, 0.0F);
  for (int oc = 0; oc < 64; ++oc) {
    const int ic = oc % 3;
    conv_weights[((oc * 3 + ic) * 7 + 3) * 7 + 3] =
        0.05F * static_cast<float>(1 + oc % 5);
  }
  std::vector<float> fc_weights(64 * 1000);
  for (std::size_t i = 0; i < fc_weights.size(); ++i) {
    fc_weights[i] = static_cast<float>(static_cast<int>(i % 11) - 5) / 101.0F;
  }

  std::vector<float> pooled(64, 0.0F);
  for (int oc = 0; oc < 64; ++oc) {
    const int ic = oc % 3;
    const float weight = 0.05F * static_cast<float>(1 + oc % 5);
    double sum = 0.0;
    for (int oh = 0; oh < 112; ++oh) {
      for (int ow = 0; ow < 112; ++ow) {
        const std::size_t index =
            static_cast<std::size_t>((ic * 224 + oh * 2) * 224 + ow * 2);
        sum += std::max(input[index] * weight, 0.0F);
      }
    }
    pooled[oc] = static_cast<float>(sum / (112.0 * 112.0));
  }
  std::vector<float> expected(1000, 0.0F);
  for (int column = 0; column < 1000; ++column) {
    for (int inner = 0; inner < 64; ++inner) {
      expected[column] += pooled[inner] * fc_weights[inner * 1000 + column];
    }
  }

  CompileOptions options;
  options.target = "cuda_triton";
  options.device_id = 0;
  options.kernel_artifact_paths = {CUTRITON_TEST_KERNEL_DIR};
  options.enable_cuda_graph = true;
  options.enable_profiling = true;
  Model model = BuildModel(conv_weights, fc_weights);
  CompileOptions missing_artifact_options = options;
  missing_artifact_options.kernel_artifact_paths = {"missing-triton-artifacts"};
  ExecutablePlan missing_artifact_plan;
  const Status missing_artifact_status = Compiler{}.Compile(
      model, missing_artifact_options, &missing_artifact_plan);
  if (missing_artifact_status.code() != ErrorCode::kNotFound) {
    std::cerr << "Missing artifacts did not fail compilation" << std::endl;
    return 1;
  }
  const auto mismatch_directory =
      std::filesystem::temp_directory_path() /
      ("cutriton-sm-mismatch-" + std::to_string(
          std::chrono::steady_clock::now().time_since_epoch().count()));
  std::filesystem::create_directories(mismatch_directory);
  const std::filesystem::path artifact_directory(CUTRITON_TEST_KERNEL_DIR);
  std::filesystem::copy(artifact_directory / "kernels",
                        mismatch_directory / "kernels",
                        std::filesystem::copy_options::recursive);
  std::ifstream manifest_input(artifact_directory / "pack.json");
  std::string manifest((std::istreambuf_iterator<char>(manifest_input)),
                       std::istreambuf_iterator<char>());
  const std::string original_manifest = manifest;
  std::string legacy_manifest = manifest;
  const std::string schema_v2 = "\"schema_version\": 2";
  const auto schema_position = legacy_manifest.find(schema_v2);
  assert(schema_position != std::string::npos);
  legacy_manifest.replace(schema_position, schema_v2.size(),
                          "\"schema_version\": 1");
  std::ofstream(mismatch_directory / "pack.json") << legacy_manifest;
  ArtifactRepository legacy_repository;
  assert(legacy_repository.Load({mismatch_directory.string()}).code() ==
         ErrorCode::kInvalidArgument);

  const std::string capability = "\"min_compute_capability\": 80";
  std::size_t position = 0;
  while ((position = manifest.find(capability, position)) != std::string::npos) {
    manifest.replace(position, capability.size(),
                     "\"min_compute_capability\": 999");
    position += capability.size();
  }
  std::ofstream(mismatch_directory / "pack.json") << manifest;
  CompileOptions mismatch_options = options;
  mismatch_options.kernel_artifact_paths = {mismatch_directory.string()};
  ExecutablePlan mismatch_plan;
  const Status mismatch_status =
      Compiler{}.Compile(model, mismatch_options, &mismatch_plan);
  if (mismatch_status.code() != ErrorCode::kUnsupported) {
    std::cerr << "SM-incompatible artifacts did not fail compilation"
              << std::endl;
    return 1;
  }
  std::ofstream(mismatch_directory / "pack.json") << original_manifest;
  std::filesystem::path damaged_ptx;
  for (const auto& entry : std::filesystem::recursive_directory_iterator(
           mismatch_directory / "kernels")) {
    if (entry.path().extension() == ".ptx") {
      damaged_ptx = entry.path();
      break;
    }
  }
  assert(!damaged_ptx.empty());
  std::ofstream(damaged_ptx, std::ios::app) << "\n";
  ArtifactRepository damaged_repository;
  assert(damaged_repository.Load({mismatch_directory.string()}).code() ==
         ErrorCode::kInvalidArgument);
  std::filesystem::remove_all(mismatch_directory);
  std::unique_ptr<Engine> engine;
  RequireOk(BuildEngine(model, options, &engine));
  auto context = engine->CreateExecutionContext();
  const auto input_desc = engine->plan().graph().FindValue("input")->tensor;
  const auto output_desc = engine->plan().graph().FindValue("output")->tensor;
  std::shared_ptr<Buffer> input_buffer;
  std::shared_ptr<Buffer> output_buffer;
  RequireOk(Buffer::AllocateCuda(input_desc.ByteSize(), 0, &input_buffer));
  RequireOk(Buffer::AllocateCuda(output_desc.ByteSize(), 0, &output_buffer));
  RequireOk(input_buffer->CopyFromHost(input.data(), input_desc.ByteSize()));
  RequireOk(context->BindInput("input", Tensor(input_desc, input_buffer)));
  RequireOk(context->BindOutput("output", Tensor(output_desc, output_buffer)));

  RequireOk(context->Run());
  std::vector<float> actual(1000);
  RequireOk(output_buffer->CopyToHost(actual.data(), output_desc.ByteSize()));
  Verify(actual, expected);
  assert(context->profiler().events().size() == 4);
  assert(context->profiler().events()[2].duration_ms == 0.0);

  RequireOk(context->Run());
  RequireOk(output_buffer->CopyToHost(actual.data(), output_desc.ByteSize()));
  Verify(actual, expected);
  if (argc == 3 && std::string(argv[1]) == "--dump") {
    std::ofstream dump(argv[2], std::ios::binary);
    dump.write(reinterpret_cast<const char*>(actual.data()),
               static_cast<std::streamsize>(output_desc.ByteSize()));
    if (!dump) {
      std::cerr << "Failed to write CUDA test output" << std::endl;
      return 1;
    }
  }

  std::shared_ptr<Buffer> rebound_output;
  RequireOk(Buffer::AllocateCuda(output_desc.ByteSize(), 0, &rebound_output));
  RequireOk(context->BindOutput(
      "output", Tensor(output_desc, rebound_output)));
  RequireOk(context->Run());
  RequireOk(rebound_output->CopyToHost(actual.data(), output_desc.ByteSize()));
  Verify(actual, expected);

  CUdevice device{};
  CUcontext cuda_context{};
  CUstream stream{};
  RequireCuda(cuInit(0));
  RequireCuda(cuDeviceGet(&device, 0));
  RequireCuda(cuDevicePrimaryCtxRetain(&cuda_context, device));
  RequireCuda(cuCtxSetCurrent(cuda_context));
  RequireCuda(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING));
  RequireOk(context->RunAsync(reinterpret_cast<void*>(stream)));
  RequireOk(context->Synchronize());
  RequireCuda(cuStreamDestroy(stream));
  RequireCuda(cuDevicePrimaryCtxRelease(device));

  TestDynamicShapeProfile(conv_weights, fc_weights, options);
  TestAutotuningAndCache(input, conv_weights, fc_weights, expected, options);

  std::cout << "CUTriton CUDA/Triton ResNet stem test passed" << std::endl;
  return 0;
}
