#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <utility>
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

class MockKernel final : public OpKernel {
 public:
  explicit MockKernel(std::string op_type) : op_type_(std::move(op_type)) {}
  const char* backend_name() const override { return "mock"; }
  const char* op_type() const override { return op_type_.c_str(); }

  Status Compute(KernelContext* context) override {
    if (context == nullptr || context->node == nullptr ||
        context->tensors == nullptr) {
      return Status::InvalidArgument("invalid mock context");
    }
    ScopedCpuTimer timer(context->profiler, context->node->name(), "mock");
    if (op_type_ != "Relu") {
      return Status::OK();
    }
    Tensor& input = context->tensors->at(context->node->inputs()[0]);
    Tensor& output = context->tensors->at(context->node->outputs()[0]);
    const auto count = static_cast<std::size_t>(input.desc().NumElements());
    const auto* source = reinterpret_cast<const float*>(
        static_cast<const std::uint8_t*>(input.buffer()->data()) +
        input.byte_offset());
    auto* destination = reinterpret_cast<float*>(
        static_cast<std::uint8_t*>(output.buffer()->data()) +
        output.byte_offset());
    for (std::size_t i = 0; i < count; ++i) {
      destination[i] = std::max(source[i], 0.0F);
    }
    return Status::OK();
  }

 private:
  std::string op_type_;
};

class MockBackend final : public Backend {
 public:
  explicit MockBackend(std::string backend_name = "mock")
      : name_(std::move(backend_name)) {}
  const char* name() const override { return name_.c_str(); }
  Status CheckSupport(const Node&, const Graph&,
                      const BackendOptions&) const override {
    return Status::OK();
  }
  Status CreateKernel(const Node& node, const Graph&, const BackendOptions&,
                      std::unique_ptr<OpKernel>* kernel) const override {
    if (kernel == nullptr) {
      return Status::InvalidArgument("null mock kernel output");
    }
    *kernel = std::make_unique<MockKernel>(node.op_type());
    return Status::OK();
  }

 private:
  std::string name_;
};

Tensor HostTensor(const std::vector<int64_t>& shape,
                  DataType dtype = DataType::kFloat32) {
  TensorDesc desc(shape, dtype);
  return Tensor(desc, Buffer::AllocateHost(desc.ByteSize()));
}

void AddConstant(Model* model, const std::string& name,
                 const std::vector<int64_t>& shape,
                 DataType dtype = DataType::kFloat32) {
  Tensor tensor = HostTensor(shape, dtype);
  RequireOk(model->AddConstant(name, std::move(tensor)));
}

Model BuildResNetStemGraph() {
  Model model;
  Graph& graph = model.graph();
  RequireOk(graph.AddInput("input",
                           TensorDesc({1, 3, 224, 224}, DataType::kFloat32)));
  AddConstant(&model, "conv_w", {64, 3, 7, 7});
  AddConstant(&model, "bn_scale", {64});
  AddConstant(&model, "bn_bias", {64});
  AddConstant(&model, "bn_mean", {64});
  AddConstant(&model, "bn_var", {64});
  AddConstant(&model, "fc_w", {64, 1000});

  Node conv("conv1", "Conv", {"input", "conv_w"}, {"conv_y"});
  conv.SetAttribute("pads", std::vector<int64_t>{3, 3, 3, 3});
  conv.SetAttribute("strides", std::vector<int64_t>{2, 2});
  conv.SetAttribute("dilations", std::vector<int64_t>{1, 1});
  RequireOk(graph.AddNode(std::move(conv)));
  Node bn("bn1", "BatchNormalization",
          {"conv_y", "bn_scale", "bn_bias", "bn_mean", "bn_var"},
          {"bn_y"});
  bn.SetAttribute("epsilon", 1e-3);
  RequireOk(graph.AddNode(std::move(bn)));
  RequireOk(graph.AddNode(Node("relu1", "Relu", {"bn_y"}, {"relu_y"})));
  RequireOk(graph.AddNode(
      Node("gap", "GlobalAveragePool", {"relu_y"}, {"gap_y"})));
  Node flatten("flatten", "Flatten", {"gap_y"}, {"flat_y"});
  flatten.SetAttribute("axis", static_cast<int64_t>(1));
  RequireOk(graph.AddNode(std::move(flatten)));
  RequireOk(graph.AddNode(Node("fc", "Gemm", {"flat_y", "fc_w"},
                               {"logits"})));
  RequireOk(graph.AddOutput("logits"));
  return model;
}

Model BuildReluGraph() {
  Model model;
  RequireOk(model.graph().AddInput(
      "x", TensorDesc({4}, DataType::kFloat32, DeviceType::kCPU)));
  RequireOk(model.graph().AddNode(Node("relu", "Relu", {"x"}, {"y"})));
  RequireOk(model.graph().AddOutput("y"));
  return model;
}

Model BuildReluChainGraph() {
  Model model;
  RequireOk(model.graph().AddInput(
      "x", TensorDesc({17}, DataType::kFloat32, DeviceType::kCPU)));
  RequireOk(model.graph().AddNode(Node("a", "Relu", {"x"}, {"a"})));
  RequireOk(model.graph().AddNode(Node("b", "Relu", {"a"}, {"b"})));
  RequireOk(model.graph().AddNode(Node("c", "Relu", {"b"}, {"c"})));
  RequireOk(model.graph().AddNode(Node("y", "Relu", {"c"}, {"y"})));
  RequireOk(model.graph().AddOutput("y"));
  return model;
}

Model BuildTransformerFusionGraph(bool expose_intermediate = false) {
  Model model;
  auto& graph = model.graph();
  const TensorDesc activation({16, 128}, DataType::kFloat16,
                              DeviceType::kCPU, 0, "");
  RequireOk(graph.AddInput("x", activation));
  RequireOk(graph.AddInput(
      "residual", TensorDesc({16, 512}, DataType::kFloat16,
                             DeviceType::kCPU, 0, "")));
  AddConstant(&model, "weight", {128, 512}, DataType::kFloat16);
  AddConstant(&model, "scale", {512}, DataType::kFloat16);
  AddConstant(&model, "bias", {512}, DataType::kFloat16);
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
  if (expose_intermediate) RequireOk(graph.AddOutput("projected"));
  RequireOk(graph.AddOutput("output"));
  return model;
}

void InstallMockBackend() {
  auto& registry = KernelRegistry::Global();
  registry.ClearForTests();
  RequireOk(registry.RegisterBackend(std::make_shared<MockBackend>()));
}

void TestTensorBufferAndConstants() {
  TensorDesc desc({1, 3, 4, 5}, DataType::kFloat32);
  assert(desc.ByteSize() == 240);
  TensorDesc half_desc({3, 5, 7}, DataType::kFloat16);
  assert(half_desc.ByteSize() == 210);
  assert(std::string(DataTypeName(DataType::kFloat16)) == "float16");
  auto buffer = Buffer::AllocateHost(desc.ByteSize());
  assert(buffer->owns_memory());

  Model model;
  RequireOk(model.AddConstant("weight", Tensor(desc, buffer)));
  assert(model.constants().size() == 1);
  assert(model.graph().FindValue("weight")->is_constant);
  Status duplicate = model.AddConstant("weight", Tensor(desc, buffer));
  assert(duplicate.code() == ErrorCode::kAlreadyExists);

  auto too_small = Buffer::AllocateHost(4);
  Status invalid =
      Model{}.AddConstant("bad", Tensor(desc, std::move(too_small)));
  assert(invalid.code() == ErrorCode::kInvalidArgument);

  Model late_constant;
  RequireOk(late_constant.graph().AddNode(
      Node("add", "Add", {"x", "late_weight"}, {"y"})));
  RequireOk(late_constant.AddConstant("late_weight", Tensor(desc, buffer)));
  assert(late_constant.graph().FindValue("late_weight")->is_constant);
}

void TestCompilerPipeline() {
  InstallMockBackend();
  Model model = BuildResNetStemGraph();
  CompileOptions options;
  options.target = "mock";
  options.enable_cuda_graph = false;
  ExecutablePlan plan;
  RequireOk(Compiler{}.Compile(model, options, &plan));
  assert(plan.constants().size() == 6);
  assert(plan.ops().size() == 4);
  assert(plan.ops()[0].op_type == "FusedConvBatchNormRelu");
  const Node& fused = plan.graph().nodes()[0];
  assert(std::abs(fused.GetAttribute<double>("batchnorm_epsilon").value() -
                  1e-3) < 1e-9);
  assert(plan.ops()[2].op_type == "Flatten");
  const auto* logits = plan.graph().FindValue("logits");
  assert(logits != nullptr);
  assert((logits->tensor.shape == std::vector<int64_t>{1, 1000}));
  assert(plan.memory_plan().workspace_size_bytes > 0);
}

void TestMemoryPlannerAlignmentReuseAndAlias() {
  Graph graph;
  RequireOk(graph.AddInput("input", TensorDesc({17}, DataType::kFloat32)));
  RequireOk(graph.AddNode(Node("a", "Relu", {"input"}, {"a"})));
  RequireOk(graph.SetValueDesc("a", TensorDesc({17}, DataType::kFloat32)));
  RequireOk(graph.AddNode(Node("b", "Relu", {"a"}, {"b"})));
  RequireOk(graph.SetValueDesc("b", TensorDesc({17}, DataType::kFloat32)));
  RequireOk(graph.AddNode(Node("c", "Relu", {"b"}, {"c"})));
  RequireOk(graph.SetValueDesc("c", TensorDesc({17}, DataType::kFloat32)));
  RequireOk(graph.AddNode(Node("out", "Relu", {"c"}, {"output"})));
  RequireOk(graph.SetValueDesc("output", TensorDesc({17}, DataType::kFloat32)));
  RequireOk(graph.AddOutput("output"));
  MemoryPlan plan;
  RequireOk(MemoryPlanner{}.Plan(graph, &plan));
  assert(plan.workspace_size_bytes % 256 == 0);
  assert(plan.allocations[0].offset == 0);
  assert(plan.allocations[1].offset == 256);
  assert(plan.allocations[2].offset == 0);
  assert(plan.allocations[2].reused);

  Graph alias_graph;
  RequireOk(alias_graph.AddInput("x", TensorDesc({1, 4}, DataType::kFloat32)));
  RequireOk(alias_graph.AddNode(Node("relu", "Relu", {"x"}, {"hidden"})));
  RequireOk(alias_graph.SetValueDesc(
      "hidden", TensorDesc({1, 4}, DataType::kFloat32)));
  RequireOk(alias_graph.AddNode(
      Node("flatten", "Flatten", {"hidden"}, {"flat"})));
  RequireOk(alias_graph.SetValueDesc(
      "flat", TensorDesc({1, 4}, DataType::kFloat32, DeviceType::kCPU, 0, "")));
  RequireOk(alias_graph.AddNode(Node("out", "Relu", {"flat"}, {"y"})));
  RequireOk(alias_graph.SetValueDesc("y", TensorDesc({1, 4}, DataType::kFloat32)));
  RequireOk(alias_graph.AddOutput("y"));
  RequireOk(MemoryPlanner{}.Plan(alias_graph, &plan));
  assert(plan.allocations.size() == 2);
  assert(plan.allocations[1].alias_of == "hidden");
  assert(plan.allocations[1].offset == plan.allocations[0].offset);
}

void TestRegistryThreadSafety() {
  auto& registry = KernelRegistry::Global();
  registry.ClearForTests();
  std::vector<std::thread> threads;
  for (int i = 0; i < 8; ++i) {
    threads.emplace_back([i, &registry]() {
      RequireOk(registry.RegisterBackend(
          std::make_shared<MockBackend>("mock_" + std::to_string(i))));
    });
  }
  for (auto& thread : threads) {
    thread.join();
  }
  assert(registry.AvailableBackends().size() == 8);
  Status duplicate =
      registry.RegisterBackend(std::make_shared<MockBackend>("mock_0"));
  assert(duplicate.code() == ErrorCode::kAlreadyExists);
}

void TestSchemaFusionAndShapeProfiles() {
  OpSchemaRegistry::Global().ClearForTests();
  FusionRegistry::Global().ClearForTests();
  RequireOk(RegisterBuiltinOpSchemas());
  RequireOk(RegisterBuiltinFusionAlternatives());
  const auto relu = OpSchemaRegistry::Global().Find("Relu");
  assert(relu != nullptr && relu->min_inputs == 1 && relu->outputs == 1);
  const auto alternatives =
      FusionRegistry::Global().Find("FusedConvBatchNormRelu");
  assert(alternatives.size() == 1);
  assert((alternatives.front().decomposed_ops ==
          std::vector<std::string>{"Conv", "BatchNormalization", "Relu"}));
  const auto gemm_gelu = FusionRegistry::Global().Find("GemmGelu");
  assert(gemm_gelu.size() == 1);
  const auto skip_norm =
      FusionRegistry::Global().Find("SkipLayerNormalization");
  assert(skip_norm.size() == 1);

  InstallMockBackend();
  CompileOptions options;
  options.target = "mock";
  options.enable_cuda_graph = false;
  ShapeProfile profile;
  profile.name = "batch";
  profile.inputs["x"] = ShapeRange{{2}, {4}, {8}};
  options.shape_profiles = {profile};
  std::unique_ptr<Engine> engine;
  RequireOk(BuildEngine(BuildReluGraph(), options, &engine));
  assert(engine->plan().profile_plans().size() == 1);
  assert((engine->plan().profile_plans()[0]
              .max_graph.FindValue("y")->tensor.shape ==
          std::vector<int64_t>{8}));
  auto context = engine->CreateExecutionContext();
  RequireOk(context->SetInputShape("x", {2}));
  RequireOk(context->ResolveShapes());
  assert((context->GetResolvedTensorDesc("y")->shape ==
          std::vector<int64_t>{2}));
  assert(context->SetInputShape("x", {9}).code() == ErrorCode::kShapeError);
}

void TestTransformerFusionPasses() {
  Model model = BuildTransformerFusionGraph();
  auto passes = CreateDefaultCompilePasses();
  RequireOk(passes.Run(&model.graph()));
  assert(model.graph().nodes().size() == 2);
  assert(model.graph().nodes()[0].op_type() == "GemmGelu");
  assert(model.graph().nodes()[1].op_type() == "SkipLayerNormalization");
  const auto* output = model.graph().FindValue("output");
  assert(output != nullptr);
  assert(output->tensor.dtype == DataType::kFloat16);
  assert((output->tensor.shape == std::vector<int64_t>{16, 512}));
  assert(model.graph().nodes()[1]
             .GetAttribute<double>("epsilon")
             .value_or(0.0) == 1e-5);

  Model observable = BuildTransformerFusionGraph(true);
  passes = CreateDefaultCompilePasses();
  RequireOk(passes.Run(&observable.graph()));
  assert(observable.graph().nodes()[0].op_type() == "Gemm");
  assert(observable.graph().nodes()[1].op_type() == "Gelu");
}

void TestKernelPackAbiCompatibility() {
  const auto root = std::filesystem::temp_directory_path() /
                    ("cutriton-pack-abi-" + std::to_string(
                         std::chrono::steady_clock::now()
                             .time_since_epoch()
                             .count()));
  std::filesystem::create_directories(root / "kernels");
  std::ofstream(root / "kernels" / "dummy.ptx", std::ios::binary)
      << "// ptx\n";
  const std::string prefix = R"({
  "schema_version": 2,
  "abi_schema_version": )";
  const std::string common = R"(,
  "pack_name": "test",
  "pack_version": 1,
  "generator_version": "test",
  "triton_version": "test",
  "kernels": [{
    "kernel_id": "relu.test",
    "op_type": "Relu",
    "kernel_version": 1,
    "variant_id": "default",
    "default": true,
    "symbol": "relu",
    "binary": "kernels/dummy.ptx",
    "binary_format": "ptx",
    "sha256": "f2ee4765bc61528cec8f6987ea5b1c2ea694169ea5ab848e2026139b285db499",
    "dtype": "float32",
    "layout": "",
    "min_compute_capability": 80,
    "num_warps": 4,
    "num_stages": 1,
    "shared_memory_bytes": 0,
    "arguments": [
      {"name":"x","type":"*fp32","source":{"kind":"input","index":0}},
      {"name":"output","type":"*fp32","source":{"kind":"output","index":0}},
      {"name":"runtime_scratch","type":"*i8","source":{"kind":"runtime_reserved"}},
      {"name":"profile_scratch","type":"*i8","source":{"kind":"runtime_reserved"}}
    ],
    "constraints": [],
    "tuning": {"BLOCK": 256},
    "grid": )";
  const std::string suffix = "\n  }]\n}\n";
  auto write = [&](int abi, const std::string& grid) {
    std::ofstream(root / "pack.json")
        << prefix << abi << common << grid << suffix;
  };

  write(1, R"({"op":"ceil_div","value":{"kind":"output_numel","index":0},"divisor":256})");
  ArtifactRepository repository;
  RequireOk(repository.Load({root.string()}));
  assert(repository.artifacts().front().abi_schema_version == 1);
  assert(repository.artifacts().front().launch_grid[0].kind ==
         GridExpressionKind::kCeilDiv);

  write(2, R"([{"kind":"ceil_div","args":[{"kind":"output_numel","index":0},{"kind":"meta","name":"BLOCK"}]}])");
  RequireOk(repository.Load({root.string()}));
  assert(repository.artifacts().front().abi_schema_version == 2);
  assert(repository.artifacts().front().launch_meta.at("BLOCK") == 256);

  write(2, R"([{"kind":"python","expression":"unsafe"}])");
  assert(repository.Load({root.string()}).code() ==
         ErrorCode::kInvalidArgument);
  std::filesystem::remove_all(root);
}

void TestBindingExecutionAndLifetime() {
  InstallMockBackend();
  CompileOptions options;
  options.target = "mock";
  options.enable_cuda_graph = false;
  options.enable_profiling = true;
  std::unique_ptr<Engine> engine;
  RequireOk(BuildEngine(BuildReluGraph(), options, &engine));
  auto context = engine->CreateExecutionContext();
  const TensorDesc input_desc = engine->plan().graph().FindValue("x")->tensor;
  const TensorDesc output_desc = engine->plan().graph().FindValue("y")->tensor;
  engine.reset();

  Tensor input(input_desc, Buffer::AllocateHost(input_desc.ByteSize()));
  Tensor output(output_desc, Buffer::AllocateHost(output_desc.ByteSize()));
  const float values[4] = {-2.0F, -0.5F, 1.0F, 3.0F};
  RequireOk(input.buffer()->CopyFromHost(values, sizeof(values)));
  assert(context->BindInput("y", input).code() == ErrorCode::kNotFound);
  Tensor wrong_shape = HostTensor({5});
  assert(context->BindInput("x", wrong_shape).code() ==
         ErrorCode::kInvalidArgument);
  RequireOk(context->BindInput("x", input));
  RequireOk(context->BindOutput("y", output));
  RequireOk(context->RunAsync(nullptr));
  assert(context->RunAsync(nullptr).code() == ErrorCode::kRuntimeError);
  RequireOk(context->Synchronize());
  float result[4]{};
  RequireOk(output.buffer()->CopyToHost(result, sizeof(result)));
  assert(result[0] == 0.0F && result[1] == 0.0F);
  assert(result[2] == 1.0F && result[3] == 3.0F);
  assert(context->profiler().events().size() == 1);

  options.enable_profiling = false;
  RequireOk(BuildEngine(BuildReluGraph(), options, &engine));
  auto no_profile_context = engine->CreateExecutionContext();
  RequireOk(no_profile_context->BindInput("x", input));
  RequireOk(no_profile_context->BindOutput("y", output));
  RequireOk(no_profile_context->Run());
  assert(no_profile_context->profiler().events().empty());
}

void TestWorkspaceIsConnectedToExecution() {
  InstallMockBackend();
  CompileOptions options;
  options.target = "mock";
  options.enable_cuda_graph = false;
  std::unique_ptr<Engine> engine;
  RequireOk(BuildEngine(BuildReluChainGraph(), options, &engine));
  auto context = engine->CreateExecutionContext();
  const auto input_desc = engine->plan().graph().FindValue("x")->tensor;
  const auto output_desc = engine->plan().graph().FindValue("y")->tensor;
  RequireOk(context->BindInput(
      "x", Tensor(input_desc, Buffer::AllocateHost(input_desc.ByteSize()))));
  RequireOk(context->BindOutput(
      "y", Tensor(output_desc, Buffer::AllocateHost(output_desc.ByteSize()))));
  RequireOk(context->Run());
  const auto& tensors = context->tensors();
  assert(tensors.at("a").buffer() == tensors.at("b").buffer());
  assert(tensors.at("a").buffer() == tensors.at("c").buffer());
  assert(tensors.at("a").byte_offset() == tensors.at("c").byte_offset());
  assert(tensors.at("a").byte_offset() != tensors.at("b").byte_offset());
}

}  // namespace

int main() {
  TestTensorBufferAndConstants();
  TestCompilerPipeline();
  TestMemoryPlannerAlignmentReuseAndAlias();
  TestRegistryThreadSafety();
  TestSchemaFusionAndShapeProfiles();
  TestTransformerFusionPasses();
  TestKernelPackAbiCompatibility();
  TestBindingExecutionAndLifetime();
  TestWorkspaceIsConnectedToExecution();
  std::cout << "CUTriton core tests passed" << std::endl;
  return 0;
}
