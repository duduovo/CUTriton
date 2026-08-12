#include "cutriton/ir/pass.h"
#include "cutriton/ir/op_schema.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <set>
#include <unordered_set>

namespace cutriton {
namespace {

//计算总的元素个数
int64_t Product(const std::vector<int64_t>& dims, std::size_t begin,
                std::size_t end) {
  int64_t value = 1;
  for (std::size_t i = begin; i < end; ++i) {
    if (dims[i] <= 0) {
      return 0;
    }
    value *= dims[i];
  }
  return value;
}

TensorDesc SameAsFirstInput(const Graph& graph, const Node& node) {
  if (node.inputs().empty()) {
    return {};
  }
  const auto* input = graph.FindValue(node.inputs().front());
  return input == nullptr ? TensorDesc{} : input->tensor;
}

Status SetOutputLikeInput(Graph* graph, const Node& node) {
  const TensorDesc desc = SameAsFirstInput(*graph, node);
  if (desc.dtype == DataType::kUnknown) {
    return Status::ShapeError("无法推导节点形状: " + node.name());
  }
  for (const auto& output : node.outputs()) {
    CUTRITON_RETURN_IF_ERROR(graph->SetValueDesc(output, desc));
  }
  return Status::OK();
}

Status InferElementwiseBinary(Graph* graph, const Node& node) {
  if (node.inputs().size() != 2 || node.outputs().size() != 1) {
    return Status::ShapeError(node.op_type() +
                              " expects two inputs and one output");
  }
  const auto* first = graph->FindValue(node.inputs()[0]);
  const auto* second = graph->FindValue(node.inputs()[1]);
  if (first == nullptr || second == nullptr) {
    return Status::ShapeError(node.op_type() + " references an unknown Tensor");
  }
  if (first->tensor.shape != second->tensor.shape ||
      first->tensor.dtype != second->tensor.dtype ||
      first->tensor.layout != second->tensor.layout) {
    return Status::ShapeError(node.op_type() +
                              " inputs must have identical descriptions");
  }
  return graph->SetValueDesc(node.outputs()[0], first->tensor);
}

Status InferConvLike(Graph* graph, const Node& node) {
  if (node.inputs().size() < 2 || node.outputs().empty()) {
    return Status::ShapeError("Conv 类节点需要输入、权重和输出");
  }
  const auto* input = graph->FindValue(node.inputs()[0]);
  const auto* weight = graph->FindValue(node.inputs()[1]);
  if (input == nullptr || weight == nullptr) {
    return Status::ShapeError("Conv 类节点引用了未定义 Tensor");
  }
  const auto& in_shape = input->tensor.shape;
  const auto& w_shape = weight->tensor.shape;
  if (in_shape.size() != 4 || w_shape.size() != 4) {
    return Status::ShapeError("V1 Conv 期望 NCHW 输入和 OIHW 权重");
  }

  const auto strides = GetIntListAttribute(node, "strides", {1, 1});
  const auto pads = GetIntListAttribute(node, "pads", {0, 0, 0, 0});
  const auto dilations = GetIntListAttribute(node, "dilations", {1, 1});
  if (strides.size() != 2 || pads.size() != 4 || dilations.size() != 2) {
    return Status::ShapeError("Conv 属性必须是 strides[2]、pads[4]、dilations[2]");
  }

  const int64_t kernel_h = w_shape[2];
  const int64_t kernel_w = w_shape[3];
  const int64_t out_h =
      (in_shape[2] + pads[0] + pads[2] - dilations[0] * (kernel_h - 1) - 1) /
          strides[0] +
      1;
  const int64_t out_w =
      (in_shape[3] + pads[1] + pads[3] - dilations[1] * (kernel_w - 1) - 1) /
          strides[1] +
      1;
  TensorDesc desc({in_shape[0], w_shape[0], out_h, out_w}, input->tensor.dtype,
                  input->tensor.device_type, input->tensor.device_id,
                  input->tensor.layout);
  CUTRITON_RETURN_IF_ERROR(desc.Validate());
  return graph->SetValueDesc(node.outputs()[0], std::move(desc));
}

Status InferPool(Graph* graph, const Node& node) {
  const auto* input = graph->FindValue(node.inputs().front());
  if (input == nullptr || input->tensor.shape.size() != 4) {
    return Status::ShapeError("Pool 期望 NCHW 输入");
  }
  const auto kernel = GetIntListAttribute(node, "kernel_shape", {1, 1});
  const auto strides = GetIntListAttribute(node, "strides", kernel);
  const auto pads = GetIntListAttribute(node, "pads", {0, 0, 0, 0});
  const auto dilations = GetIntListAttribute(node, "dilations", {1, 1});
  if (kernel.size() != 2 || strides.size() != 2 || pads.size() != 4 ||
      dilations.size() != 2) {
    return Status::ShapeError(
        "Pool 属性必须是 kernel_shape[2]、strides[2]、pads[4]、dilations[2]");
  }
  const auto& shape = input->tensor.shape;
  const int64_t effective_h = dilations[0] * (kernel[0] - 1) + 1;
  const int64_t effective_w = dilations[1] * (kernel[1] - 1) + 1;
  const int64_t out_h =
      (shape[2] + pads[0] + pads[2] - effective_h) / strides[0] + 1;
  const int64_t out_w =
      (shape[3] + pads[1] + pads[3] - effective_w) / strides[1] + 1;
  TensorDesc desc({shape[0], shape[1], out_h, out_w}, input->tensor.dtype,
                  input->tensor.device_type, input->tensor.device_id,
                  input->tensor.layout);
  CUTRITON_RETURN_IF_ERROR(desc.Validate());
  return graph->SetValueDesc(node.outputs()[0], std::move(desc));
}

Status InferFlatten(Graph* graph, const Node& node) {
  const auto* input = graph->FindValue(node.inputs().front());
  if (input == nullptr || input->tensor.shape.empty()) {
    return Status::ShapeError("Flatten 期望已知输入形状");
  }
  const auto& shape = input->tensor.shape;
  int64_t axis = GetIntAttribute(node, "axis", 1);
  if (axis < 0) {
    axis += static_cast<int64_t>(shape.size());
  }
  if (axis < 0 || axis > static_cast<int64_t>(shape.size())) {
    return Status::ShapeError("Flatten axis 超出输入 rank");
  }
  TensorDesc desc({Product(shape, 0, static_cast<std::size_t>(axis)),
                   Product(shape, static_cast<std::size_t>(axis), shape.size())},
                  input->tensor.dtype, input->tensor.device_type,
                  input->tensor.device_id, "");
  CUTRITON_RETURN_IF_ERROR(desc.Validate());
  return graph->SetValueDesc(node.outputs()[0], std::move(desc));
}

Status InferGemm(Graph* graph, const Node& node) {
  if (node.inputs().size() < 2) {
    return Status::ShapeError("Gemm 期望两个输入");
  }
  const auto* a = graph->FindValue(node.inputs()[0]);
  const auto* b = graph->FindValue(node.inputs()[1]);
  if (a == nullptr || b == nullptr || a->tensor.shape.size() != 2 ||
      b->tensor.shape.size() != 2) {
    return Status::ShapeError("Gemm 期望 rank-2 输入");
  }
  if (a->tensor.dtype != b->tensor.dtype) {
    return Status::ShapeError("Gemm inputs must have the same dtype");
  }
  const bool trans_a = GetIntAttribute(node, "transA", 0) != 0;
  const bool trans_b = GetIntAttribute(node, "transB", 0) != 0;
  const int64_t m = trans_a ? a->tensor.shape[1] : a->tensor.shape[0];
  const int64_t a_k = trans_a ? a->tensor.shape[0] : a->tensor.shape[1];
  const int64_t b_k = trans_b ? b->tensor.shape[1] : b->tensor.shape[0];
  const int64_t n = trans_b ? b->tensor.shape[0] : b->tensor.shape[1];
  if (a_k != b_k) {
    return Status::ShapeError("Gemm inner dimensions do not match");
  }
  if (node.inputs().size() == 3) {
    const auto* bias = graph->FindValue(node.inputs()[2]);
    if (bias == nullptr || bias->tensor.dtype != a->tensor.dtype ||
        (bias->tensor.shape != std::vector<int64_t>{n} &&
         bias->tensor.shape != std::vector<int64_t>{m, n})) {
      return Status::ShapeError("Gemm bias must be [N] or [M,N]");
    }
  }
  TensorDesc desc({m, n}, a->tensor.dtype,
                  a->tensor.device_type, a->tensor.device_id, "");
  CUTRITON_RETURN_IF_ERROR(desc.Validate());
  return graph->SetValueDesc(node.outputs()[0], std::move(desc));
}

Status InferLayerNormalization(Graph* graph, const Node& node) {
  if (node.inputs().size() < 2 || node.inputs().size() > 3) {
    return Status::ShapeError("LayerNormalization expects input, scale and optional bias");
  }
  const auto* input = graph->FindValue(node.inputs()[0]);
  const auto* scale = graph->FindValue(node.inputs()[1]);
  if (input == nullptr || scale == nullptr || input->tensor.shape.empty() ||
      scale->tensor.shape !=
          std::vector<int64_t>{input->tensor.shape.back()} ||
      input->tensor.dtype != scale->tensor.dtype) {
    return Status::ShapeError("LayerNormalization scale must match the last dimension");
  }
  if (node.inputs().size() == 3) {
    const auto* bias = graph->FindValue(node.inputs()[2]);
    if (bias == nullptr || bias->tensor.shape != scale->tensor.shape ||
        bias->tensor.dtype != input->tensor.dtype) {
      return Status::ShapeError("LayerNormalization bias must match scale");
    }
  }
  const int64_t axis = GetIntAttribute(node, "axis", -1);
  if (axis != -1 && axis != static_cast<int64_t>(input->tensor.shape.size()) - 1) {
    return Status::Unsupported("only last-axis LayerNormalization is supported");
  }
  return graph->SetValueDesc(node.outputs()[0], input->tensor);
}

Status InferSkipLayerNormalization(Graph* graph, const Node& node) {
  if (node.inputs().size() != 4) {
    return Status::ShapeError(
        "SkipLayerNormalization expects input, residual, scale and bias");
  }
  Node add(node.name() + "::__shape_add", "Add",
           {node.inputs()[0], node.inputs()[1]}, {node.outputs()[0]});
  CUTRITON_RETURN_IF_ERROR(InferElementwiseBinary(graph, add));
  Node layer_norm(node.name() + "::__shape_layer_norm", "LayerNormalization",
                  {node.outputs()[0], node.inputs()[2], node.inputs()[3]},
                  node.outputs());
  layer_norm.SetAttribute("axis", GetIntAttribute(node, "axis", -1));
  return InferLayerNormalization(graph, layer_norm);
}

Status InferGlobalAveragePool(Graph* graph, const Node& node) {
  const auto* input = graph->FindValue(node.inputs().front());
  if (input == nullptr || input->tensor.shape.size() != 4) {
    return Status::ShapeError("GlobalAveragePool 期望 NCHW 输入");
  }
  TensorDesc desc({input->tensor.shape[0], input->tensor.shape[1], 1, 1},
                  input->tensor.dtype, input->tensor.device_type,
                  input->tensor.device_id, input->tensor.layout);
  return graph->SetValueDesc(node.outputs()[0], std::move(desc));
}

class TopologicalSortPass final : public GraphPass {
 public:
  const char* name() const override { return "topological_sort"; }
  Status Run(Graph* graph) const override { return graph->TopologicalSort(); }
};

class ShapeInferencePass final : public GraphPass {
 public:
  const char* name() const override { return "shape_inference"; }

  Status Run(Graph* graph) const override {
    CUTRITON_RETURN_IF_ERROR(RegisterBuiltinOpSchemas());
    for (const auto& node : graph->nodes()) {
      const auto schema = OpSchemaRegistry::Global().Find(node.op_type());
      if (schema == nullptr) {
        return Status::Unsupported("形状推导暂不支持算子: " +
                                   node.op_type());
      }
      if (node.inputs().size() < schema->min_inputs ||
          node.inputs().size() > schema->max_inputs ||
          node.outputs().size() != schema->outputs) {
        return Status::ShapeError("节点输入输出数量不符合 OpSchema: " +
                                  node.name());
      }
      CUTRITON_RETURN_IF_ERROR(schema->infer(graph, node));
    }
    return Status::OK();
  }
};

class ConstantFoldingPass final : public GraphPass {
 public:
  const char* name() const override { return "constant_folding"; }
  Status Run(Graph*) const override {
    // V1 只在图边界跟踪常量 Value，不持有真实权重数据。
    // 保留这个 Pass 是为了后续加入带数据的常量折叠时，不需要改变编译流水线。
    return Status::OK();
  }
};

//如果一个节点的输出最终能够影响 Graph 的输出，这个节点就保留；否则删除
class DeadNodeEliminationPass final : public GraphPass {
 public:
  const char* name() const override { return "dead_node_elimination"; }

  Status Run(Graph* graph) const override {
    std::unordered_set<std::string> live_values(graph->outputs().begin(),
                                                graph->outputs().end());
    std::unordered_set<int> live_nodes;
    const auto& nodes = graph->nodes();
    //从后往前遍历
    for (auto it = nodes.rbegin(); it != nodes.rend(); ++it) {
      bool needed = false;
      for (const auto& output : it->outputs()) {
        if (live_values.find(output) != live_values.end()) {
          needed = true;//需要
          break;
        }
      }
      if (!needed) {
        continue;
      }
      live_nodes.insert(it->id());
      for (const auto& input : it->inputs()) {
        live_values.insert(input);
      }
    }

    std::vector<int> remove;
    //删除
    for (const auto& node : graph->nodes()) {
      if (live_nodes.find(node.id()) == live_nodes.end()) {
        remove.push_back(node.id());
      }
    }
    return graph->RemoveNodes(remove);
  }
};

class ConvBatchNormReluFusionPass final : public GraphPass {
 public:
  const char* name() const override { return "conv_batchnorm_relu_fusion"; }

  Status Run(Graph* graph) const override {
    bool changed = true;
    while (changed) {
      changed = false;
      const auto consumers = graph->BuildConsumerMap();
      int bn_id = -1;
      int relu_id = -1;
      for (auto& conv : graph->mutable_nodes()) {
        if (conv.op_type() != "Conv" || conv.outputs().size() != 1) {
          continue;
        }
        auto conv_consumers_it = consumers.find(conv.outputs()[0]);
        if (conv_consumers_it == consumers.end() ||
            conv_consumers_it->second.size() != 1) {
          continue;
        }
        bn_id = conv_consumers_it->second.front();
        if (bn_id < 0 || bn_id >= static_cast<int>(graph->nodes().size())) {
          continue;
        }
        const Node& bn = graph->nodes()[bn_id];
        if (bn.op_type() != "BatchNormalization" || bn.outputs().size() != 1) {
          continue;
        }
        bool fuse_relu = false;
        auto bn_consumers_it = consumers.find(bn.outputs()[0]);
        if (bn_consumers_it != consumers.end() &&
            bn_consumers_it->second.size() == 1) {
          relu_id = bn_consumers_it->second.front();
          if (relu_id >= 0 &&
              relu_id < static_cast<int>(graph->nodes().size())) {
            const Node& relu = graph->nodes()[relu_id];
            fuse_relu = relu.op_type() == "Relu" && !relu.outputs().empty();
          }
        }

        std::vector<std::string> fused_inputs = conv.inputs();
        for (std::size_t i = 1; i < bn.inputs().size(); ++i) {
          fused_inputs.push_back(bn.inputs()[i]);
        }
        conv.set_name(conv.name() + (fuse_relu ? "_bn_relu" : "_bn"));
        conv.set_op_type(fuse_relu ? "FusedConvBatchNormRelu"
                                   : "FusedConvBatchNorm");
        conv.set_inputs(std::move(fused_inputs));
        conv.set_outputs(fuse_relu ? graph->nodes()[relu_id].outputs()
                                   : bn.outputs());
        conv.SetAttribute("fused_batchnorm_relu", fuse_relu);
        conv.SetAttribute("batchnorm_epsilon",
                          bn.GetAttribute<double>("epsilon").value_or(1e-5));
        std::vector<int> remove{bn_id};
        if (fuse_relu) {
          remove.push_back(relu_id);
        }
        CUTRITON_RETURN_IF_ERROR(graph->RemoveNodes(remove));
        changed = true;
        break;
      }
    }
    return Status::OK();
  }
};

class AddReluFusionPass final : public GraphPass {
 public:
  const char* name() const override { return "add_relu_fusion"; }

  Status Run(Graph* graph) const override {
    bool changed = true;
    while (changed) {
      changed = false;
      const auto consumers = graph->BuildConsumerMap();
      for (auto& add : graph->mutable_nodes()) {
        if (add.op_type() != "Add" || add.outputs().size() != 1) {
          continue;
        }
        const auto it = consumers.find(add.outputs()[0]);
        if (it == consumers.end() || it->second.size() != 1) {
          continue;
        }
        const int relu_id = it->second.front();
        if (relu_id < 0 ||
            relu_id >= static_cast<int>(graph->nodes().size())) {
          continue;
        }
        const Node& relu = graph->nodes()[relu_id];
        if (relu.op_type() != "Relu" || relu.outputs().size() != 1) {
          continue;
        }
        add.set_name(add.name() + "_relu");
        add.set_op_type("AddRelu");
        add.set_outputs(relu.outputs());
        CUTRITON_RETURN_IF_ERROR(graph->RemoveNodes({relu_id}));
        changed = true;
        break;
      }
    }
    return Status::OK();
  }
};

class GemmGeluFusionPass final : public GraphPass {
 public:
  const char* name() const override { return "gemm_gelu_fusion"; }

  Status Run(Graph* graph) const override {
    bool changed = true;
    while (changed) {
      changed = false;
      const auto consumers = graph->BuildConsumerMap();
      for (auto& gemm : graph->mutable_nodes()) {
        if (gemm.op_type() != "Gemm" || gemm.outputs().size() != 1 ||
            std::find(graph->outputs().begin(), graph->outputs().end(),
                      gemm.outputs()[0]) != graph->outputs().end()) {
          continue;
        }
        const auto found = consumers.find(gemm.outputs()[0]);
        if (found == consumers.end() || found->second.size() != 1) continue;
        const int gelu_id = found->second.front();
        if (gelu_id < 0 || gelu_id >= static_cast<int>(graph->nodes().size())) continue;
        const Node& gelu = graph->nodes()[gelu_id];
        if (gelu.op_type() != "Gelu" || gelu.inputs().size() != 1 ||
            gelu.outputs().size() != 1) {
          continue;
        }
        gemm.set_name(gemm.name() + "_gelu");
        gemm.set_op_type("GemmGelu");
        gemm.set_outputs(gelu.outputs());
        CUTRITON_RETURN_IF_ERROR(graph->RemoveNodes({gelu_id}));
        changed = true;
        break;
      }
    }
    return Status::OK();
  }
};

class SkipLayerNormalizationFusionPass final : public GraphPass {
 public:
  const char* name() const override { return "skip_layer_norm_fusion"; }

  Status Run(Graph* graph) const override {
    bool changed = true;
    while (changed) {
      changed = false;
      const auto consumers = graph->BuildConsumerMap();
      for (auto& add : graph->mutable_nodes()) {
        if (add.op_type() != "Add" || add.outputs().size() != 1 ||
            std::find(graph->outputs().begin(), graph->outputs().end(),
                      add.outputs()[0]) != graph->outputs().end()) {
          continue;
        }
        const auto found = consumers.find(add.outputs()[0]);
        if (found == consumers.end() || found->second.size() != 1) continue;
        const int norm_id = found->second.front();
        if (norm_id < 0 || norm_id >= static_cast<int>(graph->nodes().size())) continue;
        const Node& norm = graph->nodes()[norm_id];
        if (norm.op_type() != "LayerNormalization" ||
            norm.inputs().size() != 3 || norm.outputs().size() != 1 ||
            norm.inputs()[0] != add.outputs()[0]) {
          continue;
        }
        std::vector<std::string> inputs = add.inputs();
        inputs.push_back(norm.inputs()[1]);
        inputs.push_back(norm.inputs()[2]);
        add.set_name(add.name() + "_layer_norm");
        add.set_op_type("SkipLayerNormalization");
        add.set_inputs(std::move(inputs));
        add.set_outputs(norm.outputs());
        add.SetAttribute("axis", GetIntAttribute(norm, "axis", -1));
        add.SetAttribute("epsilon",
                         norm.GetAttribute<double>("epsilon").value_or(1e-5));
        CUTRITON_RETURN_IF_ERROR(graph->RemoveNodes({norm_id}));
        changed = true;
        break;
      }
    }
    return Status::OK();
  }
};

class FlattenGemmNormalizationPass final : public GraphPass {
 public:
  const char* name() const override { return "flatten_gemm_normalization"; }

  Status Run(Graph* graph) const override {
    for (auto& node : graph->mutable_nodes()) {
      if (node.op_type() == "MatMul") {
        node.set_op_type("Gemm");
        node.SetAttribute("normalized_from_matmul", true);
      }
      if (node.op_type() == "Flatten" &&
          !node.GetAttribute<int64_t>("axis").has_value()) {
        node.SetAttribute("axis", static_cast<int64_t>(1));
      }
    }
    return Status::OK();
  }
};

class StaticShapeValidationPass final : public GraphPass {
 public:
  const char* name() const override { return "static_shape_validation"; }

  Status Run(Graph* graph) const override {
    for (const auto& [name, value] : graph->values()) {
      if (value.tensor.dtype == DataType::kUnknown && !value.is_constant) {
        continue;
      }
      if (!value.tensor.shape.empty()) {
        CUTRITON_RETURN_IF_ERROR(value.tensor.Validate());
      }
      (void)name;
    }
    for (const auto& output : graph->outputs()) {
      const auto* value = graph->FindValue(output);
      if (value == nullptr) {
        return Status::NotFound("图输出丢失: " + output);
      }
      CUTRITON_RETURN_IF_ERROR(value->tensor.Validate());
    }
    return Status::OK();
  }
};

}  // 匿名命名空间

Status RegisterBuiltinOpSchemas() {
  auto& registry = OpSchemaRegistry::Global();
  auto add = [&](OpSchema schema) -> Status {
    const Status status = registry.Register(std::move(schema));
    return status.code() == ErrorCode::kAlreadyExists ? Status::OK() : status;
  };
  const ShapeInferenceFunction same_as_input =
      [](Graph* graph, const Node& node) {
        return SetOutputLikeInput(graph, node);
      };
  const ShapeInferenceFunction conv = [](Graph* graph, const Node& node) {
    return InferConvLike(graph, node);
  };
  CUTRITON_RETURN_IF_ERROR(add(OpSchema{"Conv", 2, 2, 1, conv}));
  CUTRITON_RETURN_IF_ERROR(
      add(OpSchema{"FusedConvBatchNorm", 6, 6, 1, conv}));
  CUTRITON_RETURN_IF_ERROR(
      add(OpSchema{"FusedConvBatchNormRelu", 6, 6, 1, conv}));
  CUTRITON_RETURN_IF_ERROR(add(OpSchema{
      "BatchNormalization", 5, 5, 1, same_as_input}));
  for (const auto& op : {"Relu", "Gelu", "Softmax"}) {
    CUTRITON_RETURN_IF_ERROR(add(OpSchema{op, 1, 1, 1, same_as_input}));
  }
  CUTRITON_RETURN_IF_ERROR(add(OpSchema{
      "LayerNormalization", 2, 3, 1,
      [](Graph* graph, const Node& node) {
        return InferLayerNormalization(graph, node);
      }}));
  const ShapeInferenceFunction binary =
      [](Graph* graph, const Node& node) {
        return InferElementwiseBinary(graph, node);
      };
  CUTRITON_RETURN_IF_ERROR(add(OpSchema{"Add", 2, 2, 1, binary}));
  CUTRITON_RETURN_IF_ERROR(add(OpSchema{"AddRelu", 2, 2, 1, binary}));
  CUTRITON_RETURN_IF_ERROR(add(OpSchema{
      "Flatten", 1, 1, 1,
      [](Graph* graph, const Node& node) { return InferFlatten(graph, node); }}));
  const ShapeInferenceFunction gemm = [](Graph* graph, const Node& node) {
    return InferGemm(graph, node);
  };
  CUTRITON_RETURN_IF_ERROR(add(OpSchema{"Gemm", 2, 3, 1, gemm}));
  CUTRITON_RETURN_IF_ERROR(add(OpSchema{"GemmGelu", 2, 3, 1, gemm}));
  CUTRITON_RETURN_IF_ERROR(add(OpSchema{"MatMul", 2, 2, 1, gemm}));
  CUTRITON_RETURN_IF_ERROR(add(OpSchema{
      "SkipLayerNormalization", 4, 4, 1,
      [](Graph* graph, const Node& node) {
        return InferSkipLayerNormalization(graph, node);
      }}));
  const ShapeInferenceFunction pool = [](Graph* graph, const Node& node) {
    return InferPool(graph, node);
  };
  CUTRITON_RETURN_IF_ERROR(add(OpSchema{"MaxPool", 1, 1, 1, pool}));
  CUTRITON_RETURN_IF_ERROR(add(OpSchema{"AveragePool", 1, 1, 1, pool}));
  CUTRITON_RETURN_IF_ERROR(add(OpSchema{
      "GlobalAveragePool", 1, 1, 1,
      [](Graph* graph, const Node& node) {
        return InferGlobalAveragePool(graph, node);
      }}));
  CUTRITON_RETURN_IF_ERROR(add(OpSchema{
      "Constant", 0, 0, 1,
      [](Graph*, const Node&) { return Status::OK(); }}));
  return Status::OK();
}

void PassManager::Add(std::unique_ptr<GraphPass> pass) {
  passes_.push_back(std::move(pass));
}

Status PassManager::Run(Graph* graph) const {
  executed_passes_.clear();
  for (const auto& pass : passes_) {
    CUTRITON_RETURN_IF_ERROR(pass->Run(graph));
    executed_passes_.push_back(pass->name());
  }
  return Status::OK();
}

std::unique_ptr<GraphPass> CreateTopologicalSortPass() {
  return std::make_unique<TopologicalSortPass>();
}

std::unique_ptr<GraphPass> CreateShapeInferencePass() {
  return std::make_unique<ShapeInferencePass>();
}

std::unique_ptr<GraphPass> CreateConstantFoldingPass() {
  return std::make_unique<ConstantFoldingPass>();
}

std::unique_ptr<GraphPass> CreateDeadNodeEliminationPass() {
  return std::make_unique<DeadNodeEliminationPass>();
}

std::unique_ptr<GraphPass> CreateConvBatchNormReluFusionPass() {
  return std::make_unique<ConvBatchNormReluFusionPass>();
}

std::unique_ptr<GraphPass> CreateAddReluFusionPass() {
  return std::make_unique<AddReluFusionPass>();
}

std::unique_ptr<GraphPass> CreateGemmGeluFusionPass() {
  return std::make_unique<GemmGeluFusionPass>();
}

std::unique_ptr<GraphPass> CreateSkipLayerNormalizationFusionPass() {
  return std::make_unique<SkipLayerNormalizationFusionPass>();
}

std::unique_ptr<GraphPass> CreateFlattenGemmNormalizationPass() {
  return std::make_unique<FlattenGemmNormalizationPass>();
}

std::unique_ptr<GraphPass> CreateStaticShapeValidationPass() {
  return std::make_unique<StaticShapeValidationPass>();
}

PassManager CreateDefaultCompilePasses(bool enable_transformer_fusions) {
  PassManager manager;
  manager.Add(CreateTopologicalSortPass());
  manager.Add(CreateShapeInferencePass());
  manager.Add(CreateConstantFoldingPass());
  manager.Add(CreateDeadNodeEliminationPass());
  manager.Add(CreateConvBatchNormReluFusionPass());
  manager.Add(CreateAddReluFusionPass());
  if (enable_transformer_fusions) {
    manager.Add(CreateGemmGeluFusionPass());
    manager.Add(CreateSkipLayerNormalizationFusionPass());
  }
  manager.Add(CreateTopologicalSortPass());
  manager.Add(CreateShapeInferencePass());
  manager.Add(CreateFlattenGemmNormalizationPass());
  manager.Add(CreateDeadNodeEliminationPass());
  manager.Add(CreateStaticShapeValidationPass());
  return manager;
}

}  // 命名空间 cutriton
