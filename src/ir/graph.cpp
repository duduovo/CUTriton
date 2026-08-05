#include "cutriton/ir/graph.h"
#include "cutriton/core/buffer.h"

#include <algorithm>
#include <queue>
#include <set>
#include <unordered_set>

namespace cutriton {

Status Graph::AddValue(ValueDesc value) {
  if (value.name.empty()) {
    return Status::InvalidArgument("Value 名称不能为空");
  }
  auto [_, inserted] = values_.emplace(value.name, std::move(value));
  if (!inserted) {
    return Status::AlreadyExists("Value 已存在");
  }
  return Status::OK();
}

Status Graph::AddInput(std::string name, TensorDesc desc) {
  if (name.empty()) {
    return Status::InvalidArgument("输入名称不能为空");
  }
  if (values_.find(name) == values_.end()) {
    CUTRITON_RETURN_IF_ERROR(AddValue(ValueDesc{name, std::move(desc), false}));
  } else {
    CUTRITON_RETURN_IF_ERROR(SetValueDesc(name, std::move(desc)));
  }
  inputs_.push_back(std::move(name));
  return Status::OK();
}

Status Graph::AddOutput(std::string name) {
  if (values_.find(name) == values_.end()) {
    return Status::NotFound("图输出 Value 未定义: " + name);
  }
  outputs_.push_back(std::move(name));
  return Status::OK();
}

Status Graph::AddNode(Node node) {
  if (node.name().empty()) {
    return Status::InvalidArgument("节点名称不能为空");
  }
  if (node.op_type().empty()) {
    return Status::InvalidArgument("节点 op_type 不能为空");
  }
  for (const auto& output : node.outputs()) {
    CUTRITON_RETURN_IF_ERROR(EnsureValue(output));
  }
  for (const auto& input : node.inputs()) {
    CUTRITON_RETURN_IF_ERROR(EnsureValue(input));
  }
  node.set_id(static_cast<int>(nodes_.size()));
  nodes_.push_back(std::move(node));
  return Status::OK();
}

const ValueDesc* Graph::FindValue(const std::string& name) const {
  auto it = values_.find(name);
  return it == values_.end() ? nullptr : &it->second;
}

ValueDesc* Graph::MutableValue(const std::string& name) {
  auto it = values_.find(name);
  return it == values_.end() ? nullptr : &it->second;
}

Status Graph::SetValueDesc(const std::string& name, TensorDesc desc) {
  auto* value = MutableValue(name);
  if (value == nullptr) {
    return Status::NotFound("Value 未定义: " + name);
  }
  value->tensor = std::move(desc);
  return Status::OK();
}

Status Graph::EnsureValue(std::string name, TensorDesc desc) {
  if (name.empty()) {
    return Status::InvalidArgument("Value 名称不能为空");
  }
  if (values_.find(name) == values_.end()) {
    values_.emplace(name, ValueDesc{name, std::move(desc), false});
  }
  return Status::OK();
}

std::unordered_map<std::string, int> Graph::BuildProducerMap() const {
  std::unordered_map<std::string, int> producers;
  for (const auto& node : nodes_) {
    for (const auto& output : node.outputs()) {
      producers[output] = node.id();
    }
  }
  return producers;
}

std::unordered_map<std::string, std::vector<int>> Graph::BuildConsumerMap() const {
  std::unordered_map<std::string, std::vector<int>> consumers;
  for (const auto& node : nodes_) {
    for (const auto& input : node.inputs()) {
      consumers[input].push_back(node.id());
    }
  }
  return consumers;
}
//Kahn 算法
Status Graph::TopologicalSort() {
  const auto producers = BuildProducerMap();
  std::vector<int> indegree(nodes_.size(), 0);
  std::vector<std::vector<int>> outgoing(nodes_.size());
  for (const auto& node : nodes_) {
    std::set<int> unique_deps;//当前节点依赖的节点ID
    for (const auto& input : node.inputs()) {
      auto producer = producers.find(input);//找到生产input的节点ID
      if (producer != producers.end() && producer->second != node.id()) {
        unique_deps.insert(producer->second);
      }
    }
    indegree[node.id()] = static_cast<int>(unique_deps.size());
    //建立后续节点表
    for (int dep : unique_deps) {
      outgoing[dep].push_back(node.id());
    }
  }
  //找出最开始可以执行的节点
  std::queue<int> ready;
  for (std::size_t i = 0; i < indegree.size(); ++i) {
    if (indegree[i] == 0) {
      ready.push(static_cast<int>(i));
    }
  }

  std::vector<Node> sorted;
  sorted.reserve(nodes_.size());
  //只要还有可以执行的节点，就继续处理
  while (!ready.empty()) {
    const int id = ready.front();
    ready.pop();
    sorted.push_back(nodes_[id]);
    for (int next : outgoing[id]) {
      --indegree[next];
      if (indegree[next] == 0) {
        ready.push(next);
      }
    }
  }

  if (sorted.size() != nodes_.size()) {
    return Status::InvalidArgument("计算图存在环");
  }
  nodes_ = std::move(sorted);
  //重新排序节点ID
  RenumberNodes();
  return Status::OK();
}

Status Graph::RemoveNodes(const std::vector<int>& node_ids) {
  std::unordered_set<int> remove(node_ids.begin(), node_ids.end());
  std::vector<Node> kept;
  kept.reserve(nodes_.size());
  for (const auto& node : nodes_) {
    if (remove.find(node.id()) == remove.end()) {
      kept.push_back(node);
    }
  }
  nodes_ = std::move(kept);
  //删除后重新排序
  RenumberNodes();
  return Status::OK();
}

void Graph::RenumberNodes() {
  for (std::size_t i = 0; i < nodes_.size(); ++i) {
    nodes_[i].set_id(static_cast<int>(i));
  }
}

std::vector<int64_t> GetIntListAttribute(const Node& node,
                                         const std::string& key,
                                         std::vector<int64_t> fallback) {
  if (auto value = node.GetAttribute<std::vector<int64_t>>(key)) {
    return *value;
  }
  return fallback;
}

int64_t GetIntAttribute(const Node& node, const std::string& key,
                        int64_t fallback) {
  if (auto value = node.GetAttribute<int64_t>(key)) {
    return *value;
  }
  return fallback;
}

Status Model::AddConstant(std::string name, Tensor tensor) {
  if (name.empty()) {
    return Status::InvalidArgument("Constant name must not be empty");
  }
  CUTRITON_RETURN_IF_ERROR(tensor.desc().Validate());
  if (!tensor.defined() || tensor.buffer()->empty()) {
    return Status::InvalidArgument("Constant Tensor must have a Buffer: " + name);
  }
  if (tensor.buffer()->device_type() != DeviceType::kCPU ||
      tensor.desc().device_type != DeviceType::kCPU) {
    return Status::InvalidArgument("Model constants must use host memory: " + name);
  }
  if (tensor.byte_offset() > tensor.buffer()->size_bytes() ||
      tensor.desc().ByteSize() > tensor.buffer()->size_bytes() - tensor.byte_offset()) {
    return Status::InvalidArgument("Constant Buffer is too small: " + name);
  }
  if (constants_.find(name) != constants_.end()) {
    return Status::AlreadyExists("Constant already exists: " + name);
  }
  if (const auto* existing = graph_.FindValue(name); existing != nullptr) {
    if (existing->is_constant) {
      return Status::AlreadyExists("Constant already exists: " + name);
    }
    CUTRITON_RETURN_IF_ERROR(graph_.SetValueDesc(name, tensor.desc()));
    graph_.MutableValue(name)->is_constant = true;
  } else {
    CUTRITON_RETURN_IF_ERROR(
        graph_.AddValue(ValueDesc{name, tensor.desc(), true}));
  }
  constants_.emplace(std::move(name), std::move(tensor));
  return Status::OK();
}

}  // 命名空间 cutriton
