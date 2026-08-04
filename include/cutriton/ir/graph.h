#pragma once

#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

#include "cutriton/core/tensor.h"

/*
计算图
    参数：数据Value，节点Node
    Graph 负责将两者整合
    只描述怎么计算，并没有真正执行计算。
*/

namespace cutriton {

//算子的额外参数 e.g. strides,pads...统一名称为<Attribute>
using Attribute = std::variant<int64_t, double, bool, std::string,
                               std::vector<int64_t>, std::vector<double>>;

//数据描述:描述计算图中的一份数据
struct ValueDesc {
  std::string name;
  TensorDesc tensor;
  bool is_constant{false};
};

//计算步骤。Node表示一个算子
class Node {
 public:
  Node() = default;
  //Node=名称+算子类型+输入+输出 
  Node(std::string name, std::string op_type, std::vector<std::string> inputs,
       std::vector<std::string> outputs)
      : name_(std::move(name)),
        op_type_(std::move(op_type)),
        inputs_(std::move(inputs)),
        outputs_(std::move(outputs)) {}
  //读取Node信息
  int id() const { return id_; }
  const std::string& name() const { return name_; }
  const std::string& op_type() const { return op_type_; }
  const std::vector<std::string>& inputs() const { return inputs_; }
  const std::vector<std::string>& outputs() const { return outputs_; }
  const std::unordered_map<std::string, Attribute>& attributes() const {
    return attributes_;
  }
  //修改Node信息
  void set_name(std::string name) { name_ = std::move(name); }
  void set_op_type(std::string op_type) { op_type_ = std::move(op_type); }
  void set_inputs(std::vector<std::string> inputs) {
    inputs_ = std::move(inputs);
  }
  void set_outputs(std::vector<std::string> outputs) {
    outputs_ = std::move(outputs);
  }
  //设置算子属性
  void SetAttribute(std::string key, Attribute value) {
    attributes_[std::move(key)] = std::move(value);
  }
  //读取算子属性
  template <typename T>
  std::optional<T> GetAttribute(const std::string& key) const {
    auto it = attributes_.find(key);
    if (it == attributes_.end()) {
      return std::nullopt;
    }
    if (const auto* value = std::get_if<T>(&it->second)) {
      return *value;
    }
    return std::nullopt;
  }

 private:
  friend class Graph;
  //设置Node_ID,只有Graph可以修改，全局统一管理
  void set_id(int id) { id_ = id; }

  int id_{-1};//节点编号
  std::string name_;//节点名称
  std::string op_type_;//算子类型
  std::vector<std::string> inputs_;//输入名称
  std::vector<std::string> outputs_;//输出名称
  std::unordered_map<std::string, Attribute> attributes_;//算子属性 e.g.{"strides":{2,2},"pads":{1,1,1,1}...}
};

//完整计算图
class Graph {
 public:
  //添加数据描述
  Status AddValue(ValueDesc value);
  //添加计算图输入
  Status AddInput(std::string name, TensorDesc desc);
  //指定计算图输出
  Status AddOutput(std::string name);
  //添加计算节点
  Status AddNode(Node node);
  //只读查看
  const std::vector<Node>& nodes() const { return nodes_; }
  //允许修改Node
  std::vector<Node>& mutable_nodes() { return nodes_; }
  const std::unordered_map<std::string, ValueDesc>& values() const {
    return values_;
  }
  const std::vector<std::string>& inputs() const { return inputs_; }
  const std::vector<std::string>& outputs() const { return outputs_; }
  
  //查找Value
  const ValueDesc* FindValue(const std::string& name) const;
  //修改查找
  ValueDesc* MutableValue(const std::string& name);
  //更新Tensor描述
  Status SetValueDesc(const std::string& name, TensorDesc desc);
  //确保Tensor存在，不存在则自己创建
  Status EnsureValue(std::string name, TensorDesc desc = {});
  //拓扑排序
  Status TopologicalSort();
  //删除节点，删除后调用 RenumberNodes() 
  Status RemoveNodes(const std::vector<int>& node_ids);
  //创建生产者map {value名称:生成这个value的节点ID...}
  std::unordered_map<std::string, int> BuildProducerMap() const;
  //创建消费者map {value名称:使用这个value的节点ID...}
  std::unordered_map<std::string, std::vector<int>> BuildConsumerMap() const;

 private:
  //重新编号
  void RenumberNodes();

  std::vector<Node> nodes_;//所有计算节点 node_name
  std::unordered_map<std::string, ValueDesc> values_;//所有数据描述
  std::vector<std::string> inputs_;//整张图输入
  std::vector<std::string> outputs_;//整张图输出
};

//模型
class Model {
 public:
  Graph& graph() { return graph_; }
  const Graph& graph() const { return graph_; }

 private:
  Graph graph_;
};
//读取整数数组属性
std::vector<int64_t> GetIntListAttribute(const Node& node,
                                         const std::string& key,
                                         std::vector<int64_t> fallback);
                            
//读取单个整数
int64_t GetIntAttribute(const Node& node, const std::string& key,
                        int64_t fallback);

}  // 命名空间 cutriton
