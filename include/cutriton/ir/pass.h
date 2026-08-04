#pragma once

#include <memory>
#include <string>
#include <vector>

#include "cutriton/core/status.h"
#include "cutriton/ir/graph.h"

//对计算图执行一次检查、分析或修改的处理步骤
namespace cutriton {

//所有Pass共同规则
class GraphPass {
 public:
  virtual ~GraphPass() = default;
  virtual const char* name() const = 0;//pass名称
  virtual Status Run(Graph* graph) const = 0;//pass处理
};

//Pass流水线管理
class PassManager {
 public:
  //添加Pass
  void Add(std::unique_ptr<GraphPass> pass);
  //执行Pass
  Status Run(Graph* graph) const;
  //查看执行通过的Pass
  const std::vector<std::string>& executed_passes() const {
    return executed_passes_;
  }

 private:
  std::vector<std::unique_ptr<GraphPass>> passes_;//等待执行的Pass
  mutable std::vector<std::string> executed_passes_;//已经执行的Pass
};
//1.拓扑排序/形状推导-->2.常量折叠/删除无用节点-->3.算子融合
//                  -->4.再次拓扑排序/再次形状推导-->5.Flatten/Gemm 规范化/再次删除无用节点/静态形状验证
//按依赖关系重新排列节点
std::unique_ptr<GraphPass> CreateTopologicalSortPass();
//根据输入形状和算子参数，计算输出 Tensor 的形状
std::unique_ptr<GraphPass> CreateShapeInferencePass();
//如果一个计算的输入都是常量，就提前计算结果
std::unique_ptr<GraphPass> CreateConstantFoldingPass();
//删除不会影响最终输出的节点
std::unique_ptr<GraphPass> CreateDeadNodeEliminationPass();
//把连续的 Conv、BatchNormalization、ReLU 合并成一个节点
std::unique_ptr<GraphPass> CreateConvBatchNormReluFusionPass();
//把多种相似写法转换成项目内部统一的写法
std::unique_ptr<GraphPass> CreateFlattenGemmNormalizationPass();
//检查最终 Tensor 描述是否合法
std::unique_ptr<GraphPass> CreateStaticShapeValidationPass();

PassManager CreateDefaultCompilePasses();

}  // 命名空间 cutriton
