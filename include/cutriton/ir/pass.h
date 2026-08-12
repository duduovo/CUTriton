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
//把残差分支末尾的 Add、Relu 合并成一个节点
std::unique_ptr<GraphPass> CreateAddReluFusionPass();
/// 创建 Gemm + Gelu -> GemmGelu 模式融合 Pass。
/// 仅当 Gemm 输出只有目标 Gelu 一个消费者且不是图输出时才执行融合。
/// \return 用于执行 GemmGelu 模式匹配与图改写的 GraphPass。
std::unique_ptr<GraphPass> CreateGemmGeluFusionPass();
/// 创建 Add + LayerNormalization -> SkipLayerNormalization 模式融合 Pass。
/// 仅融合满足单消费者等安全约束的残差归一化子图，避免改变图的原有语义。
/// \return 用于执行 SkipLayerNormalization 模式匹配与图改写的 GraphPass。
std::unique_ptr<GraphPass> CreateSkipLayerNormalizationFusionPass();
//把多种相似写法转换成项目内部统一的写法
std::unique_ptr<GraphPass> CreateFlattenGemmNormalizationPass();
//检查最终 Tensor 描述是否合法
std::unique_ptr<GraphPass> CreateStaticShapeValidationPass();

/// 组装 Compiler 使用的默认优化流水线。
/// \param enable_transformer_fusions 是否加入 Transformer 融合 Pass；关闭后可用于
/// 正确性调试以及融合前后的性能基准对比。
/// \return 按默认执行顺序配置完成的 PassManager。
PassManager CreateDefaultCompilePasses(bool enable_transformer_fusions = true);

}  // 命名空间 cutriton
