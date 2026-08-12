#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "cutriton/core/status.h"
#include "cutriton/ir/graph.h"
#include "cutriton/runtime/engine.h"
#include "cutriton/runtime/executable_plan.h"
#include "cutriton/runtime/tuning.h"
#include "cutriton/runtime/shape_profile.h"

//Compiler层把 Model 编译成 ExecutablePlan
//Model描述需要计算什么
//ExecutablePlan补充每一个节点由谁执行，按什么图执行，内存如何分配
namespace cutriton {

struct CompileOptions {
  std::string target{"cuda_triton"};//指定后端
  bool allow_cpu_fallback{false};//控制目标后端不支持某个算子时，能否退回 CPU
  bool enable_cuda_graph{true};//控制是否使用 CUDA Graph
  bool enable_profiling{true};//控制是否记录性能事件
  bool enable_transformer_fusions{true};// 是否启用 Transformer 子图融合；关闭后保留原始算子，便于正确性定位和性能对比。
  int device_id{0};
  std::string kernel_artifact_dir;
  std::vector<std::string> kernel_artifact_paths;
  std::string tuning_cache_dir;
  TuningMode tuning_mode{TuningMode::kUseCache};
  std::vector<ShapeProfile> shape_profiles;
  std::size_t cuda_graph_cache_capacity{4};
  int tuning_warmup_iterations{5};
  int tuning_measurement_iterations{20};
};

class Compiler {
 public:
  //model:待编译的图模型  options:配置信息  plan:生成的可执行计划 
  //return Status::OK()
  Status Compile(const Model& model, const CompileOptions& options,
                 ExecutablePlan* plan) const;
};
//便利函数:把plan和BuildEngine组合起来,BuildEngine=compiler.Compile+make_unique<Engine>()
Status BuildEngine(const Model& model, const CompileOptions& options,
                   std::unique_ptr<Engine>* engine);

}  // 命名空间 cutriton
