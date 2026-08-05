#pragma once

#include <memory>
#include <string>
#include <vector>

#include "cutriton/core/status.h"
#include "cutriton/ir/graph.h"
#include "cutriton/runtime/engine.h"
#include "cutriton/runtime/executable_plan.h"

//Compiler层把 Model 编译成 ExecutablePlan
//Model描述需要计算什么
//ExecutablePlan补充每一个节点由谁执行，按什么图执行，内存如何分配
namespace cutriton {

struct CompileOptions {
  std::string target{"cuda_triton"};//指定后端
  bool allow_cpu_fallback{false};//控制目标后端不支持某个算子时，能否退回 CPU
  bool enable_cuda_graph{true};//控制是否使用 CUDA Graph
  bool enable_profiling{true};//控制是否记录性能事件
  int device_id{0};
  std::string kernel_artifact_dir;
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
