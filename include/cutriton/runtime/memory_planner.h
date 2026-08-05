#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "cutriton/core/status.h"
#include "cutriton/ir/graph.h"

//负责规划中间张量如何共享一块workspace
namespace cutriton {
//中间变量内存安排
struct Allocation {
  std::string value_name;
  std::size_t size_bytes{0};
  std::size_t offset{0};
  bool reused{false};
  std::string alias_of;
};
struct MemoryPlan {
  std::size_t workspace_size_bytes{0};//最大分配内存大小
  std::vector<Allocation> allocations;
};

class MemoryPlanner {
 public:
  //1.计算最后使用位置 2.释放死亡张量 3.最佳适配复用 4.扩展workspace 
  Status Plan(const Graph& graph, MemoryPlan* plan) const;
};

}  // 命名空间 cutriton
