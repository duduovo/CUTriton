#include "cutriton/runtime/memory_planner.h"

#include <algorithm>
#include <unordered_map>
#include <unordered_set>

namespace cutriton {
namespace {

constexpr std::size_t kWorkspaceAlignment = 256;

std::size_t AlignUp(std::size_t value) {
  return (value + kWorkspaceAlignment - 1) & ~(kWorkspaceAlignment - 1);
}

struct FreeBlock {
  std::size_t offset{0};
  std::size_t size{0};
};

struct LiveBlock {
  std::string value_name;
  std::size_t offset{0};
  std::size_t reserved_size{0};
};

}  // namespace

Status MemoryPlanner::Plan(const Graph& graph, MemoryPlan* plan) const {
  if (plan == nullptr) {
    return Status::InvalidArgument("MemoryPlan output pointer is null");
  }
  *plan = MemoryPlan{};

  std::unordered_set<std::string> external_values(graph.inputs().begin(),
                                                  graph.inputs().end());
  external_values.insert(graph.outputs().begin(), graph.outputs().end());

  std::unordered_map<std::string, std::string> aliases;
  for (const auto& node : graph.nodes()) {
    if (node.op_type() == "Flatten" && node.inputs().size() == 1 &&
        node.outputs().size() == 1) {
      aliases[node.outputs()[0]] = node.inputs()[0];
    }
  }

  std::unordered_map<std::string, int> last_use;
  for (std::size_t i = 0; i < graph.nodes().size(); ++i) {
    const auto& node = graph.nodes()[i];
    for (const auto& input : node.inputs()) {
      last_use[input] = static_cast<int>(i);
    }
    for (const auto& output : node.outputs()) {
      last_use.emplace(output, static_cast<int>(i));
    }
  }
  const int graph_end = static_cast<int>(graph.nodes().size());
  for (const auto& output : graph.outputs()) {
    last_use[output] = graph_end;
  }
  for (const auto& alias : aliases) {
    const auto alias_use = last_use.find(alias.first);
    if (alias_use != last_use.end()) {
      last_use[alias.second] =
          std::max(last_use[alias.second], alias_use->second);
    }
  }

  std::vector<FreeBlock> free_blocks;
  std::vector<LiveBlock> live_blocks;
  std::unordered_map<std::string, Allocation> assigned;
  std::size_t workspace_end = 0;

  auto release_dead = [&](int node_index) {
    std::vector<LiveBlock> still_live;
    for (const auto& block : live_blocks) {
      const auto it = last_use.find(block.value_name);
      if (it != last_use.end() && it->second < node_index) {
        free_blocks.push_back({block.offset, block.reserved_size});
      } else {
        still_live.push_back(block);
      }
    }
    live_blocks = std::move(still_live);
  };

  for (std::size_t i = 0; i < graph.nodes().size(); ++i) {
    release_dead(static_cast<int>(i));
    const auto& node = graph.nodes()[i];
    for (const auto& output : node.outputs()) {
      if (external_values.find(output) != external_values.end()) {
        continue;
      }
      const auto alias = aliases.find(output);
      if (alias != aliases.end()) {
        const auto source = assigned.find(alias->second);
        if (source == assigned.end() &&
            external_values.find(alias->second) == external_values.end()) {
          return Status::Internal("Flatten alias source has no allocation: " +
                                  alias->second);
        }
        Allocation allocation;
        if (source != assigned.end()) {
          allocation = source->second;
        } else {
          const auto* source_value = graph.FindValue(alias->second);
          if (source_value == nullptr) {
            return Status::Internal("Flatten alias source is unknown: " +
                                    alias->second);
          }
          allocation.size_bytes = source_value->tensor.ByteSize();
        }
        allocation.value_name = output;
        allocation.alias_of = alias->second;
        allocation.reused = true;
        const auto* value = graph.FindValue(output);
        if (value == nullptr || value->tensor.ByteSize() != allocation.size_bytes) {
          return Status::ShapeError("Flatten alias changes Tensor byte size: " +
                                    output);
        }
        plan->allocations.push_back(allocation);
        assigned.emplace(output, allocation);
        continue;
      }

      const auto* value = graph.FindValue(output);
      if (value == nullptr || value->is_constant) {
        continue;
      }
      const std::size_t size = value->tensor.ByteSize();
      if (size == 0) {
        return Status::ShapeError(
            "Cannot plan memory for a dynamically sized Value: " + output);
      }
      const std::size_t reserved_size = AlignUp(size);
      auto best = free_blocks.end();
      for (auto it = free_blocks.begin(); it != free_blocks.end(); ++it) {
        if (it->size >= reserved_size &&
            (best == free_blocks.end() || it->size < best->size)) {
          best = it;
        }
      }

      Allocation allocation;
      allocation.value_name = output;
      allocation.size_bytes = size;
      if (best != free_blocks.end()) {
        allocation.offset = best->offset;
        allocation.reused = true;
        if (best->size > reserved_size) {
          best->offset += reserved_size;
          best->size -= reserved_size;
        } else {
          free_blocks.erase(best);
        }
      } else {
        allocation.offset = AlignUp(workspace_end);
        workspace_end = allocation.offset + reserved_size;
      }
      plan->allocations.push_back(allocation);
      assigned.emplace(output, allocation);
      live_blocks.push_back(
          LiveBlock{output, allocation.offset, reserved_size});
    }
  }

  plan->workspace_size_bytes = AlignUp(workspace_end);
  return Status::OK();
}

}  // namespace cutriton
