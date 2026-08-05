#pragma once

#include <functional>
#include <memory>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "cutriton/core/status.h"
#include "cutriton/ir/graph.h"

namespace cutriton {

using ShapeInferenceFunction = std::function<Status(Graph*, const Node&)>;

struct OpSchema {
  std::string op_type;
  std::size_t min_inputs{0};
  std::size_t max_inputs{0};
  std::size_t outputs{1};
  ShapeInferenceFunction infer;
};

class OpSchemaRegistry {
 public:
  static OpSchemaRegistry& Global();
  Status Register(OpSchema schema);
  std::shared_ptr<const OpSchema> Find(const std::string& op_type) const;
  std::vector<std::string> AvailableOps() const;
  void ClearForTests();

 private:
  mutable std::shared_mutex mutex_;
  std::unordered_map<std::string, std::shared_ptr<const OpSchema>> schemas_;
};

Status RegisterBuiltinOpSchemas();

}  // namespace cutriton
