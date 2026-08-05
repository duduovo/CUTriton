#include "cutriton/ir/op_schema.h"

#include <algorithm>
#include <mutex>
#include <utility>

namespace cutriton {

OpSchemaRegistry& OpSchemaRegistry::Global() {
  static OpSchemaRegistry registry;
  return registry;
}

Status OpSchemaRegistry::Register(OpSchema schema) {
  if (schema.op_type.empty() || !schema.infer ||
      schema.min_inputs > schema.max_inputs || schema.outputs == 0) {
    return Status::InvalidArgument("OpSchema is incomplete");
  }
  std::unique_lock<std::shared_mutex> lock(mutex_);
  const std::string op_type = schema.op_type;
  if (!schemas_.emplace(op_type,
                        std::make_shared<const OpSchema>(std::move(schema))).second) {
    return Status::AlreadyExists("OpSchema is already registered");
  }
  return Status::OK();
}

std::shared_ptr<const OpSchema> OpSchemaRegistry::Find(
    const std::string& op_type) const {
  std::shared_lock<std::shared_mutex> lock(mutex_);
  const auto found = schemas_.find(op_type);
  return found == schemas_.end() ? nullptr : found->second;
}

std::vector<std::string> OpSchemaRegistry::AvailableOps() const {
  std::shared_lock<std::shared_mutex> lock(mutex_);
  std::vector<std::string> result;
  for (const auto& entry : schemas_) result.push_back(entry.first);
  std::sort(result.begin(), result.end());
  return result;
}

void OpSchemaRegistry::ClearForTests() {
  std::unique_lock<std::shared_mutex> lock(mutex_);
  schemas_.clear();
}

}  // namespace cutriton
