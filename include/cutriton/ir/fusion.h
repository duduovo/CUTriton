#pragma once

#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "cutriton/core/status.h"

namespace cutriton {

struct FusionAlternative {
  std::string fused_op_type;
  std::string candidate_id;
  std::vector<std::string> decomposed_ops;
};

class FusionRegistry {
 public:
  static FusionRegistry& Global();
  Status Register(FusionAlternative alternative);
  std::vector<FusionAlternative> Find(const std::string& fused_op_type) const;
  void ClearForTests();

 private:
  mutable std::shared_mutex mutex_;
  std::unordered_map<std::string, std::vector<FusionAlternative>> alternatives_;
};

Status RegisterBuiltinFusionAlternatives();

}  // namespace cutriton
