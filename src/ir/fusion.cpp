#include "cutriton/ir/fusion.h"

#include <mutex>
#include <utility>

namespace cutriton {

FusionRegistry& FusionRegistry::Global() {
  static FusionRegistry registry;
  return registry;
}

Status FusionRegistry::Register(FusionAlternative alternative) {
  if (alternative.fused_op_type.empty() || alternative.candidate_id.empty() ||
      alternative.decomposed_ops.empty()) {
    return Status::InvalidArgument("FusionAlternative is incomplete");
  }
  std::unique_lock<std::shared_mutex> lock(mutex_);
  auto& entries = alternatives_[alternative.fused_op_type];
  for (const auto& entry : entries) {
    if (entry.candidate_id == alternative.candidate_id) {
      return Status::AlreadyExists("Fusion alternative already exists");
    }
  }
  entries.push_back(std::move(alternative));
  return Status::OK();
}

std::vector<FusionAlternative> FusionRegistry::Find(
    const std::string& fused_op_type) const {
  std::shared_lock<std::shared_mutex> lock(mutex_);
  const auto found = alternatives_.find(fused_op_type);
  return found == alternatives_.end() ? std::vector<FusionAlternative>{}
                                      : found->second;
}

void FusionRegistry::ClearForTests() {
  std::unique_lock<std::shared_mutex> lock(mutex_);
  alternatives_.clear();
}

Status RegisterBuiltinFusionAlternatives() {
  auto& registry = FusionRegistry::Global();
  for (FusionAlternative alternative : {
           FusionAlternative{"FusedConvBatchNormRelu", "unfused:conv_bn_relu",
                             {"Conv", "BatchNormalization", "Relu"}},
           FusionAlternative{"FusedConvBatchNorm", "unfused:conv_bn",
                             {"Conv", "BatchNormalization"}},
           FusionAlternative{"AddRelu", "unfused:add_relu", {"Add", "Relu"}},
           FusionAlternative{"GemmGelu", "unfused:gemm_gelu",
                             {"Gemm", "Gelu"}},
           FusionAlternative{"SkipLayerNormalization",
                             "unfused:add_layer_norm",
                             {"Add", "LayerNormalization"}},
       }) {
    const Status status = registry.Register(std::move(alternative));
    if (!status.ok() && status.code() != ErrorCode::kAlreadyExists) return status;
  }
  return Status::OK();
}

}  // namespace cutriton
