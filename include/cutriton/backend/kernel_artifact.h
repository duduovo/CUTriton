#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <unordered_map>
#include <vector>

#include "cutriton/core/status.h"

namespace cutriton {

enum class KernelArgumentSourceKind {
  kInput,
  kOutput,
  kInputDim,
  kInputDimProduct,
  kOutputDim,
  kOutputNumElements,
  kAttributeInt64,
  kAttributeFloat64,
  kLiteralInt32,
  kRuntimeReserved,
};

struct KernelArgumentSource {
  KernelArgumentSourceKind kind{KernelArgumentSourceKind::kRuntimeReserved};
  int index{0};
  int axis{0};
  int begin_axis{0};
  std::string attribute_name;
  std::string alternate_attribute_name;
  int attribute_index{0};
  int64_t default_int{0};
  double default_float{0.0};
};

struct KernelArgumentSpec {
  std::string name;
  std::string type;
  KernelArgumentSource source;
};

enum class KernelConstraintKind {
  kAttributeInt64Equals,
  kInputCountEquals,
  kTensorRankEquals,
};

// A deliberately small constraint language. Kernel Packs cannot inject code;
// every condition is parsed once and evaluated by the C++ catalog.
struct KernelConstraint {
  KernelConstraintKind kind{KernelConstraintKind::kInputCountEquals};
  std::string attribute_name;
  bool tensor_is_output{false};
  int tensor_index{0};
  int64_t expected{0};
  int64_t default_value{0};
};

struct KernelArtifact {
  std::string pack_name;
  int pack_version{0};
  std::string generator_version;
  std::string triton_version;
  int abi_schema_version{0};
  std::string kernel_id;
  std::string op_type;
  int kernel_version{0};
  std::string variant_id;
  bool is_default{false};
  std::string symbol;
  std::filesystem::path binary_path;
  std::string binary_format;
  std::string sha256;
  std::string dtype;
  std::string layout;
  int min_compute_capability{80};
  int num_warps{4};
  int num_stages{1};
  int shared_memory_bytes{0};
  int grid_divisor{1};
  std::vector<KernelArgumentSpec> arguments;
  std::vector<KernelConstraint> constraints;
  std::string identity() const {
    return pack_name + ":" + kernel_id + ":" + variant_id;
  }
};

// Immutable, validated view over one or more AOT Kernel Packs.
class ArtifactRepository {
 public:
  Status Load(const std::vector<std::string>& paths);
  const std::vector<KernelArtifact>* Find(const std::string& op_type) const;
  const std::vector<KernelArtifact>& artifacts() const { return artifacts_; }
  const std::string& repository_hash() const { return repository_hash_; }

 private:
  std::vector<KernelArtifact> artifacts_;
  std::unordered_map<std::string, std::vector<KernelArtifact>> by_op_type_;
  std::string repository_hash_;
};

// Lightweight query layer over an immutable repository. Semantic Node
// constraints are evaluated by Backend::Lower; this catalog performs the
// artifact-level op/dtype/layout/SM filtering shared by all lowerings.
class KernelCatalog {
 public:
  explicit KernelCatalog(const ArtifactRepository* repository)
      : repository_(repository) {}
  std::vector<const KernelArtifact*> Query(const std::string& op_type,
                                           const std::string& dtype,
                                           const std::string& layout,
                                           int compute_capability) const;

 private:
  const ArtifactRepository* repository_{nullptr};
};

}  // namespace cutriton
