#pragma once

#include <array>
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

enum class GridExpressionKind {
  kLiteral,            ///< 整数字面量，例如固定的 grid 维度 1。
  kInputDim,           ///< 读取指定输入 Tensor 某一轴的长度。
  kOutputDim,          ///< 读取指定输出 Tensor 某一轴的长度。
  kInputNumElements,  ///< 读取指定输入 Tensor 的元素总数。
  kOutputNumElements, ///< 读取指定输出 Tensor 的元素总数。
  kVariantMeta,       ///< 读取当前 Kernel variant 的 launch metadata。
  kCeilDiv,           ///< 对两个子表达式执行向上取整除法。
  kMultiply,          ///< 对两个子表达式执行乘法。
};

/// Kernel Pack ABI v2 使用的受限 launch-grid 表达式树。
///
/// Runtime 只解释此处定义的节点，不执行 manifest 中的任意代码。这样既能根据
/// Tensor shape 和 variant 配置动态计算 grid，也能在加载阶段拒绝未知表达式。
struct GridExpression {
  /// 当前表达式节点的类型。
  GridExpressionKind kind{GridExpressionKind::kLiteral};
  /// 字面量的值；仅在 kind == kLiteral 时有效。
  int64_t literal{1};
  /// 输入或输出 Tensor 的索引；仅供 Tensor 相关表达式使用。
  int tensor_index{0};
  /// Tensor 的维度索引；仅供 kInputDim 和 kOutputDim 使用。
  int axis{0};
  /// launch metadata 的键名；仅在 kind == kVariantMeta 时有效。
  std::string meta_name;
  /// 运算节点的子表达式；kCeilDiv 和 kMultiply 必须恰好包含两个操作数。
  std::vector<GridExpression> operands;
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
  /// ABI v2 的三维 launch grid 表达式，依次对应 CUDA grid 的 x、y、z 维。
  std::array<GridExpression, 3> launch_grid{};
  /// 当前 Kernel variant 的 launch 参数，例如 BLOCK_M、BLOCK_N 和 BLOCK_K。
  std::unordered_map<std::string, int64_t> launch_meta;
  /// Kernel Pack ABI v1 的兼容字段；新生成的 ABI v2 Pack 以 launch_grid 为准。
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
