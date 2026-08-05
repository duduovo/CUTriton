#include "cutriton/backend/kernel_artifact.h"

#include <array>
#include <cctype>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <map>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <unordered_set>
#include <variant>

namespace cutriton {
namespace {

class JsonValue {
 public:
  using Object = std::map<std::string, JsonValue>;
  using Array = std::vector<JsonValue>;
  using Storage =
      std::variant<std::nullptr_t, bool, double, std::string, Object, Array>;
  explicit JsonValue(Storage storage = nullptr) : storage_(std::move(storage)) {}

  const Object& object() const { return std::get<Object>(storage_); }
  const Array& array() const { return std::get<Array>(storage_); }
  const std::string& string() const { return std::get<std::string>(storage_); }
  double number() const { return std::get<double>(storage_); }
  bool boolean() const { return std::get<bool>(storage_); }
  const JsonValue& at(const std::string& key) const {
    const auto it = object().find(key);
    if (it == object().end()) {
      throw std::runtime_error("missing JSON field: " + key);
    }
    return it->second;
  }
  const JsonValue* find(const std::string& key) const {
    const auto it = object().find(key);
    return it == object().end() ? nullptr : &it->second;
  }

 private:
  Storage storage_;
};

class JsonParser {
 public:
  explicit JsonParser(const std::string& source) : source_(source) {}
  JsonValue Parse() {
    JsonValue value = ParseValue();
    SkipWhitespace();
    if (position_ != source_.size()) {
      throw std::runtime_error("trailing JSON data");
    }
    return value;
  }

 private:
  void SkipWhitespace() {
    while (position_ < source_.size() &&
           std::isspace(static_cast<unsigned char>(source_[position_]))) {
      ++position_;
    }
  }
  char Take() {
    if (position_ >= source_.size()) throw std::runtime_error("unexpected EOF");
    return source_[position_++];
  }
  bool Consume(char value) {
    SkipWhitespace();
    if (position_ < source_.size() && source_[position_] == value) {
      ++position_;
      return true;
    }
    return false;
  }
  JsonValue ParseValue() {
    SkipWhitespace();
    if (position_ >= source_.size()) throw std::runtime_error("unexpected EOF");
    const char value = source_[position_];
    if (value == '{') return JsonValue(ParseObject());
    if (value == '[') return JsonValue(ParseArray());
    if (value == '"') return JsonValue(ParseString());
    if (value == '-' || std::isdigit(static_cast<unsigned char>(value))) {
      return JsonValue(ParseNumber());
    }
    if (source_.compare(position_, 4, "true") == 0) {
      position_ += 4;
      return JsonValue(true);
    }
    if (source_.compare(position_, 5, "false") == 0) {
      position_ += 5;
      return JsonValue(false);
    }
    if (source_.compare(position_, 4, "null") == 0) {
      position_ += 4;
      return JsonValue(nullptr);
    }
    throw std::runtime_error("invalid JSON value");
  }
  JsonValue::Object ParseObject() {
    if (!Consume('{')) throw std::runtime_error("expected object");
    JsonValue::Object result;
    if (Consume('}')) return result;
    do {
      SkipWhitespace();
      if (position_ >= source_.size() || source_[position_] != '"') {
        throw std::runtime_error("expected object key");
      }
      std::string key = ParseString();
      if (!Consume(':')) throw std::runtime_error("expected colon");
      if (!result.emplace(std::move(key), ParseValue()).second) {
        throw std::runtime_error("duplicate object key");
      }
    } while (Consume(','));
    if (!Consume('}')) throw std::runtime_error("expected object end");
    return result;
  }
  JsonValue::Array ParseArray() {
    if (!Consume('[')) throw std::runtime_error("expected array");
    JsonValue::Array result;
    if (Consume(']')) return result;
    do {
      result.push_back(ParseValue());
    } while (Consume(','));
    if (!Consume(']')) throw std::runtime_error("expected array end");
    return result;
  }
  std::string ParseString() {
    if (Take() != '"') throw std::runtime_error("expected string");
    std::string result;
    while (true) {
      const char value = Take();
      if (value == '"') return result;
      if (value != '\\') {
        result.push_back(value);
        continue;
      }
      const char escaped = Take();
      switch (escaped) {
        case '"': result.push_back('"'); break;
        case '\\': result.push_back('\\'); break;
        case '/': result.push_back('/'); break;
        case 'b': result.push_back('\b'); break;
        case 'f': result.push_back('\f'); break;
        case 'n': result.push_back('\n'); break;
        case 'r': result.push_back('\r'); break;
        case 't': result.push_back('\t'); break;
        default: throw std::runtime_error("unsupported JSON escape");
      }
    }
  }
  double ParseNumber() {
    const std::size_t begin = position_;
    if (source_[position_] == '-') ++position_;
    while (position_ < source_.size() &&
           std::isdigit(static_cast<unsigned char>(source_[position_]))) ++position_;
    if (position_ < source_.size() && source_[position_] == '.') {
      ++position_;
      while (position_ < source_.size() &&
             std::isdigit(static_cast<unsigned char>(source_[position_]))) ++position_;
    }
    if (position_ < source_.size() &&
        (source_[position_] == 'e' || source_[position_] == 'E')) {
      ++position_;
      if (position_ < source_.size() &&
          (source_[position_] == '+' || source_[position_] == '-')) ++position_;
      while (position_ < source_.size() &&
             std::isdigit(static_cast<unsigned char>(source_[position_]))) ++position_;
    }
    return std::stod(source_.substr(begin, position_ - begin));
  }

  const std::string& source_;
  std::size_t position_{0};
};

std::uint32_t RotateRight(std::uint32_t value, unsigned int count) {
  return (value >> count) | (value << (32U - count));
}

std::string Sha256(const std::string& input) {
  static constexpr std::array<std::uint32_t, 64> constants = {
      0x428a2f98U,0x71374491U,0xb5c0fbcfU,0xe9b5dba5U,0x3956c25bU,0x59f111f1U,0x923f82a4U,0xab1c5ed5U,
      0xd807aa98U,0x12835b01U,0x243185beU,0x550c7dc3U,0x72be5d74U,0x80deb1feU,0x9bdc06a7U,0xc19bf174U,
      0xe49b69c1U,0xefbe4786U,0x0fc19dc6U,0x240ca1ccU,0x2de92c6fU,0x4a7484aaU,0x5cb0a9dcU,0x76f988daU,
      0x983e5152U,0xa831c66dU,0xb00327c8U,0xbf597fc7U,0xc6e00bf3U,0xd5a79147U,0x06ca6351U,0x14292967U,
      0x27b70a85U,0x2e1b2138U,0x4d2c6dfcU,0x53380d13U,0x650a7354U,0x766a0abbU,0x81c2c92eU,0x92722c85U,
      0xa2bfe8a1U,0xa81a664bU,0xc24b8b70U,0xc76c51a3U,0xd192e819U,0xd6990624U,0xf40e3585U,0x106aa070U,
      0x19a4c116U,0x1e376c08U,0x2748774cU,0x34b0bcb5U,0x391c0cb3U,0x4ed8aa4aU,0x5b9cca4fU,0x682e6ff3U,
      0x748f82eeU,0x78a5636fU,0x84c87814U,0x8cc70208U,0x90befffaU,0xa4506cebU,0xbef9a3f7U,0xc67178f2U};
  std::vector<std::uint8_t> data(input.begin(), input.end());
  const std::uint64_t bits = static_cast<std::uint64_t>(data.size()) * 8U;
  data.push_back(0x80U);
  while (data.size() % 64U != 56U) data.push_back(0U);
  for (int shift = 56; shift >= 0; shift -= 8) data.push_back(static_cast<std::uint8_t>(bits >> shift));
  std::array<std::uint32_t, 8> hash = {0x6a09e667U,0xbb67ae85U,0x3c6ef372U,0xa54ff53aU,0x510e527fU,0x9b05688cU,0x1f83d9abU,0x5be0cd19U};
  for (std::size_t offset = 0; offset < data.size(); offset += 64U) {
    std::array<std::uint32_t, 64> words{};
    for (std::size_t i = 0; i < 16U; ++i) {
      const std::size_t base = offset + i * 4U;
      words[i] = (static_cast<std::uint32_t>(data[base]) << 24U) |
                 (static_cast<std::uint32_t>(data[base+1]) << 16U) |
                 (static_cast<std::uint32_t>(data[base+2]) << 8U) | data[base+3];
    }
    for (std::size_t i = 16; i < 64; ++i) {
      const auto s0=RotateRight(words[i-15],7)^RotateRight(words[i-15],18)^(words[i-15]>>3);
      const auto s1=RotateRight(words[i-2],17)^RotateRight(words[i-2],19)^(words[i-2]>>10);
      words[i]=words[i-16]+s0+words[i-7]+s1;
    }
    auto a=hash[0],b=hash[1],c=hash[2],d=hash[3],e=hash[4],f=hash[5],g=hash[6],h=hash[7];
    for (std::size_t i=0;i<64;++i) {
      const auto s1=RotateRight(e,6)^RotateRight(e,11)^RotateRight(e,25);
      const auto ch=(e&f)^((~e)&g);
      const auto t1=h+s1+ch+constants[i]+words[i];
      const auto s0=RotateRight(a,2)^RotateRight(a,13)^RotateRight(a,22);
      const auto maj=(a&b)^(a&c)^(b&c);
      const auto t2=s0+maj; h=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2;
    }
    hash[0]+=a;hash[1]+=b;hash[2]+=c;hash[3]+=d;hash[4]+=e;hash[5]+=f;hash[6]+=g;hash[7]+=h;
  }
  std::ostringstream output; output<<std::hex<<std::setfill('0');
  for (auto value:hash) output<<std::setw(8)<<value;
  return output.str();
}

int Integer(const JsonValue& value) { return static_cast<int>(value.number()); }

KernelArgumentSourceKind SourceKind(const std::string& value) {
  if (value == "input") return KernelArgumentSourceKind::kInput;
  if (value == "output") return KernelArgumentSourceKind::kOutput;
  if (value == "input_dim") return KernelArgumentSourceKind::kInputDim;
  if (value == "input_dim_product") return KernelArgumentSourceKind::kInputDimProduct;
  if (value == "output_dim") return KernelArgumentSourceKind::kOutputDim;
  if (value == "output_numel") return KernelArgumentSourceKind::kOutputNumElements;
  if (value == "attribute_i64") return KernelArgumentSourceKind::kAttributeInt64;
  if (value == "attribute_f64") return KernelArgumentSourceKind::kAttributeFloat64;
  if (value == "literal_i32") return KernelArgumentSourceKind::kLiteralInt32;
  if (value == "runtime_reserved") return KernelArgumentSourceKind::kRuntimeReserved;
  throw std::runtime_error("unsupported argument source: " + value);
}

int OptionalInt(const JsonValue& object, const std::string& key, int fallback) {
  const auto* value = object.find(key);
  return value == nullptr ? fallback : Integer(*value);
}

double OptionalDouble(const JsonValue& object, const std::string& key, double fallback) {
  const auto* value = object.find(key);
  return value == nullptr ? fallback : value->number();
}

}  // namespace

Status ArtifactRepository::Load(const std::vector<std::string>& paths) {
  artifacts_.clear();
  by_op_type_.clear();
  repository_hash_.clear();
  if (paths.empty()) return Status::InvalidArgument("kernel artifact paths are empty");
  std::string repository_material;
  try {
    for (const auto& item : paths) {
      std::filesystem::path root(item);
      const auto pack_path = std::filesystem::is_directory(root) ? root / "pack.json" : root;
      if (!std::filesystem::exists(pack_path)) {
        return Status::NotFound("Kernel Pack not found: " + pack_path.string());
      }
      root = pack_path.parent_path();
      std::ifstream input(pack_path, std::ios::binary);
      std::ostringstream contents; contents << input.rdbuf();
      const std::string source = contents.str();
      const JsonValue pack = JsonParser(source).Parse();
      if (Integer(pack.at("schema_version")) != 2) {
        return Status::InvalidArgument("Kernel Pack schema v2 is required; regenerate artifacts");
      }
      if (Integer(pack.at("abi_schema_version")) != 1) {
        return Status::InvalidArgument("Unsupported Kernel Pack ABI schema");
      }
      const std::string pack_name = pack.at("pack_name").string();
      const int pack_version = Integer(pack.at("pack_version"));
      const std::string generator = pack.at("generator_version").string();
      const std::string triton_version = pack.at("triton_version").string();
      repository_material += Sha256(source);
      for (const auto& entry : pack.at("kernels").array()) {
        KernelArtifact artifact;
        artifact.pack_name = pack_name;
        artifact.pack_version = pack_version;
        artifact.generator_version = generator;
        artifact.triton_version = triton_version;
        artifact.abi_schema_version = 1;
        artifact.kernel_id = entry.at("kernel_id").string();
        artifact.op_type = entry.at("op_type").string();
        artifact.kernel_version = Integer(entry.at("kernel_version"));
        artifact.variant_id = entry.at("variant_id").string();
        artifact.is_default = entry.at("default").boolean();
        artifact.symbol = entry.at("symbol").string();
        artifact.binary_format = entry.at("binary_format").string();
        artifact.sha256 = entry.at("sha256").string();
        artifact.dtype = entry.at("dtype").string();
        artifact.layout = entry.at("layout").string();
        artifact.min_compute_capability = Integer(entry.at("min_compute_capability"));
        artifact.num_warps = Integer(entry.at("num_warps"));
        artifact.num_stages = Integer(entry.at("num_stages"));
        artifact.shared_memory_bytes = Integer(entry.at("shared_memory_bytes"));
        const auto& grid = entry.at("grid");
        if (grid.at("op").string() != "ceil_div" ||
            grid.at("value").at("kind").string() != "output_numel") {
          throw std::runtime_error("unsupported grid expression");
        }
        artifact.grid_divisor = Integer(grid.at("divisor"));
        const std::filesystem::path relative(entry.at("binary").string());
        if (relative.is_absolute() || relative.string().find("..") != std::string::npos) {
          throw std::runtime_error("unsafe Kernel Pack binary path");
        }
        artifact.binary_path = root / relative;
        std::ifstream binary(artifact.binary_path, std::ios::binary);
        if (!binary) return Status::NotFound("Kernel binary not found: " + artifact.binary_path.string());
        std::ostringstream binary_contents; binary_contents << binary.rdbuf();
        if (Sha256(binary_contents.str()) != artifact.sha256) {
          return Status::InvalidArgument("Kernel binary SHA-256 mismatch: " + artifact.binary_path.string());
        }
        for (const auto& argument_value : entry.at("arguments").array()) {
          KernelArgumentSpec argument;
          argument.name = argument_value.at("name").string();
          argument.type = argument_value.at("type").string();
          const auto& source_value = argument_value.at("source");
          argument.source.kind = SourceKind(source_value.at("kind").string());
          argument.source.index = OptionalInt(source_value, "index", 0);
          argument.source.axis = OptionalInt(source_value, "axis", 0);
          argument.source.begin_axis = OptionalInt(source_value, "begin_axis", 0);
          if (const auto* name = source_value.find("name")) argument.source.attribute_name = name->string();
          if (const auto* alternate = source_value.find("alternate_name")) {
            argument.source.alternate_attribute_name = alternate->string();
          }
          argument.source.attribute_index = OptionalInt(source_value, "index", 0);
          argument.source.default_int = OptionalInt(source_value, "default", 0);
          argument.source.default_float = OptionalDouble(source_value, "default", 0.0);
          artifact.arguments.push_back(std::move(argument));
        }
        std::unordered_set<std::string> argument_names;
        int reserved_arguments = 0;
        for (std::size_t index = 0; index < artifact.arguments.size(); ++index) {
          const auto& argument = artifact.arguments[index];
          if (!argument_names.insert(argument.name).second) {
            throw std::runtime_error("duplicate Kernel argument name");
          }
          if (argument.source.kind ==
              KernelArgumentSourceKind::kRuntimeReserved) {
            ++reserved_arguments;
            if (index + 2 < artifact.arguments.size()) {
              throw std::runtime_error(
                  "runtime reserved arguments must be at the ABI tail");
            }
          }
        }
        if (reserved_arguments != 2) {
          throw std::runtime_error(
              "ABI schema 1 requires two runtime reserved arguments");
        }
        for (const auto& constraint_value : entry.at("constraints").array()) {
          KernelConstraint constraint;
          const std::string kind = constraint_value.at("kind").string();
          constraint.expected = Integer(constraint_value.at("value"));
          if (kind == "attribute_i64_eq") {
            constraint.kind = KernelConstraintKind::kAttributeInt64Equals;
            constraint.attribute_name = constraint_value.at("name").string();
            constraint.default_value =
                OptionalInt(constraint_value, "default", 0);
          } else if (kind == "input_count_eq") {
            constraint.kind = KernelConstraintKind::kInputCountEquals;
          } else if (kind == "tensor_rank_eq") {
            constraint.kind = KernelConstraintKind::kTensorRankEquals;
            const std::string source = constraint_value.at("source").string();
            if (source != "input" && source != "output") {
              throw std::runtime_error(
                  "tensor_rank_eq source must be input or output");
            }
            constraint.tensor_is_output = source == "output";
            constraint.tensor_index = OptionalInt(constraint_value, "index", 0);
          } else {
            throw std::runtime_error("unsupported Kernel constraint: " + kind);
          }
          artifact.constraints.push_back(std::move(constraint));
        }
        if (artifact.kernel_id.empty() || artifact.variant_id.empty() ||
            artifact.symbol.empty() || artifact.arguments.empty() ||
            artifact.binary_format != "ptx" || artifact.grid_divisor <= 0) {
          throw std::runtime_error("incomplete kernel artifact entry");
        }
        repository_material += artifact.sha256;
        artifacts_.push_back(artifact);
        by_op_type_[artifact.op_type].push_back(std::move(artifact));
      }
    }
  } catch (const std::exception& error) {
    artifacts_.clear(); by_op_type_.clear();
    return Status::InvalidArgument(std::string("Invalid Kernel Pack: ") + error.what());
  }
  for (const auto& entry : by_op_type_) {
    int defaults = 0;
    for (const auto& artifact : entry.second) defaults += artifact.is_default ? 1 : 0;
    if (defaults == 0) return Status::InvalidArgument("No default kernel variant for " + entry.first);
  }
  repository_hash_ = Sha256(repository_material);
  return Status::OK();
}

const std::vector<KernelArtifact>* ArtifactRepository::Find(
    const std::string& op_type) const {
  const auto it = by_op_type_.find(op_type);
  return it == by_op_type_.end() ? nullptr : &it->second;
}

std::vector<const KernelArtifact*> KernelCatalog::Query(
    const std::string& op_type, const std::string& dtype,
    const std::string& layout, int compute_capability) const {
  std::vector<const KernelArtifact*> result;
  if (repository_ == nullptr) return result;
  const auto* artifacts = repository_->Find(op_type);
  if (artifacts == nullptr) return result;
  for (const auto& artifact : *artifacts) {
    if (artifact.dtype == dtype &&
        (layout.empty() || artifact.layout.empty() ||
         artifact.layout == layout) &&
        compute_capability >= artifact.min_compute_capability) {
      result.push_back(&artifact);
    }
  }
  return result;
}

}  // namespace cutriton
