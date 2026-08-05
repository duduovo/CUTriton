#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace cutriton {

struct ShapeRange {
  std::vector<int64_t> min;
  std::vector<int64_t> opt;
  std::vector<int64_t> max;
};

struct ShapeProfile {
  std::string name;
  std::unordered_map<std::string, ShapeRange> inputs;
};

}  // namespace cutriton
