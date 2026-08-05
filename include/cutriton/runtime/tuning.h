#pragma once

#include <string>

namespace cutriton {

enum class TuningMode {
  kDisabled,
  kUseCache,
  kTuneOnMiss,
  kForceRetune,
};

struct TuningConfig {
  TuningMode mode{TuningMode::kUseCache};
  std::string cache_dir;
  int warmup_iterations{5};
  int measurement_iterations{20};
};

}  // namespace cutriton
