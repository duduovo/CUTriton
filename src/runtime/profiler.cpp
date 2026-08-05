#include "cutriton/runtime/profiler.h"

#include <utility>

namespace cutriton {

void Profiler::Record(std::string name, std::string backend,
                      double duration_ms) {
  events_.push_back(ProfileEvent{std::move(name), std::move(backend),
                                 duration_ms});
}

ScopedCpuTimer::ScopedCpuTimer(Profiler* profiler, std::string name,
                               std::string backend)
    : profiler_(profiler),
      name_(std::move(name)),
      backend_(std::move(backend)),
      start_(std::chrono::steady_clock::now()) {}

ScopedCpuTimer::~ScopedCpuTimer() {
  if (profiler_ == nullptr) {
    return;
  }
  const auto end = std::chrono::steady_clock::now();
  const auto elapsed =
      std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(
          end - start_);
  profiler_->Record(name_, backend_, elapsed.count());
}

}  // 命名空间 cutriton
