#pragma once

#include <chrono>
#include <string>
#include <vector>

//性能计时
namespace cutriton {

struct ProfileEvent {
  std::string name;
  std::string backend; //使用后端
  double duration_ms{0.0}; //持续时间(ms)
};
// CPU Kernel 可使用 ScopedCpuTimer；CUDA Runtime 使用 CUDA Event 记录耗时。
class Profiler {
 public:
  void Record(std::string name, std::string backend, double duration_ms);
  const std::vector<ProfileEvent>& events() const { return events_; }
  void Clear() { events_.clear(); }

 private:
  std::vector<ProfileEvent> events_;
};

class ScopedCpuTimer {
 public:
  ScopedCpuTimer(Profiler* profiler, std::string name, std::string backend);
  ~ScopedCpuTimer();

 private:
  Profiler* profiler_{nullptr};
  std::string name_;
  std::string backend_;
  std::chrono::steady_clock::time_point start_;
};

}  // 命名空间 cutriton
