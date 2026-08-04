#pragma once

#include <string>

#include "cutriton/core/tensor.h"

namespace cutriton {
//==========设备管理============
struct Device {
  DeviceType type{DeviceType::kCPU};
  int id{0};
  std::string name{"cpu"};

  static Device CPU() { return Device{}; }
  static Device CUDA(int device_id = 0) {
    return Device{DeviceType::kCUDA, device_id, "cuda:" + std::to_string(device_id)};
  }
};
//=====设备具体信息==========
struct DeviceProperties {
  Device device;
  int sm_count{0};
  std::size_t l2_cache_bytes{0};
  std::size_t global_memory_bytes{0};
};

}  // 命名空间 cutriton
