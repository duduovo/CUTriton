#include "cutriton/core/buffer.h"

namespace cutriton {

std::shared_ptr<Buffer> Buffer::AllocateHost(std::size_t size_bytes) {
  if (size_bytes == 0) {
    return std::make_shared<Buffer>();
  }
  auto deleter = [](void* ptr) {
    delete[] static_cast<std::uint8_t*>(ptr);
  };
  void* raw = new std::uint8_t[size_bytes]();
  std::shared_ptr<void> owner(raw, deleter);
  return std::shared_ptr<Buffer>(
      new Buffer(raw, size_bytes, DeviceType::kCPU, 0, owner));
}

std::shared_ptr<Buffer> Buffer::WrapExternal(void* data, std::size_t size_bytes,
                                             DeviceType device_type,
                                             int device_id) {
  return std::shared_ptr<Buffer>(
      new Buffer(data, size_bytes, device_type, device_id, nullptr));//关键区别:owner_初始化为NULL
}

}  // 命名空间 cutriton
