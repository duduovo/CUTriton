#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>

#include "cutriton/core/status.h"
#include "cutriton/core/tensor.h"

namespace cutriton {
//========数据内存管理=============
class Buffer {
 public:
  Buffer() = default;
  //申请内存-->创建buffer对象-->记录地址和大小-->返回内存shared_ptr
  static std::shared_ptr<Buffer> AllocateHost(std::size_t size_bytes);
  //申请CUDA内存
  static Status AllocateCuda(std::size_t size_bytes, int device_id,
                             std::shared_ptr<Buffer>* buffer);
  //不申请内存，将存在的内存数据包装为buffer，不负责释放内存，data存在周期比buffer长
  static std::shared_ptr<Buffer> WrapExternal(void* data, std::size_t size_bytes,
                                              DeviceType device_type,
                                              int device_id = 0);
  //拷贝数据
  Status CopyFromHost(const void* source, std::size_t size_bytes,
                      std::size_t destination_offset = 0);
  Status CopyToHost(void* destination, std::size_t size_bytes,
                    std::size_t source_offset = 0) const;

  void* data() const { return data_; }
  std::size_t size_bytes() const { return size_bytes_; }
  DeviceType device_type() const { return device_type_; }
  int device_id() const { return device_id_; }
  //是否拥有内存，如果onwer_指针存在，buffer负责生命周期，否者不负责
  bool owns_memory() const { return owner_ != nullptr; }
  bool empty() const { return data_ == nullptr || size_bytes_ == 0; }

 private:
  Buffer(void* data, std::size_t size_bytes, DeviceType device_type,
         int device_id, std::shared_ptr<void> owner)
      : data_(data),
        size_bytes_(size_bytes),
        device_type_(device_type),
        device_id_(device_id),
        owner_(std::move(owner)) {}

  void* data_{nullptr}; //内存起始地址
  std::size_t size_bytes_{0};
  DeviceType device_type_{DeviceType::kCPU};
  int device_id_{0};
  std::shared_ptr<void> owner_;//内存所有者，负责释放内存，void表示不关心具体元素类型，只负责内存生命周期
};

}  // 命名空间 cutriton
