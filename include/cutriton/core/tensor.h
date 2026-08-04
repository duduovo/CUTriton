#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include "cutriton/core/status.h"

namespace cutriton {

enum class DataType {
  kUnknown = 0,//0
  kFloat32,//4
  kFloat16,//2
  kInt32,//4
  kInt64,//8
};

//数据存放位置
enum class DeviceType {
  kCPU = 0,
  kCUDA,
};

const char* DataTypeName(DataType dtype);
const char* DeviceTypeName(DeviceType device_type);
std::size_t ElementSize(DataType dtype);//计算元素字节数  

//============张量描述===================
struct TensorDesc {
  std::vector<int64_t> shape;
  DataType dtype{DataType::kUnknown};
  DeviceType device_type{DeviceType::kCPU};
  int device_id{0};//卡id
  std::string layout{"NCHW"};

  TensorDesc() = default;
  TensorDesc(std::vector<int64_t> dims, DataType type,
             DeviceType device = DeviceType::kCPU, int id = 0,
             std::string tensor_layout = "NCHW")
      : shape(std::move(dims)),
        dtype(type),
        device_type(device),
        device_id(id),
        layout(std::move(tensor_layout)) {}
  //是否为静态
  bool IsStatic() const;
  int64_t NumElements() const;//总的数据个数
  std::size_t ByteSize() const;
  Status Validate() const;
};

//前向声明
class Buffer;

//==========描述信息，存放真实数据=============
class Tensor {
 public:
  Tensor() = default;
  //desc:张量描述     tensor:存放真实数据     buffer:实际存放地址
  Tensor(TensorDesc desc, std::shared_ptr<Buffer> buffer,//多个tensor可以指向同一块buffer
         std::size_t byte_offset = 0)
      : desc_(std::move(desc)),
        buffer_(std::move(buffer)),
        byte_offset_(byte_offset) {}

  const TensorDesc& desc() const { return desc_; }//获取描述
  TensorDesc& mutable_desc() { return desc_; }//修改描述
  const std::shared_ptr<Buffer>& buffer() const { return buffer_; }
  std::size_t byte_offset() const { return byte_offset_; }//获取偏移量
  bool defined() const { return buffer_ != nullptr; }//是否分配

 private:
  TensorDesc desc_;
  std::shared_ptr<Buffer> buffer_;
  std::size_t byte_offset_{0};
};

}  // 命名空间 cutriton
