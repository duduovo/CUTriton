#include "cutriton/core/tensor.h"

#include <limits>

namespace cutriton {

const char* DataTypeName(DataType dtype) {
  switch (dtype) {
    case DataType::kFloat32:
      return "float32";
    case DataType::kFloat16:
      return "float16";
    case DataType::kInt32:
      return "int32";
    case DataType::kInt64:
      return "int64";
    case DataType::kUnknown:
      return "未知";
  }
  return "未知";
}

const char* DeviceTypeName(DeviceType device_type) {
  switch (device_type) {
    case DeviceType::kCPU:
      return "cpu";
    case DeviceType::kCUDA:
      return "cuda";
  }
  return "未知";
}

std::size_t ElementSize(DataType dtype) {
  switch (dtype) {
    case DataType::kFloat32:
    case DataType::kInt32:
      return 4;
    case DataType::kFloat16:
      return 2;
    case DataType::kInt64:
      return 8;
    case DataType::kUnknown:
      return 0;
  }
  return 0;
}

bool TensorDesc::IsStatic() const {
  if (shape.empty()) {
    return false;
  }
  for (int64_t dim : shape) {
    if (dim <= 0) {
      return false;
    }
  }
  return dtype != DataType::kUnknown;
}

int64_t TensorDesc::NumElements() const {
  if (shape.empty()) {
    return 0;
  }
  int64_t elements = 1;
  for (int64_t dim : shape) {
    if (dim <= 0) {
      return 0;
    }
    if (elements > std::numeric_limits<int64_t>::max() / dim) {
      return 0;
    }
    elements *= dim;
  }
  return elements;
}

std::size_t TensorDesc::ByteSize() const {
  const std::size_t elem_size = ElementSize(dtype);
  const int64_t elements = NumElements();
  if (elem_size == 0 || elements <= 0) {
    return 0;
  }
  return static_cast<std::size_t>(elements) * elem_size;
}

Status TensorDesc::Validate() const {
  if (dtype == DataType::kUnknown) {
    return Status::ShapeError("Tensor dtype 未知");
  }
  if (shape.empty()) {
    return Status::ShapeError("Tensor shape 为空");
  }
  for (int64_t dim : shape) {
    if (dim <= 0) {
      return Status::ShapeError("静态 Tensor shape 包含非正维度");
    }
  }
  if (ByteSize() == 0) {
    return Status::ShapeError("Tensor 字节大小为 0");
  }
  return Status::OK();
}

}  // 命名空间 cutriton
