#include "cutriton/core/buffer.h"

#include <cstring>
#include <limits>
#include <string>

#ifndef CUTRITON_ENABLE_CUDA
#define CUTRITON_ENABLE_CUDA 0
#endif

#if CUTRITON_ENABLE_CUDA
#include <cuda.h>
#endif

namespace cutriton {
namespace {

#if CUTRITON_ENABLE_CUDA
Status CudaStatus(CUresult result, const char* operation) {
  if (result == CUDA_SUCCESS) {
    return Status::OK();
  }
  const char* name = nullptr;
  const char* message = nullptr;
  cuGetErrorName(result, &name);
  cuGetErrorString(result, &message);
  return Status::RuntimeError(std::string(operation) + " failed: " +
                              (name == nullptr ? "CUDA_ERROR" : name) +
                              " (" + (message == nullptr ? "unknown" : message) +
                              ")");
}

Status SetCudaDevice(int device_id, CUcontext* context) {
  CUTRITON_RETURN_IF_ERROR(CudaStatus(cuInit(0), "cuInit"));
  CUdevice device{};
  CUTRITON_RETURN_IF_ERROR(
      CudaStatus(cuDeviceGet(&device, device_id), "cuDeviceGet"));
  CUTRITON_RETURN_IF_ERROR(CudaStatus(
      cuDevicePrimaryCtxRetain(context, device), "cuDevicePrimaryCtxRetain"));
  CUTRITON_RETURN_IF_ERROR(CudaStatus(cuCtxSetCurrent(*context), "cuCtxSetCurrent"));
  return Status::OK();
}
#endif

}  // namespace

std::shared_ptr<Buffer> Buffer::AllocateHost(std::size_t size_bytes) {
  if (size_bytes == 0) {
    return std::make_shared<Buffer>();
  }
  auto deleter = [](void* ptr) { delete[] static_cast<std::uint8_t*>(ptr); };
  void* raw = new std::uint8_t[size_bytes]();
  std::shared_ptr<void> owner(raw, deleter);
  return std::shared_ptr<Buffer>(
      new Buffer(raw, size_bytes, DeviceType::kCPU, 0, owner));
}

Status Buffer::AllocateCuda(std::size_t size_bytes, int device_id,
                            std::shared_ptr<Buffer>* buffer) {
  if (buffer == nullptr) {
    return Status::InvalidArgument("CUDA Buffer output pointer is null");
  }
  if (size_bytes == 0) {
    *buffer = std::make_shared<Buffer>();
    return Status::OK();
  }
#if CUTRITON_ENABLE_CUDA
  CUcontext context{};
  CUTRITON_RETURN_IF_ERROR(SetCudaDevice(device_id, &context));
  CUdeviceptr pointer{};
  Status status = CudaStatus(cuMemAlloc(&pointer, size_bytes), "cuMemAlloc");
  if (!status.ok()) {
    CUdevice device{};
    if (cuDeviceGet(&device, device_id) == CUDA_SUCCESS) {
      cuDevicePrimaryCtxRelease(device);
    }
    return status;
  }
  void* raw = reinterpret_cast<void*>(static_cast<std::uintptr_t>(pointer));
  std::shared_ptr<void> owner(raw, [device_id](void* value) {
    CUcontext owner_context{};
    if (SetCudaDevice(device_id, &owner_context).ok()) {
      cuMemFree(static_cast<CUdeviceptr>(reinterpret_cast<std::uintptr_t>(value)));
      CUdevice device{};
      if (cuDeviceGet(&device, device_id) == CUDA_SUCCESS) {
        cuDevicePrimaryCtxRelease(device);
        cuDevicePrimaryCtxRelease(device);
      }
    }
  });
  *buffer = std::shared_ptr<Buffer>(
      new Buffer(raw, size_bytes, DeviceType::kCUDA, device_id, owner));
  return Status::OK();
#else
  (void)size_bytes;
  (void)device_id;
  return Status::Unsupported("CUTriton was built without CUDA support");
#endif
}

std::shared_ptr<Buffer> Buffer::WrapExternal(void* data, std::size_t size_bytes,
                                             DeviceType device_type,
                                             int device_id) {
  return std::shared_ptr<Buffer>(
      new Buffer(data, size_bytes, device_type, device_id, nullptr));
}

Status Buffer::CopyFromHost(const void* source, std::size_t size_bytes,
                            std::size_t destination_offset) {
  if (source == nullptr && size_bytes != 0) {
    return Status::InvalidArgument("Host source pointer is null");
  }
  if (destination_offset > size_bytes_ ||
      size_bytes > size_bytes_ - destination_offset) {
    return Status::InvalidArgument("Buffer copy exceeds destination capacity");
  }
  if (size_bytes == 0) {
    return Status::OK();
  }
  if (device_type_ == DeviceType::kCPU) {
    std::memcpy(static_cast<std::uint8_t*>(data_) + destination_offset, source,
                size_bytes);
    return Status::OK();
  }
#if CUTRITON_ENABLE_CUDA
  CUcontext context{};
  CUTRITON_RETURN_IF_ERROR(SetCudaDevice(device_id_, &context));
  const CUdeviceptr destination =
      static_cast<CUdeviceptr>(reinterpret_cast<std::uintptr_t>(data_)) +
      destination_offset;
  Status status = CudaStatus(cuMemcpyHtoD(destination, source, size_bytes),
                             "cuMemcpyHtoD");
  CUdevice device{};
  if (cuDeviceGet(&device, device_id_) == CUDA_SUCCESS) {
    cuDevicePrimaryCtxRelease(device);
  }
  return status;
#else
  return Status::Unsupported("CUTriton was built without CUDA support");
#endif
}

Status Buffer::CopyToHost(void* destination, std::size_t size_bytes,
                          std::size_t source_offset) const {
  if (destination == nullptr && size_bytes != 0) {
    return Status::InvalidArgument("Host destination pointer is null");
  }
  if (source_offset > size_bytes_ || size_bytes > size_bytes_ - source_offset) {
    return Status::InvalidArgument("Buffer copy exceeds source capacity");
  }
  if (size_bytes == 0) {
    return Status::OK();
  }
  if (device_type_ == DeviceType::kCPU) {
    std::memcpy(destination,
                static_cast<const std::uint8_t*>(data_) + source_offset,
                size_bytes);
    return Status::OK();
  }
#if CUTRITON_ENABLE_CUDA
  CUcontext context{};
  CUTRITON_RETURN_IF_ERROR(SetCudaDevice(device_id_, &context));
  const CUdeviceptr source =
      static_cast<CUdeviceptr>(reinterpret_cast<std::uintptr_t>(data_)) +
      source_offset;
  Status status = CudaStatus(cuMemcpyDtoH(destination, source, size_bytes),
                             "cuMemcpyDtoH");
  CUdevice device{};
  if (cuDeviceGet(&device, device_id_) == CUDA_SUCCESS) {
    cuDevicePrimaryCtxRelease(device);
  }
  return status;
#else
  return Status::Unsupported("CUTriton was built without CUDA support");
#endif
}

}  // namespace cutriton
