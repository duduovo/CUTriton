#pragma once

#include <string>
#include <utility>

namespace cutriton {

// CUTriton 公共接口使用的错误分类，调用方可据此进行分支处理。
enum class ErrorCode {
  kOk = 0,
  kInvalidArgument,
  kNotFound,
  kAlreadyExists,
  kUnsupported,
  kShapeError,
  kRuntimeError,
  kInternal,
};

// 轻量级返回状态：成功时 code 为 kOk 且消息为空，失败时携带错误类别和上下文。
// 可预期的参数、能力和运行时错误通过 Status 返回，不依赖 C++ 异常传播。
class Status {
 public:
  Status() = default;
  Status(ErrorCode code, std::string message)
      : code_(code), message_(std::move(message)) {}

  static Status OK() { return Status(); }
  static Status InvalidArgument(std::string message) {
    return Status(ErrorCode::kInvalidArgument, std::move(message));
  }
  static Status NotFound(std::string message) {
    return Status(ErrorCode::kNotFound, std::move(message));
  }
  static Status AlreadyExists(std::string message) {
    return Status(ErrorCode::kAlreadyExists, std::move(message));
  }
  static Status Unsupported(std::string message) {
    return Status(ErrorCode::kUnsupported, std::move(message));
  }
  static Status ShapeError(std::string message) {
    return Status(ErrorCode::kShapeError, std::move(message));
  }
  static Status RuntimeError(std::string message) {
    return Status(ErrorCode::kRuntimeError, std::move(message));
  }
  static Status Internal(std::string message) {
    return Status(ErrorCode::kInternal, std::move(message));
  }

  bool ok() const { return code_ == ErrorCode::kOk; }
  ErrorCode code() const { return code_; }
  const std::string& message() const { return message_; }
  // 返回适合日志展示的“错误类别: 详细消息”；成功时返回 "OK"。
  std::string ToString() const;

 private:
  ErrorCode code_{ErrorCode::kOk};
  std::string message_;
};

// 返回稳定的错误类别名称，不包含某次失败的详细上下文。
const char* ErrorCodeName(ErrorCode code);

}  // 命名空间 cutriton

// 在返回 Status 的函数中传播首个错误，避免遗漏中间步骤的失败状态。
#define CUTRITON_RETURN_IF_ERROR(expr) \
  do {                                 \
    ::cutriton::Status _status = (expr); \
    if (!_status.ok()) {               \
      return _status;                  \
    }                                  \
  } while (false)
