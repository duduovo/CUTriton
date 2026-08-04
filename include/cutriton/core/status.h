#pragma once

#include <string>
#include <utility>

namespace cutriton {

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
  std::string ToString() const;

 private:
  ErrorCode code_{ErrorCode::kOk};
  std::string message_;
};

const char* ErrorCodeName(ErrorCode code);

}  // 命名空间 cutriton

#define CUTRITON_RETURN_IF_ERROR(expr) \
  do {                                 \
    ::cutriton::Status _status = (expr); \
    if (!_status.ok()) {               \
      return _status;                  \
    }                                  \
  } while (false)
