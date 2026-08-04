#include "cutriton/core/status.h"

namespace cutriton {

const char* ErrorCodeName(ErrorCode code) {
  switch (code) {
    case ErrorCode::kOk:
      return "OK";
    case ErrorCode::kInvalidArgument:
      return "参数无效";
    case ErrorCode::kNotFound:
      return "未找到";
    case ErrorCode::kAlreadyExists:
      return "已存在";
    case ErrorCode::kUnsupported:
      return "不支持";
    case ErrorCode::kShapeError:
      return "形状错误";
    case ErrorCode::kRuntimeError:
      return "运行时错误";
    case ErrorCode::kInternal:
      return "内部错误";
  }
  return "未知错误";
}

std::string Status::ToString() const {
  if (ok()) {
    return "OK";
  }
  return std::string(ErrorCodeName(code_)) + ": " + message_;
}

}  // 命名空间 cutriton
