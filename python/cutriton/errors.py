class CUTritonError(RuntimeError):
    """Base exception for CUTriton."""


class ModelValidationError(CUTritonError):
    """The supplied ONNX model is invalid or outside configured limits."""


class CompilationError(CUTritonError):
    """The model could not be compiled into an execution plan."""


class InputValidationError(CUTritonError):
    """Runtime inputs do not match the compiled model contract."""


class BackendUnavailableError(CUTritonError):
    """A required runtime backend is not installed or cannot initialize."""


class ServiceOverloadedError(CUTritonError):
    """The serving queue is full and cannot accept more work."""
