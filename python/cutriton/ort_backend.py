from __future__ import annotations

import json
import logging
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import FallbackPolicy
from .errors import BackendUnavailableError, InputValidationError

LOGGER = logging.getLogger("cutriton.ort")


def _imports() -> tuple[Any, Any, Any]:
    try:
        import numpy as np
        import onnxruntime as ort
        import torch
    except ImportError as error:
        raise BackendUnavailableError(
            "ORT execution requires torch, numpy and onnxruntime-gpu"
        ) from error
    return np, ort, torch


def _numpy_dtype(dtype: Any, np: Any, torch: Any) -> Any:
    mapping = {
        torch.float32: np.float32,
        torch.float16: np.float16,
        torch.float64: np.float64,
        torch.int64: np.int64,
        torch.int32: np.int32,
        torch.int16: np.int16,
        torch.int8: np.int8,
        torch.uint8: np.uint8,
        torch.bool: np.bool_,
    }
    if dtype not in mapping:
        raise InputValidationError(f"ORT pointer binding does not support dtype {dtype}")
    return mapping[dtype]


def _from_dlpack(ort: Any, torch: Any, tensor: Any) -> Any | None:
    factory = getattr(ort.OrtValue, "from_dlpack", None)
    if factory is not None:
        return factory(tensor)
    native = getattr(getattr(ort, "capi", None), "_pybind_state", None)
    native_type = getattr(native, "OrtValue", None)
    native_factory = getattr(native_type, "from_dlpack", None)
    if native_factory is not None:
        value = native_factory(torch.utils.dlpack.to_dlpack(tensor))
        wrapper = getattr(ort.OrtValue, "_from_native", None)
        return wrapper(value) if wrapper is not None else value
    return None


def _to_torch(value: Any, torch: Any) -> Any:
    if hasattr(value, "__dlpack__"):
        return torch.from_dlpack(value)
    to_dlpack = getattr(value, "to_dlpack", None)
    if to_dlpack is None and hasattr(value, "_ortvalue"):
        to_dlpack = getattr(value._ortvalue, "to_dlpack", None)
    if to_dlpack is not None:
        return torch.utils.dlpack.from_dlpack(to_dlpack())
    raise BackendUnavailableError(
        "the pinned ONNX Runtime build does not expose DLPack output interop"
    )


@dataclass(frozen=True)
class OrtProviderReport:
    requested: tuple[str, ...]
    active: tuple[str, ...]


class OrtSegmentRunner:
    def __init__(
        self,
        model_bytes: bytes,
        device_id: int,
        stream: Any,
        fallback: FallbackPolicy,
        reject_cpu_fallback: bool,
    ) -> None:
        np, ort, torch = _imports()
        del np
        available = set(ort.get_available_providers())
        if "CUDAExecutionProvider" not in available:
            raise BackendUnavailableError(
                "onnxruntime-gpu has no CUDAExecutionProvider; check the locked CUDA/cuDNN stack"
            )
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        session_options.enable_mem_pattern = True
        cpu_fallback_allowed = fallback == FallbackPolicy.ORT_CUDA_CPU and not reject_cpu_fallback
        if cpu_fallback_allowed:
            session_options.enable_profiling = True
            session_options.profile_file_prefix = str(
                Path(tempfile.gettempdir()) / "cutriton-ort-profile"
            )
        if fallback == FallbackPolicy.ORT_CUDA or reject_cpu_fallback:
            session_options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
        stream_handle = (
            int(stream.cuda_stream)
            if stream is not None
            else int(torch.cuda.current_stream(device_id).cuda_stream)
        )
        cuda_options = {
            "device_id": device_id,
            "user_compute_stream": str(stream_handle),
            "do_copy_in_default_stream": "1",
        }
        providers: list[Any] = [("CUDAExecutionProvider", cuda_options)]
        if fallback == FallbackPolicy.ORT_CUDA_CPU and not reject_cpu_fallback:
            providers.append("CPUExecutionProvider")
        try:
            self.session = ort.InferenceSession(
                model_bytes, sess_options=session_options, providers=providers
            )
        except Exception as error:
            raise BackendUnavailableError(f"failed to create ORT segment: {error}") from error
        self.device_id = device_id
        self.input_names = tuple(item.name for item in self.session.get_inputs())
        self.output_names = tuple(item.name for item in self.session.get_outputs())
        self.provider_report = OrtProviderReport(
            tuple(item[0] if isinstance(item, tuple) else item for item in providers),
            tuple(self.session.get_providers()),
        )
        self.cpu_fallback_observed = False
        self._profile_pending = cpu_fallback_allowed

    def run(self, tensors: Mapping[str, Any]) -> dict[str, Any]:
        np, ort, torch = _imports()
        binding = self.session.io_binding()
        keepalive: list[Any] = []
        for name in self.input_names:
            if name not in tensors:
                raise InputValidationError(f"missing ORT segment input {name!r}")
            tensor = tensors[name]
            if not tensor.is_contiguous():
                raise InputValidationError(f"ORT segment input {name!r} is not contiguous")
            if tensor.device.type != "cuda" or tensor.device.index != self.device_id:
                raise InputValidationError(f"ORT segment input {name!r} is on the wrong device")
            ort_value = _from_dlpack(ort, torch, tensor)
            if ort_value is not None and hasattr(binding, "bind_ortvalue_input"):
                binding.bind_ortvalue_input(name, ort_value)
                keepalive.append(ort_value)
            else:
                binding.bind_input(
                    name,
                    "cuda",
                    self.device_id,
                    _numpy_dtype(tensor.dtype, np, torch),
                    tuple(tensor.shape),
                    tensor.data_ptr(),
                )
            keepalive.append(tensor)
        for name in self.output_names:
            binding.bind_output(name, "cuda", self.device_id)
        self.session.run_with_iobinding(binding)
        values = binding.get_outputs()
        result = {
            name: _to_torch(value, torch)
            for name, value in zip(self.output_names, values, strict=False)
        }
        keepalive.extend(values)
        self._inspect_first_run_profile()
        return result

    def _inspect_first_run_profile(self) -> None:
        if not self._profile_pending:
            return
        self._profile_pending = False
        path: Path | None = None
        try:
            path = Path(self.session.end_profiling())
            events = json.loads(path.read_text(encoding="utf-8"))
            self.cpu_fallback_observed = any(
                event.get("args", {}).get("provider") == "CPUExecutionProvider" for event in events
            )
            if self.cpu_fallback_observed:
                LOGGER.warning("ORT CPU fallback was observed for an inference segment")
        except Exception:
            LOGGER.exception("failed to inspect ORT execution-provider profile")
        finally:
            if path is not None:
                path.unlink(missing_ok=True)

    def close(self) -> None:
        if self._profile_pending:
            self._profile_pending = False
            try:
                profile = self.session.end_profiling()
                if profile:
                    Path(profile).unlink(missing_ok=True)
            except Exception:
                LOGGER.exception("failed to finalize ORT provider profile")
