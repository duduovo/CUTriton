from __future__ import annotations

import hashlib
import inspect
import json
import logging
import os
import subprocess
import threading
import time
from collections.abc import Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any

from .cache import JitDecisionCache
from .config import CompileOptions, ShapeProfile
from .errors import BackendUnavailableError, InputValidationError
from .graph import GraphSegment, ModelGraph, SegmentKind, load_model, partition_graph
from .ort_backend import OrtSegmentRunner
from .registry import KernelRegistry
from .report import CompileReport, SegmentReport
from .triton_backend import create_default_registry

LOGGER = logging.getLogger("cutriton.runtime")


class RuntimeStats:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._values: dict[str, float] = {
            "triton_segment_runs": 0,
            "ort_segment_runs": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "compile_failures": 0,
            "compile_seconds": 0.0,
            "cuda_graph_replays": 0,
        }

    def add(self, name: str, value: float = 1.0) -> None:
        with self._lock:
            self._values[name] = self._values.get(name, 0.0) + value

    def snapshot(self) -> dict[str, float]:
        with self._lock:
            return dict(self._values)


def _torch() -> Any:
    try:
        import torch
    except ImportError as error:
        raise BackendUnavailableError("CUTriton requires PyTorch 2.11") from error
    return torch


def _torch_dtype(elem_type: int, torch: Any) -> Any:
    mapping = {
        1: torch.float32,
        2: torch.uint8,
        3: torch.int8,
        4: torch.uint16,
        5: torch.int16,
        6: torch.int32,
        7: torch.int64,
        9: torch.bool,
        10: torch.float16,
        11: torch.float64,
        12: torch.uint32,
        13: torch.uint64,
        16: torch.bfloat16,
    }
    try:
        return mapping[elem_type]
    except KeyError as error:
        raise InputValidationError(f"unsupported ONNX tensor element type {elem_type}") from error


def _load_initializers(graph: ModelGraph, device: Any) -> dict[str, Any]:
    import numpy as np
    import onnx

    torch = _torch()
    result: dict[str, Any] = {}
    for initializer in graph.model.graph.initializer:
        array = onnx.numpy_helper.to_array(initializer)
        if int(initializer.data_type) == 16:
            words = np.asarray(array).view(np.uint16).copy()
            tensor = torch.from_numpy(words).view(torch.bfloat16)
        else:
            tensor = torch.from_numpy(np.asarray(array).copy())
        result[initializer.name] = tensor.to(device=device, non_blocking=False).contiguous()
    return result


def _profile_shape(profile: ShapeProfile, name: str, point: str) -> tuple[int, ...]:
    try:
        return getattr(profile.inputs[name], point)
    except KeyError as error:
        raise InputValidationError(
            f"profile {profile.name!r} has no shape for model input {name!r}"
        ) from error


class Engine:
    """Immutable model state shared by independent execution contexts."""

    def __init__(
        self,
        graph: ModelGraph,
        options: CompileOptions,
        registry: KernelRegistry,
        segments: tuple[GraphSegment, ...],
    ) -> None:
        torch = _torch()
        if not torch.cuda.is_available():
            raise BackendUnavailableError("CUTriton production runtime requires an NVIDIA CUDA GPU")
        if options.device_id >= torch.cuda.device_count():
            raise BackendUnavailableError(f"CUDA device {options.device_id} does not exist")
        self.graph = graph
        self.options = options
        self.registry = registry
        self.segments = segments
        self.device = torch.device("cuda", options.device_id)
        triton_cache = options.cache_dir / "triton"
        triton_cache.mkdir(parents=True, exist_ok=True)
        # One model owns Triton's process-wide cache in the production topology.
        os.environ["TRITON_CACHE_DIR"] = str(triton_cache)
        self.gpu_uuid, self.driver_version = _nvidia_identity(options.device_id)
        with torch.cuda.device(self.device):
            self.initializers = _load_initializers(graph, self.device)
        self.cache = JitDecisionCache(options.cache_dir, options.max_cache_bytes)
        self._compiler = ThreadPoolExecutor(max_workers=1, thread_name_prefix="cutriton-jit")
        self._compile_lock = threading.Lock()
        self._compiling: dict[str, Future[Any]] = {}
        self._closed = False
        self.stats = RuntimeStats()
        warnings: list[str] = []
        if options.fallback.value == "ort_cuda_cpu" and not options.reject_cpu_fallback:
            warnings.append("CPU fallback is permitted and will be recorded when observed")
        if options.cuda_graph and any(item.kind == SegmentKind.ORT for item in segments):
            warnings.append("whole CUDA Graph capture is disabled for plans containing ORT")
        self.report = CompileReport(
            model_path=str(graph.path),
            model_hash=graph.model_hash,
            device_id=options.device_id,
            segments=tuple(
                SegmentReport(
                    item.index,
                    item.kind.value if item.kind == SegmentKind.TRITON else options.fallback.value,
                    tuple(node.name for node in item.nodes),
                    item.reason,
                    item.inputs,
                    item.outputs,
                )
                for item in segments
            ),
            cpu_fallback_allowed=(
                options.fallback.value == "ort_cuda_cpu" and not options.reject_cpu_fallback
            ),
            warnings=tuple(warnings),
        )

    def create_context(self) -> ExecutionContext:
        if self._closed:
            raise RuntimeError("engine is closed")
        return ExecutionContext(self)

    def warmup(self) -> None:
        """Synchronously compile, validate and benchmark every declared optimum profile."""
        if not self.options.profiles:
            LOGGER.info("warmup skipped: no shape profiles declared")
            return
        torch = _torch()
        context = self.create_context()
        try:
            for profile in self.options.profiles:
                seen_shapes: set[tuple[tuple[str, tuple[int, ...]], ...]] = set()
                for point in ("minimum", "optimum", "maximum"):
                    shape_key = tuple(
                        (name, _profile_shape(profile, name, point)) for name in self.graph.inputs
                    )
                    if shape_key in seen_shapes:
                        continue
                    seen_shapes.add(shape_key)
                    inputs: dict[str, Any] = {}
                    for name, shape in shape_key:
                        meta = self.graph.values[name]
                        dtype = _torch_dtype(meta.elem_type, torch)
                        if dtype == torch.bool:
                            inputs[name] = torch.zeros(shape, dtype=dtype, device=self.device)
                        elif dtype.is_floating_point:
                            inputs[name] = torch.randn(shape, dtype=dtype, device=self.device)
                        else:
                            inputs[name] = torch.zeros(shape, dtype=dtype, device=self.device)
                    context._run(inputs, synchronous_qualification=True)
                    # A second pass observes decisions and captures stable pure-Triton plans.
                    context._run(inputs, synchronous_qualification=True)
        finally:
            context.close()

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self._compiler.shutdown(wait=True, cancel_futures=False)

    def _submit_qualification(
        self, key: str, segment: GraphSegment, inputs: Mapping[str, Any], ready_event: Any
    ) -> Future[Any] | None:
        with self._compile_lock:
            existing = self._compiling.get(key)
            if existing is not None and not existing.done():
                return existing
            retained = {name: tensor.detach() for name, tensor in inputs.items()}
            future = self._compiler.submit(self._qualify, key, segment, retained, ready_event)
            self._compiling[key] = future
            future.add_done_callback(
                lambda completed, cache_key=key: self._finish(cache_key, completed)
            )
            return future

    def _submit_plan_qualification(
        self, key: str, inputs: Mapping[str, Any], ready_event: Any
    ) -> Future[Any] | None:
        with self._compile_lock:
            existing = self._compiling.get(key)
            if existing is not None and not existing.done():
                return existing
            retained = {name: tensor.detach() for name, tensor in inputs.items()}
            future = self._compiler.submit(self._qualify_plan, key, retained, ready_event)
            self._compiling[key] = future
            future.add_done_callback(
                lambda completed, cache_key=key: self._finish(cache_key, completed)
            )
            return future

    def _finish(self, key: str, future: Future[Any]) -> None:
        with self._compile_lock:
            self._compiling.pop(key, None)
        try:
            future.result()
        except Exception:
            LOGGER.exception("background JIT qualification failed", extra={"cache_key": key})

    def _qualify(
        self, key: str, segment: GraphSegment, inputs: Mapping[str, Any], ready_event: Any
    ) -> None:
        torch = _torch()
        started = time.monotonic()
        try:
            with torch.cuda.device(self.device):
                stream = torch.cuda.Stream(device=self.device, priority=2)
                stream.wait_event(ready_event)
                runner = OrtSegmentRunner(
                    segment.model_bytes,
                    self.options.device_id,
                    stream,
                    self.options.fallback,
                    self.options.reject_cpu_fallback,
                )
                with torch.cuda.stream(stream):
                    reference = runner.run(inputs)
                    candidate = self._run_triton(segment, inputs)
                stream.synchronize()
                accurate, detail = _compare_outputs(candidate, reference, torch)
                if not accurate:
                    self.cache.put(key, "disabled", {"reason": "numerical_mismatch", **detail})
                    return
                triton_ms = _median_cuda_ms(
                    lambda: self._run_triton(segment, inputs), stream, torch
                )
                ort_ms = _median_cuda_ms(lambda: runner.run(inputs), stream, torch)
                faster = triton_ms <= ort_ms * (1.0 - self.options.performance_margin)
                self.cache.put(
                    key,
                    "enabled" if faster else "disabled",
                    {
                        "reason": "performance_gate_passed"
                        if faster
                        else "performance_gate_failed",
                        "triton_ms": triton_ms,
                        "ort_ms": ort_ms,
                        "required_margin": self.options.performance_margin,
                    },
                )
        except Exception as error:
            self.stats.add("compile_failures")
            self.cache.put(key, "failed", {"reason": type(error).__name__, "message": str(error)})
            raise
        finally:
            self.stats.add("compile_seconds", time.monotonic() - started)

    def _run_triton(self, segment: GraphSegment, inputs: Mapping[str, Any]) -> dict[str, Any]:
        values = dict(inputs)
        for node, spec in zip(segment.nodes, segment.specs, strict=False):
            if spec is None:
                raise RuntimeError(f"segment node {node.name!r} has no Triton KernelSpec")
            node_inputs = [values[name] for name in node.inputs]
            outputs = spec.execute(node, node_inputs, self.graph.values)
            if len(outputs) != len(node.outputs):
                raise RuntimeError(f"kernel {spec.kernel_id!r} returned the wrong output count")
            values.update(zip(node.outputs, outputs, strict=False))
        return {name: values[name] for name in segment.outputs}

    def _run_all_triton(self, inputs: Mapping[str, Any]) -> dict[str, Any]:
        values = {**self.initializers, **inputs}
        for segment in self.segments:
            if segment.kind != SegmentKind.TRITON:
                raise RuntimeError("CUDA Graph capture requires a pure Triton plan")
            segment_inputs = {name: values[name] for name in segment.inputs}
            values.update(self._run_triton(segment, segment_inputs))
        return {name: values[name] for name in self.graph.outputs}

    def _qualify_plan(self, key: str, inputs: Mapping[str, Any], ready_event: Any) -> None:
        torch = _torch()
        started = time.monotonic()
        try:
            with torch.cuda.device(self.device):
                stream = torch.cuda.Stream(device=self.device, priority=2)
                stream.wait_event(ready_event)
                full_runner = OrtSegmentRunner(
                    self.graph.model.SerializeToString(),
                    self.options.device_id,
                    stream,
                    self.options.fallback,
                    self.options.reject_cpu_fallback,
                )
                segment_runners = {
                    segment.index: OrtSegmentRunner(
                        segment.model_bytes,
                        self.options.device_id,
                        stream,
                        self.options.fallback,
                        self.options.reject_cpu_fallback,
                    )
                    for segment in self.segments
                }

                def hybrid() -> dict[str, Any]:
                    values = {**self.initializers, **inputs}
                    for segment in self.segments:
                        segment_inputs = {name: values[name] for name in segment.inputs}
                        decision = (
                            self.cache.get(_segment_signature(self, segment, segment_inputs))
                            if segment.kind == SegmentKind.TRITON
                            else None
                        )
                        if decision is not None and decision.status == "enabled":
                            outputs = self._run_triton(segment, segment_inputs)
                        else:
                            outputs = segment_runners[segment.index].run(segment_inputs)
                        values.update(outputs)
                    return {name: values[name] for name in self.graph.outputs}

                with torch.cuda.stream(stream):
                    reference = full_runner.run(inputs)
                    candidate = hybrid()
                stream.synchronize()
                accurate, detail = _compare_outputs(candidate, reference, torch)
                if not accurate:
                    self.cache.put(key, "disabled", {"reason": "whole_plan_mismatch", **detail})
                    return
                hybrid_ms = _median_cuda_ms(hybrid, stream, torch)
                ort_ms = _median_cuda_ms(lambda: full_runner.run(inputs), stream, torch)
                enabled = hybrid_ms <= ort_ms
                self.cache.put(
                    key,
                    "enabled" if enabled else "disabled",
                    {
                        "reason": "whole_plan_gate_passed" if enabled else "whole_plan_gate_failed",
                        "hybrid_ms": hybrid_ms,
                        "ort_ms": ort_ms,
                    },
                )
        except Exception as error:
            self.stats.add("compile_failures")
            self.cache.put(key, "failed", {"reason": type(error).__name__, "message": str(error)})
            raise
        finally:
            self.stats.add("compile_seconds", time.monotonic() - started)


class ExecutionContext:
    """Owns one CUDA stream and its ORT sessions; concurrent reuse is rejected."""

    def __init__(self, engine: Engine) -> None:
        torch = _torch()
        self.engine = engine
        with torch.cuda.device(engine.device):
            self.stream = torch.cuda.Stream(device=engine.device)
            self._runners = {
                segment.index: OrtSegmentRunner(
                    segment.model_bytes,
                    engine.options.device_id,
                    self.stream,
                    engine.options.fallback,
                    engine.options.reject_cpu_fallback,
                )
                for segment in engine.segments
            }
            self._full_runner = OrtSegmentRunner(
                engine.graph.model.SerializeToString(),
                engine.options.device_id,
                self.stream,
                engine.options.fallback,
                engine.options.reject_cpu_fallback,
            )
        self._run_lock = threading.Lock()
        self._cuda_graphs: dict[str, tuple[Any, dict[str, Any], dict[str, Any]]] = {}
        self._closed = False

    def run(self, inputs: Mapping[str, Any]) -> dict[str, Any]:
        return self._run(inputs, synchronous_qualification=False)

    def _run(self, inputs: Mapping[str, Any], synchronous_qualification: bool) -> dict[str, Any]:
        if self._closed:
            raise RuntimeError("execution context is closed")
        if not self._run_lock.acquire(blocking=False):
            raise RuntimeError("ExecutionContext cannot be used concurrently")
        try:
            validated = self._validate_inputs(inputs)
            torch = _torch()
            caller_stream = torch.cuda.current_stream(self.engine.device)
            self.stream.wait_stream(caller_stream)
            values = {**self.engine.initializers, **validated}
            with torch.cuda.device(self.engine.device), torch.cuda.stream(self.stream):
                plan_key = _plan_signature(self.engine, validated)
                plan_decision = self.engine.cache.get(plan_key)
                if plan_decision is not None and plan_decision.status != "enabled":
                    self.engine.stats.add("cache_hits")
                    self.engine.stats.add("ort_segment_runs")
                    outputs = self._full_runner.run(validated)
                    if self._full_runner.cpu_fallback_observed:
                        self.engine.stats.add("cpu_fallback_segment_runs")
                    caller_stream.wait_stream(self.stream)
                    return outputs
                if (
                    plan_decision is not None
                    and plan_decision.status == "enabled"
                    and self.engine.options.cuda_graph
                    and all(item.kind == SegmentKind.TRITON for item in self.engine.segments)
                ):
                    outputs = self._run_cuda_graph(plan_key, validated, torch)
                    self.engine.stats.add("triton_segment_runs", len(self.engine.segments))
                    caller_stream.wait_stream(self.stream)
                    return outputs
                for segment in self.engine.segments:
                    segment_inputs = {name: values[name] for name in segment.inputs}
                    if segment.kind == SegmentKind.ORT:
                        self.engine.stats.add("ort_segment_runs")
                        runner = self._runners[segment.index]
                        outputs = runner.run(segment_inputs)
                        if runner.cpu_fallback_observed:
                            self.engine.stats.add("cpu_fallback_segment_runs")
                    else:
                        key = self._signature(segment, segment_inputs)
                        decision = self.engine.cache.get(key)
                        if decision is not None and decision.status == "enabled":
                            self.engine.stats.add("cache_hits")
                            self.engine.stats.add("triton_segment_runs")
                            try:
                                outputs = self.engine._run_triton(segment, segment_inputs)
                            except Exception:
                                LOGGER.exception("enabled Triton path failed; falling back to ORT")
                                self.engine.stats.add("ort_segment_runs")
                                runner = self._runners[segment.index]
                                outputs = runner.run(segment_inputs)
                                if runner.cpu_fallback_observed:
                                    self.engine.stats.add("cpu_fallback_segment_runs")
                        else:
                            if decision is None:
                                self.engine.stats.add("cache_misses")
                            else:
                                self.engine.stats.add("cache_hits")
                            self.engine.stats.add("ort_segment_runs")
                            runner = self._runners[segment.index]
                            outputs = runner.run(segment_inputs)
                            if runner.cpu_fallback_observed:
                                self.engine.stats.add("cpu_fallback_segment_runs")
                            if decision is None:
                                ready_event = torch.cuda.Event()
                                ready_event.record(self.stream)
                                future = self.engine._submit_qualification(
                                    key, segment, segment_inputs, ready_event
                                )
                                if synchronous_qualification and future is not None:
                                    future.result()
                    values.update(outputs)
                if any(item.kind == SegmentKind.TRITON for item in self.engine.segments):
                    if plan_decision is None:
                        ready_event = torch.cuda.Event()
                        ready_event.record(self.stream)
                        future = self.engine._submit_plan_qualification(
                            plan_key, validated, ready_event
                        )
                        if synchronous_qualification and future is not None:
                            future.result()
            caller_stream.wait_stream(self.stream)
            return {name: values[name] for name in self.engine.graph.outputs}
        finally:
            self._run_lock.release()

    def run_numpy(self, inputs: Mapping[str, Any]) -> dict[str, Any]:
        """Convenience API with explicit host/device transfers and synchronization."""
        import numpy as np

        torch = _torch()
        tensors = {
            name: torch.as_tensor(np.asarray(value)).to(self.engine.device).contiguous()
            for name, value in inputs.items()
        }
        outputs = self.run(tensors)
        return {name: value.detach().cpu().numpy() for name, value in outputs.items()}

    def _validate_inputs(self, inputs: Mapping[str, Any]) -> dict[str, Any]:
        torch = _torch()
        expected = set(self.engine.graph.inputs)
        actual = set(inputs)
        if actual != expected:
            raise InputValidationError(
                "model inputs mismatch; "
                f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
            )
        result: dict[str, Any] = {}
        for name in self.engine.graph.inputs:
            tensor = inputs[name]
            if not isinstance(tensor, torch.Tensor):
                raise InputValidationError(f"input {name!r} must be a torch.Tensor")
            if tensor.device.type != "cuda" or tensor.device.index != self.engine.options.device_id:
                raise InputValidationError(
                    f"input {name!r} must be on cuda:{self.engine.options.device_id}"
                )
            if not tensor.is_contiguous():
                raise InputValidationError(f"input {name!r} must be contiguous")
            meta = self.engine.graph.values[name]
            if tensor.dtype != _torch_dtype(meta.elem_type, torch):
                raise InputValidationError(
                    f"input {name!r} has dtype {tensor.dtype}; "
                    f"expected {_torch_dtype(meta.elem_type, torch)}"
                )
            if len(tensor.shape) != len(meta.shape):
                raise InputValidationError(f"input {name!r} rank mismatch")
            for actual_dim, expected_dim in zip(tensor.shape, meta.shape, strict=False):
                if (
                    isinstance(expected_dim, int)
                    and expected_dim > 0
                    and actual_dim != expected_dim
                ):
                    raise InputValidationError(f"input {name!r} shape does not match ONNX metadata")
            result[name] = tensor
        if self.engine.options.profiles and not any(
            _matches_profile(profile, result) for profile in self.engine.options.profiles
        ):
            raise InputValidationError("input shapes do not fit any declared ShapeProfile")
        return result

    def _signature(self, segment: GraphSegment, inputs: Mapping[str, Any]) -> str:
        return _segment_signature(self.engine, segment, inputs)

    def _run_cuda_graph(self, key: str, inputs: Mapping[str, Any], torch: Any) -> dict[str, Any]:
        captured = self._cuda_graphs.get(key)
        if captured is None:
            static_inputs = {
                name: torch.empty_like(tensor).copy_(tensor) for name, tensor in inputs.items()
            }
            # Allocator and compiled kernels are warmed before stream capture.
            self.engine._run_all_triton(static_inputs)
            self.stream.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=self.stream):
                static_outputs = self.engine._run_all_triton(static_inputs)
            captured = (graph, static_inputs, static_outputs)
            self._cuda_graphs[key] = captured
        graph, static_inputs, static_outputs = captured
        for name, tensor in inputs.items():
            static_inputs[name].copy_(tensor)
        graph.replay()
        self.engine.stats.add("cuda_graph_replays")
        # Callers own outputs; fixed graph buffers remain private to this context.
        return {name: tensor.clone() for name, tensor in static_outputs.items()}

    def close(self) -> None:
        self._closed = True
        for runner in self._runners.values():
            runner.close()
        self._runners.clear()
        if self._full_runner is not None:
            self._full_runner.close()
        self._full_runner = None
        self._cuda_graphs.clear()


def _compare_outputs(
    candidate: Mapping[str, Any], reference: Mapping[str, Any], torch: Any
) -> tuple[bool, dict[str, Any]]:
    if set(candidate) != set(reference):
        return False, {"detail": "output names differ"}
    worst = 0.0
    for name in candidate:
        left, right = candidate[name], reference[name]
        if left.shape != right.shape or left.dtype != right.dtype:
            return False, {"detail": f"output metadata differs for {name}"}
        tolerance = 1e-4
        if left.dtype == torch.float16:
            tolerance = 1e-2
        elif left.dtype == torch.bfloat16:
            tolerance = 2e-2
        difference = (left.float() - right.float()).abs()
        finite = torch.isfinite(difference)
        if finite.any():
            worst = max(worst, float(difference[finite].max().item()))
        if not torch.allclose(left, right, rtol=tolerance, atol=tolerance, equal_nan=True):
            return False, {"detail": f"output {name} exceeds tolerance", "max_abs": worst}
    return True, {"max_abs": worst}


def _matches_profile(profile: ShapeProfile, inputs: Mapping[str, Any]) -> bool:
    if set(profile.inputs) != set(inputs):
        return False
    for name, tensor in inputs.items():
        shape_range = profile.inputs[name]
        if len(tensor.shape) != len(shape_range.minimum):
            return False
        if any(
            actual < low or actual > high
            for actual, low, high in zip(
                tensor.shape, shape_range.minimum, shape_range.maximum, strict=False
            )
        ):
            return False
    return True


def _median_cuda_ms(call: Any, stream: Any, torch: Any, iterations: int = 9) -> float:
    samples: list[float] = []
    with torch.cuda.stream(stream):
        for _ in range(3):
            call()
        stream.synchronize()
        for _ in range(iterations):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record(stream)
            call()
            end.record(stream)
            end.synchronize()
            samples.append(float(start.elapsed_time(end)))
    samples.sort()
    return samples[len(samples) // 2]


def _environment(engine: Engine) -> dict[str, Any]:
    torch = _torch()
    device = torch.cuda.get_device_properties(engine.options.device_id)
    try:
        import triton

        triton_version = triton.__version__
    except ImportError:
        triton_version = "unavailable"
    return {
        "gpu": {
            "uuid": engine.gpu_uuid,
            "name": device.name,
            "sm": [device.major, device.minor],
        },
        "versions": {
            "driver": engine.driver_version,
            "cuda": torch.version.cuda,
            "torch": torch.__version__,
            "triton": triton_version,
        },
    }


def _tensor_signatures(inputs: Mapping[str, Any]) -> dict[str, Any]:
    return {
        name: {
            "shape": list(tensor.shape),
            "stride": list(tensor.stride()),
            "dtype": str(tensor.dtype),
        }
        for name, tensor in sorted(inputs.items())
    }


def _hash_payload(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _segment_signature(engine: Engine, segment: GraphSegment, inputs: Mapping[str, Any]) -> str:
    return _hash_payload(
        {
            "scope": "segment",
            "model": engine.graph.model_hash,
            "segment": segment.index,
            "nodes": [node.op_type for node in segment.nodes],
            "kernel_source": _kernel_source_hash(engine),
            "autotune": engine.options.autotune,
            "performance_margin": engine.options.performance_margin,
            "inputs": _tensor_signatures(inputs),
            **_environment(engine),
        }
    )


def _plan_signature(engine: Engine, inputs: Mapping[str, Any]) -> str:
    return _hash_payload(
        {
            "scope": "whole_plan",
            "model": engine.graph.model_hash,
            "kernel_source": _kernel_source_hash(engine),
            "autotune": engine.options.autotune,
            "performance_margin": engine.options.performance_margin,
            "fallback": engine.options.fallback.value,
            "inputs": _tensor_signatures(inputs),
            **_environment(engine),
        }
    )


def _kernel_source_hash(engine: Engine) -> str:
    cached = getattr(engine, "_kernel_hash", None)
    if cached is not None:
        return cached
    sources: list[str] = []
    modules: set[Any] = set()
    for spec in engine.registry.specs():
        sources.append(spec.kernel_id)
        sources.append(repr(tuple(dict(config) for config in spec.configs)))
        try:
            sources.append(inspect.getsource(spec.execute))
        except (OSError, TypeError):
            sources.append(repr(spec.execute))
        module = inspect.getmodule(spec.execute)
        if module is not None:
            modules.add(module)
    for module in sorted(modules, key=lambda value: value.__name__):
        try:
            sources.append(inspect.getsource(module))
        except (OSError, TypeError):
            sources.append(module.__name__)
    result = hashlib.sha256("\n".join(sources).encode()).hexdigest()
    engine._kernel_hash = result
    return result


def _nvidia_identity(device_id: int) -> tuple[str, str]:
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                f"--id={device_id}",
                "--query-gpu=uuid,driver_version",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=5,
        ).strip()
        uuid, driver = (item.strip() for item in output.split(",", maxsplit=1))
        return uuid, driver
    except (OSError, subprocess.SubprocessError, ValueError):
        return f"cuda-device-{device_id}", "unknown"


def compile(
    model_path: str | Path,
    options: CompileOptions | None = None,
    *,
    registry: KernelRegistry | None = None,
) -> Engine:
    """Validate and partition an ONNX model into an immutable JIT engine."""
    effective = options or CompileOptions()
    selected_registry = registry or create_default_registry(autotune=effective.autotune)
    graph = load_model(model_path, effective)
    segments = partition_graph(graph, selected_registry)
    return Engine(graph, effective, selected_registry, segments)
