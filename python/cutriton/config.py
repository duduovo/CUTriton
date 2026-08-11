from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


class FallbackPolicy(str, Enum):
    ORT_CUDA_CPU = "ort_cuda_cpu"
    ORT_CUDA = "ort_cuda"


@dataclass(frozen=True)
class ShapeRange:
    minimum: tuple[int, ...]
    optimum: tuple[int, ...]
    maximum: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.minimum or not (len(self.minimum) == len(self.optimum) == len(self.maximum)):
            raise ValueError("shape range ranks must be equal and non-empty")
        for low, opt, high in zip(self.minimum, self.optimum, self.maximum, strict=False):
            if low <= 0 or not low <= opt <= high:
                raise ValueError("shape range must satisfy 0 < min <= opt <= max")


@dataclass(frozen=True)
class ShapeProfile:
    name: str
    inputs: Mapping[str, ShapeRange]

    def __post_init__(self) -> None:
        if not self.name or not self.inputs:
            raise ValueError("shape profile requires a name and at least one input")


@dataclass(frozen=True)
class CompileOptions:
    device_id: int = 0
    fallback: FallbackPolicy | str = FallbackPolicy.ORT_CUDA_CPU
    reject_cpu_fallback: bool = False
    cache_dir: Path | str = Path.home() / ".cache" / "cutriton"
    profiles: Sequence[ShapeProfile] = ()
    autotune: bool = True
    cuda_graph: bool = True
    performance_margin: float = 0.05
    max_model_bytes: int = 512 * 1024 * 1024
    max_external_data_bytes: int = 8 * 1024 * 1024 * 1024
    max_cache_bytes: int = 20 * 1024 * 1024 * 1024
    compile_workers: int = 1
    trusted_model_root: Path | str | None = None

    def __post_init__(self) -> None:
        policy = FallbackPolicy(self.fallback)
        object.__setattr__(self, "fallback", policy)
        object.__setattr__(self, "cache_dir", Path(self.cache_dir).expanduser())
        if self.trusted_model_root is not None:
            object.__setattr__(
                self, "trusted_model_root", Path(self.trusted_model_root).expanduser().resolve()
            )
        if self.device_id < 0:
            raise ValueError("device_id must be non-negative")
        if not 0.0 <= self.performance_margin < 1.0:
            raise ValueError("performance_margin must be in [0, 1)")
        if self.max_model_bytes <= 0 or self.max_external_data_bytes <= 0:
            raise ValueError("model size limits must be positive")
        if self.max_cache_bytes <= 0 or self.compile_workers != 1:
            raise ValueError("JIT cache must be positive and compile_workers must be 1")
        names = [profile.name for profile in self.profiles]
        if len(names) != len(set(names)):
            raise ValueError("shape profile names must be unique")


@dataclass(frozen=True)
class BatchingOptions:
    max_batch_size: int = 32
    preferred_batch_sizes: tuple[int, ...] = (8, 16, 32)
    max_queue_delay_us: int = 2_000
    max_inflight_batches: int = 2
    max_queue_size: int = 1_024

    def __post_init__(self) -> None:
        if self.max_batch_size <= 0 or self.max_queue_delay_us < 0:
            raise ValueError("invalid batching limits")
        if self.max_inflight_batches <= 0 or self.max_queue_size <= 0:
            raise ValueError("batch concurrency and queue size must be positive")
        if any(value <= 0 or value > self.max_batch_size for value in self.preferred_batch_sizes):
            raise ValueError("preferred batch sizes must be within max_batch_size")
        if tuple(sorted(set(self.preferred_batch_sizes))) != self.preferred_batch_sizes:
            raise ValueError("preferred batch sizes must be sorted and unique")


@dataclass(frozen=True)
class ServiceOptions:
    grpc_host: str = "0.0.0.0"
    grpc_port: int = 8001
    http_host: str = "0.0.0.0"
    http_port: int = 8002
    max_request_bytes: int = 256 * 1024 * 1024
    max_tensor_elements: int = 2**31
    max_dimension: int = 1_048_576
    tls_cert: Path | str | None = None
    tls_key: Path | str | None = None
    tls_client_ca: Path | str | None = None
    batching: BatchingOptions = field(default_factory=BatchingOptions)

    def __post_init__(self) -> None:
        if not 0 < self.grpc_port < 65536 or not 0 < self.http_port < 65536:
            raise ValueError("service ports must be valid")
        if self.max_request_bytes <= 0:
            raise ValueError("max_request_bytes must be positive")
        if self.max_tensor_elements <= 0 or self.max_dimension <= 0:
            raise ValueError("tensor limits must be positive")
        if (self.tls_cert is None) != (self.tls_key is None):
            raise ValueError("tls_cert and tls_key must be configured together")
        for name in ("tls_cert", "tls_key", "tls_client_ca"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, Path(value).expanduser().resolve())
