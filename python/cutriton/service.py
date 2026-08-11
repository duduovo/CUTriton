from __future__ import annotations

import argparse
import asyncio
import json
import logging
import signal
import ssl
import time
from pathlib import Path
from typing import Any

from .api import compile as compile_model
from .batching import DynamicBatcher
from .config import CompileOptions, ServiceOptions
from .errors import InputValidationError, ServiceOverloadedError

LOGGER = logging.getLogger("cutriton.service")


def _protocol() -> tuple[Any, Any, Any]:
    try:
        import grpc

        from .generated import grpc_service_pb2, grpc_service_pb2_grpc
    except ImportError as error:
        raise RuntimeError("service extras are required: pip install cutriton[service]") from error
    return grpc, grpc_service_pb2, grpc_service_pb2_grpc


class _Metrics:
    def __init__(self) -> None:
        try:
            from prometheus_client import Counter, Gauge, Histogram

            self.requests = Counter("cutriton_requests_total", "Inference requests", ["status"])
            self.latency = Histogram("cutriton_request_seconds", "End-to-end inference latency")
            self.inflight = Gauge("cutriton_inflight_requests", "In-flight inference requests")
            self.batch = Histogram("cutriton_batch_size", "Request batch sizes")
        except ImportError:
            self.requests = self.latency = self.inflight = self.batch = None


class InferenceService:
    def __init__(self, engine: Any, model_name: str, options: ServiceOptions) -> None:
        self.engine = engine
        self.model_name = model_name
        self.options = options
        self.batcher = DynamicBatcher(engine, options.batching)
        self.metrics = _Metrics()
        self.live = True
        self.ready = True
        self.draining = False

    async def ServerLive(self, request: Any, context: Any) -> Any:
        _, pb, _ = _protocol()
        return pb.ServerLiveResponse(live=self.live)

    async def ServerReady(self, request: Any, context: Any) -> Any:
        _, pb, _ = _protocol()
        return pb.ServerReadyResponse(ready=self.ready and not self.draining)

    async def ModelReady(self, request: Any, context: Any) -> Any:
        _, pb, _ = _protocol()
        return pb.ModelReadyResponse(
            ready=request.name == self.model_name and self.ready and not self.draining
        )

    async def ServerMetadata(self, request: Any, context: Any) -> Any:
        _, pb, _ = _protocol()
        return pb.ServerMetadataResponse(
            name="cutriton", version="0.2.0", extensions=["binary_tensor_data"]
        )

    async def ModelMetadata(self, request: Any, context: Any) -> Any:
        grpc, pb, _ = _protocol()
        if request.name != self.model_name:
            await context.abort(grpc.StatusCode.NOT_FOUND, "unknown model")
        graph = self.engine.graph
        return pb.ModelMetadataResponse(
            name=self.model_name,
            versions=["1"],
            platform="onnx_cutriton",
            inputs=[
                pb.ModelMetadataResponse.TensorMetadata(
                    name=name,
                    datatype=_kserve_dtype(graph.values[name].elem_type),
                    shape=[
                        dimension if isinstance(dimension, int) else -1
                        for dimension in graph.values[name].shape
                    ],
                )
                for name in graph.inputs
            ],
            outputs=[
                pb.ModelMetadataResponse.TensorMetadata(
                    name=name,
                    datatype=_kserve_dtype(graph.values[name].elem_type),
                    shape=[
                        dimension if isinstance(dimension, int) else -1
                        for dimension in graph.values[name].shape
                    ],
                )
                for name in graph.outputs
            ],
        )

    async def ModelInfer(self, request: Any, context: Any) -> Any:
        grpc, pb, _ = _protocol()
        started = time.monotonic()
        if request.model_name != self.model_name:
            await context.abort(grpc.StatusCode.NOT_FOUND, "unknown model")
        if self.draining:
            await context.abort(grpc.StatusCode.UNAVAILABLE, "service is draining")
        if not request.raw_input_contents or len(request.raw_input_contents) != len(request.inputs):
            await context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                "CUTriton requires one raw_input_contents entry per input",
            )
        if sum(len(item) for item in request.raw_input_contents) > self.options.max_request_bytes:
            await context.abort(grpc.StatusCode.RESOURCE_EXHAUSTED, "request is too large")
        if self.metrics.inflight:
            self.metrics.inflight.inc()
        try:
            tensors = _decode_inputs(request, self.engine.device, self.options)
            deadline = None
            remaining = context.time_remaining()
            if remaining is not None:
                deadline = time.monotonic() + max(0.0, remaining)
            outputs = await self.batcher.infer(tensors, deadline=deadline)
            response = pb.ModelInferResponse(
                model_name=self.model_name, model_version="1", id=request.id
            )
            requested = {item.name for item in request.outputs}
            for name, tensor in outputs.items():
                if requested and name not in requested:
                    continue
                response.outputs.add(
                    name=name,
                    datatype=_torch_kserve_dtype(tensor.dtype),
                    shape=list(tensor.shape),
                )
                response.raw_output_contents.append(_encode_output(tensor))
            if self.metrics.requests:
                self.metrics.requests.labels("ok").inc()
                self.metrics.batch.observe(next(iter(tensors.values())).shape[0])
            LOGGER.info(
                "inference completed",
                extra={
                    "request_id": request.id,
                    "duration_ms": (time.monotonic() - started) * 1_000,
                    "batch_size": int(next(iter(tensors.values())).shape[0]),
                },
            )
            return response
        except (InputValidationError, ValueError) as error:
            if self.metrics.requests:
                self.metrics.requests.labels("invalid").inc()
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(error))
        except TimeoutError as error:
            if self.metrics.requests:
                self.metrics.requests.labels("deadline").inc()
            await context.abort(grpc.StatusCode.DEADLINE_EXCEEDED, str(error))
        except ServiceOverloadedError as error:
            if self.metrics.requests:
                self.metrics.requests.labels("overloaded").inc()
            await context.abort(grpc.StatusCode.RESOURCE_EXHAUSTED, str(error))
        except Exception as error:
            if type(error).__name__ == "OutOfMemoryError":
                self.engine.stats.add("oom_total")
                if self.metrics.requests:
                    self.metrics.requests.labels("oom").inc()
                await context.abort(grpc.StatusCode.RESOURCE_EXHAUSTED, "GPU out of memory")
            LOGGER.exception("inference request failed")
            if self.metrics.requests:
                self.metrics.requests.labels("error").inc()
            await context.abort(grpc.StatusCode.INTERNAL, str(error))
        finally:
            if self.metrics.inflight:
                self.metrics.inflight.dec()
                self.metrics.latency.observe(time.monotonic() - started)


async def _admin_app(service: InferenceService) -> Any:
    from aiohttp import web

    async def health(request: Any) -> Any:
        path = request.path
        healthy = service.live if path == "/live" else service.ready and not service.draining
        return web.json_response({"ok": healthy}, status=200 if healthy else 503)

    async def metadata(request: Any) -> Any:
        return web.json_response(service.engine.report.to_dict())

    async def warmup(request: Any) -> Any:
        await asyncio.to_thread(service.engine.warmup)
        return web.json_response({"warmed": True})

    async def drain(request: Any) -> Any:
        service.draining = True
        service.ready = False
        await service.batcher.drain()
        return web.json_response({"drained": True})

    async def metrics(request: Any) -> Any:
        import torch

        try:
            from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

            body = generate_latest().decode()
        except ImportError:
            body = ""
            CONTENT_TYPE_LATEST = "text/plain; version=0.0.4"
        stats = service.engine.stats.snapshot()
        stats["gpu_memory_allocated_bytes"] = float(
            torch.cuda.memory_allocated(service.engine.options.device_id)
        )
        stats["gpu_memory_reserved_bytes"] = float(
            torch.cuda.memory_reserved(service.engine.options.device_id)
        )
        body += "".join(f"cutriton_{name} {value}\n" for name, value in stats.items())
        return web.Response(body=body.encode(), headers={"Content-Type": CONTENT_TYPE_LATEST})

    app = web.Application(client_max_size=service.options.max_request_bytes)
    app.add_routes(
        [
            web.get("/live", health),
            web.get("/ready", health),
            web.get("/metrics", metrics),
            web.get("/metadata", metadata),
            web.post("/warmup", warmup),
            web.post("/drain", drain),
        ]
    )
    return app


async def serve(engine: Any, model_name: str, options: ServiceOptions | None = None) -> None:
    from aiohttp import web

    grpc, _, pb_grpc = _protocol()
    effective = options or ServiceOptions()
    service = InferenceService(engine, model_name, effective)
    await service.batcher.start()
    grpc_server = grpc.aio.server(
        options=[
            ("grpc.max_receive_message_length", effective.max_request_bytes),
            ("grpc.max_send_message_length", effective.max_request_bytes),
        ]
    )
    pb_grpc.add_GRPCInferenceServiceServicer_to_server(service, grpc_server)
    grpc_address = f"{effective.grpc_host}:{effective.grpc_port}"
    ssl_context = None
    if effective.tls_cert is not None:
        private_key = effective.tls_key.read_bytes()
        certificate = effective.tls_cert.read_bytes()
        client_ca = effective.tls_client_ca.read_bytes() if effective.tls_client_ca else None
        credentials = grpc.ssl_server_credentials(
            [(private_key, certificate)],
            root_certificates=client_ca,
            require_client_auth=client_ca is not None,
        )
        grpc_server.add_secure_port(grpc_address, credentials)
        ssl_context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
        ssl_context.load_cert_chain(effective.tls_cert, effective.tls_key)
        if effective.tls_client_ca:
            ssl_context.load_verify_locations(effective.tls_client_ca)
            ssl_context.verify_mode = ssl.CERT_REQUIRED
    else:
        grpc_server.add_insecure_port(grpc_address)
    runner = web.AppRunner(await _admin_app(service))
    await runner.setup()
    site = web.TCPSite(runner, effective.http_host, effective.http_port, ssl_context=ssl_context)
    await grpc_server.start()
    await site.start()
    LOGGER.info(
        "CUTriton service ready",
        extra={"grpc_port": effective.grpc_port, "http_port": effective.http_port},
    )
    stopped = asyncio.Event()
    loop = asyncio.get_running_loop()
    for signum in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(signum, stopped.set)
        except NotImplementedError:
            pass
    await stopped.wait()
    service.draining = True
    service.ready = False
    await service.batcher.drain()
    await grpc_server.stop(grace=30.0)
    await service.batcher.close()
    await runner.cleanup()


def _decode_inputs(request: Any, device: Any, options: ServiceOptions) -> dict[str, Any]:
    import numpy as np
    import torch

    result: dict[str, Any] = {}
    for metadata, raw in zip(request.inputs, request.raw_input_contents, strict=False):
        numpy_dtype = _numpy_dtype(metadata.datatype, np)
        count = 1
        for dimension in metadata.shape:
            if dimension <= 0 or dimension > options.max_dimension:
                raise ValueError("request dimension is outside the configured range")
            count *= dimension
        if count > options.max_tensor_elements:
            raise ValueError("request tensor element limit exceeded")
        array = np.frombuffer(raw, dtype=numpy_dtype)
        if array.size != count:
            raise ValueError(f"raw byte count does not match shape for {metadata.name!r}")
        pinned_dtype = (
            torch.bfloat16
            if metadata.datatype == "BF16"
            else _torch_dtype_name(metadata.datatype, torch)
        )
        pinned = torch.empty(tuple(metadata.shape), dtype=pinned_dtype, pin_memory=True)
        source = torch.from_numpy(array.copy()).reshape(tuple(metadata.shape))
        if metadata.datatype == "BF16":
            source = source.view(torch.bfloat16)
        pinned.copy_(source)
        result[metadata.name] = pinned.to(device, non_blocking=True)
    return result


def _encode_output(tensor: Any) -> bytes:
    import torch

    host = torch.empty_like(tensor, device="cpu", pin_memory=True)
    host.copy_(tensor, non_blocking=True)
    torch.cuda.current_stream(tensor.device).synchronize()
    if tensor.dtype == torch.bfloat16:
        return host.view(torch.uint16).numpy().tobytes()
    return host.numpy().tobytes()


def _numpy_dtype(name: str, np: Any) -> Any:
    mapping = {
        "BOOL": np.bool_,
        "UINT8": np.uint8,
        "INT8": np.int8,
        "UINT16": np.uint16,
        "INT16": np.int16,
        "INT32": np.int32,
        "INT64": np.int64,
        "UINT32": np.uint32,
        "UINT64": np.uint64,
        "FP16": np.float16,
        "FP32": np.float32,
        "FP64": np.float64,
        "BF16": np.uint16,
    }
    if name not in mapping:
        raise ValueError(f"unsupported KServe datatype {name!r}")
    return mapping[name]


def _torch_dtype_name(name: str, torch: Any) -> Any:
    mapping = {
        "BOOL": torch.bool,
        "UINT8": torch.uint8,
        "INT8": torch.int8,
        "UINT16": torch.uint16,
        "INT16": torch.int16,
        "INT32": torch.int32,
        "INT64": torch.int64,
        "UINT32": torch.uint32,
        "UINT64": torch.uint64,
        "FP16": torch.float16,
        "FP32": torch.float32,
        "FP64": torch.float64,
        "BF16": torch.uint16,
    }
    if name not in mapping:
        raise ValueError(f"unsupported KServe datatype {name!r}")
    return mapping[name]


def _kserve_dtype(elem_type: int) -> str:
    mapping = {
        1: "FP32",
        2: "UINT8",
        3: "INT8",
        4: "UINT16",
        5: "INT16",
        6: "INT32",
        7: "INT64",
        9: "BOOL",
        10: "FP16",
        11: "FP64",
        12: "UINT32",
        13: "UINT64",
        16: "BF16",
    }
    return mapping.get(elem_type, "BYTES")


def _torch_kserve_dtype(dtype: Any) -> str:
    import torch

    mapping = {
        torch.float32: "FP32",
        torch.float16: "FP16",
        torch.bfloat16: "BF16",
        torch.float64: "FP64",
        torch.uint8: "UINT8",
        torch.int8: "INT8",
        torch.int16: "INT16",
        torch.int32: "INT32",
        torch.int64: "INT64",
        torch.bool: "BOOL",
        torch.uint16: "UINT16",
        torch.uint32: "UINT32",
        torch.uint64: "UINT64",
    }
    return mapping[dtype]


def main() -> None:
    parser = argparse.ArgumentParser(description="CUTriton KServe V2 inference server")
    parser.add_argument("model", type=Path)
    parser.add_argument("--model-name", default="model")
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--cache-dir", type=Path, default=Path("/var/cache/cutriton"))
    parser.add_argument("--model-root", type=Path, default=Path("/models"))
    parser.add_argument("--tls-cert", type=Path)
    parser.add_argument("--tls-key", type=Path)
    parser.add_argument("--tls-client-ca", type=Path)
    args = parser.parse_args()
    import torch

    if torch.cuda.device_count() != 1:
        raise RuntimeError(
            "cutriton-serve requires exactly one visible NVIDIA GPU; "
            "configure the container runtime"
        )
    handler = logging.StreamHandler()
    handler.setFormatter(_JsonFormatter())
    logging.basicConfig(level=logging.INFO, handlers=[handler])
    engine = compile_model(
        args.model,
        CompileOptions(
            device_id=args.device_id,
            cache_dir=args.cache_dir,
            trusted_model_root=args.model_root,
        ),
    )
    try:
        engine.warmup()
        asyncio.run(
            serve(
                engine,
                args.model_name,
                ServiceOptions(
                    tls_cert=args.tls_cert,
                    tls_key=args.tls_key,
                    tls_client_ca=args.tls_client_ca,
                ),
            )
        )
    finally:
        engine.close()


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "time": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for name in ("request_id", "duration_ms", "batch_size", "grpc_port", "http_port"):
            if hasattr(record, name):
                payload[name] = getattr(record, name)
        return json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
        )


if __name__ == "__main__":
    main()
