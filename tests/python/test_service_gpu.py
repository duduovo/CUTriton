from __future__ import annotations

import asyncio

import numpy as np
import pytest

aiohttp = pytest.importorskip("aiohttp")
grpc = pytest.importorskip("grpc")
onnx = pytest.importorskip("onnx")
torch = pytest.importorskip("torch")

from cutriton import CompileOptions, compile
from cutriton.config import BatchingOptions, ServiceOptions
from cutriton.generated import grpc_service_pb2 as pb
from cutriton.generated import grpc_service_pb2_grpc as pb_grpc
from cutriton.service import InferenceService, _admin_app

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")


def _write_relu(path):
    helper = onnx.helper
    x = helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT, [None, 4])
    output = helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, [None, 4])
    graph = helper.make_graph(
        [helper.make_node("Relu", ["input"], ["output"])], "service", [x], [output]
    )
    onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)]), path)


@pytest.mark.gpu
@pytest.mark.integration
def test_kserve_grpc_and_http_admin_end_to_end(tmp_path):
    async def scenario():
        path = tmp_path / "model.onnx"
        _write_relu(path)
        engine = compile(path, CompileOptions(cache_dir=tmp_path / "cache"))
        options = ServiceOptions(
            batching=BatchingOptions(max_queue_delay_us=0, max_inflight_batches=1)
        )
        service = InferenceService(engine, "relu", options)
        await service.batcher.start()
        server = grpc.aio.server()
        pb_grpc.add_GRPCInferenceServiceServicer_to_server(service, server)
        port = server.add_insecure_port("127.0.0.1:0")
        await server.start()
        channel = grpc.aio.insecure_channel(f"127.0.0.1:{port}")
        client = pb_grpc.GRPCInferenceServiceStub(channel)
        from aiohttp.test_utils import TestClient, TestServer

        http_client = TestClient(TestServer(await _admin_app(service)))
        await http_client.start_server()
        try:
            live = await client.ServerLive(pb.ServerLiveRequest())
            assert live.live
            values = np.array([[-1.0, 2.0, 3.0, -4.0]], dtype=np.float32)
            request = pb.ModelInferRequest(model_name="relu", id="request-1")
            request.inputs.add(name="input", datatype="FP32", shape=[1, 4])
            request.raw_input_contents.append(values.tobytes())
            response = await client.ModelInfer(request)
            assert response.id == "request-1"
            actual = np.frombuffer(response.raw_output_contents[0], dtype=np.float32)
            np.testing.assert_array_equal(actual, np.maximum(values, 0).reshape(-1))

            assert (await http_client.get("/live")).status == 200
            metadata = await (await http_client.get("/metadata")).json()
            assert metadata["model_hash"] == engine.graph.model_hash
            metrics = await (await http_client.get("/metrics")).text()
            assert "cutriton_requests_total" in metrics
        finally:
            await http_client.close()
            await channel.close()
            await server.stop(grace=0)
            await service.batcher.close()
            engine.close()

    asyncio.run(scenario())
