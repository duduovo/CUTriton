from __future__ import annotations

from cutriton.generated import grpc_service_pb2 as pb


def test_kserve_v2_wire_messages_round_trip():
    request = pb.ModelInferRequest(model_name="resnet50", id="request-1")
    request.inputs.add(name="input", datatype="FP32", shape=[1, 3, 224, 224])
    request.raw_input_contents.append(b"payload")
    decoded = pb.ModelInferRequest.FromString(request.SerializeToString())
    assert decoded.model_name == "resnet50"
    assert decoded.inputs[0].shape == [1, 3, 224, 224]
