"""gRPC bindings for the KServe V2 inference service."""

import grpc

from . import grpc_service_pb2 as grpc__service__pb2


class GRPCInferenceServiceStub:
    def __init__(self, channel):
        for name, request, response in _METHODS:
            setattr(
                self,
                name,
                channel.unary_unary(
                    f"/inference.GRPCInferenceService/{name}",
                    request_serializer=request.SerializeToString,
                    response_deserializer=response.FromString,
                ),
            )


class GRPCInferenceServiceServicer:
    pass


_METHODS = (
    ("ServerLive", grpc__service__pb2.ServerLiveRequest, grpc__service__pb2.ServerLiveResponse),
    ("ServerReady", grpc__service__pb2.ServerReadyRequest, grpc__service__pb2.ServerReadyResponse),
    ("ModelReady", grpc__service__pb2.ModelReadyRequest, grpc__service__pb2.ModelReadyResponse),
    (
        "ServerMetadata",
        grpc__service__pb2.ServerMetadataRequest,
        grpc__service__pb2.ServerMetadataResponse,
    ),
    (
        "ModelMetadata",
        grpc__service__pb2.ModelMetadataRequest,
        grpc__service__pb2.ModelMetadataResponse,
    ),
    ("ModelInfer", grpc__service__pb2.ModelInferRequest, grpc__service__pb2.ModelInferResponse),
)


def add_GRPCInferenceServiceServicer_to_server(servicer, server):
    handlers = {
        name: grpc.unary_unary_rpc_method_handler(
            getattr(servicer, name),
            request_deserializer=request.FromString,
            response_serializer=response.SerializeToString,
        )
        for name, request, response in _METHODS
    }
    server.add_generic_rpc_handlers(
        (grpc.method_handlers_generic_handler("inference.GRPCInferenceService", handlers),)
    )
