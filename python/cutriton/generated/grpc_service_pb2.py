"""KServe V2 protobuf messages built from the checked-in grpc_service.proto schema."""

from google.protobuf import descriptor_pb2 as _descriptor_pb2
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf.internal import builder as _builder

_file = _descriptor_pb2.FileDescriptorProto(
    name="grpc_service.proto", package="inference", syntax="proto3"
)


def _message(name):
    value = _file.message_type.add()
    value.name = name
    return value


def _nested(parent, name, map_entry=False):
    value = parent.nested_type.add()
    value.name = name
    value.options.map_entry = map_entry
    return value


def _field(parent, name, number, kind, *, repeated=False, type_name=None, oneof=None):
    value = parent.field.add()
    value.name = name
    value.number = number
    value.type = kind
    value.label = 3 if repeated else 1
    if type_name:
        value.type_name = type_name
    if oneof is not None:
        value.oneof_index = oneof
    return value


TYPE = _descriptor_pb2.FieldDescriptorProto
for request_name in ("ServerLiveRequest", "ServerReadyRequest", "ServerMetadataRequest"):
    _message(request_name)

value = _message("ServerLiveResponse")
_field(value, "live", 1, TYPE.TYPE_BOOL)
value = _message("ServerReadyResponse")
_field(value, "ready", 1, TYPE.TYPE_BOOL)
value = _message("ModelReadyRequest")
_field(value, "name", 1, TYPE.TYPE_STRING)
_field(value, "version", 2, TYPE.TYPE_STRING)
value = _message("ModelReadyResponse")
_field(value, "ready", 1, TYPE.TYPE_BOOL)
value = _message("ServerMetadataResponse")
_field(value, "name", 1, TYPE.TYPE_STRING)
_field(value, "version", 2, TYPE.TYPE_STRING)
_field(value, "extensions", 3, TYPE.TYPE_STRING, repeated=True)
value = _message("ModelMetadataRequest")
_field(value, "name", 1, TYPE.TYPE_STRING)
_field(value, "version", 2, TYPE.TYPE_STRING)

value = _message("ModelMetadataResponse")
tensor_meta = _nested(value, "TensorMetadata")
_field(tensor_meta, "name", 1, TYPE.TYPE_STRING)
_field(tensor_meta, "datatype", 2, TYPE.TYPE_STRING)
_field(tensor_meta, "shape", 3, TYPE.TYPE_INT64, repeated=True)
_field(value, "name", 1, TYPE.TYPE_STRING)
_field(value, "versions", 2, TYPE.TYPE_STRING, repeated=True)
_field(value, "platform", 3, TYPE.TYPE_STRING)
_field(
    value,
    "inputs",
    4,
    TYPE.TYPE_MESSAGE,
    repeated=True,
    type_name=".inference.ModelMetadataResponse.TensorMetadata",
)
_field(
    value,
    "outputs",
    5,
    TYPE.TYPE_MESSAGE,
    repeated=True,
    type_name=".inference.ModelMetadataResponse.TensorMetadata",
)

parameter = _message("InferParameter")
parameter.oneof_decl.add().name = "parameter_choice"
_field(parameter, "bool_param", 1, TYPE.TYPE_BOOL, oneof=0)
_field(parameter, "int64_param", 2, TYPE.TYPE_INT64, oneof=0)
_field(parameter, "string_param", 3, TYPE.TYPE_STRING, oneof=0)

contents = _message("InferTensorContents")
for name, number, kind in (
    ("bool_contents", 1, TYPE.TYPE_BOOL),
    ("int_contents", 2, TYPE.TYPE_INT32),
    ("int64_contents", 3, TYPE.TYPE_INT64),
    ("uint_contents", 4, TYPE.TYPE_UINT32),
    ("uint64_contents", 5, TYPE.TYPE_UINT64),
    ("fp32_contents", 6, TYPE.TYPE_FLOAT),
    ("fp64_contents", 7, TYPE.TYPE_DOUBLE),
    ("bytes_contents", 8, TYPE.TYPE_BYTES),
):
    _field(contents, name, number, kind, repeated=True)


def _map(parent, nested_name, field_name, number):
    entry = _nested(parent, nested_name, map_entry=True)
    _field(entry, "key", 1, TYPE.TYPE_STRING)
    _field(entry, "value", 2, TYPE.TYPE_MESSAGE, type_name=".inference.InferParameter")
    _field(
        parent,
        field_name,
        number,
        TYPE.TYPE_MESSAGE,
        repeated=True,
        type_name=f".inference.{parent.name}.{nested_name}",
    )


request = _message("ModelInferRequest")
input_tensor = _nested(request, "InferInputTensor")
_field(input_tensor, "name", 1, TYPE.TYPE_STRING)
_field(input_tensor, "datatype", 2, TYPE.TYPE_STRING)
_field(input_tensor, "shape", 3, TYPE.TYPE_INT64, repeated=True)
input_map = _nested(input_tensor, "ParametersEntry", map_entry=True)
_field(input_map, "key", 1, TYPE.TYPE_STRING)
_field(input_map, "value", 2, TYPE.TYPE_MESSAGE, type_name=".inference.InferParameter")
_field(
    input_tensor,
    "parameters",
    4,
    TYPE.TYPE_MESSAGE,
    repeated=True,
    type_name=".inference.ModelInferRequest.InferInputTensor.ParametersEntry",
)
_field(input_tensor, "contents", 5, TYPE.TYPE_MESSAGE, type_name=".inference.InferTensorContents")
requested_output = _nested(request, "InferRequestedOutputTensor")
_field(requested_output, "name", 1, TYPE.TYPE_STRING)
output_map = _nested(requested_output, "ParametersEntry", map_entry=True)
_field(output_map, "key", 1, TYPE.TYPE_STRING)
_field(output_map, "value", 2, TYPE.TYPE_MESSAGE, type_name=".inference.InferParameter")
_field(
    requested_output,
    "parameters",
    2,
    TYPE.TYPE_MESSAGE,
    repeated=True,
    type_name=".inference.ModelInferRequest.InferRequestedOutputTensor.ParametersEntry",
)
_field(request, "model_name", 1, TYPE.TYPE_STRING)
_field(request, "model_version", 2, TYPE.TYPE_STRING)
_field(request, "id", 3, TYPE.TYPE_STRING)
_map(request, "ParametersEntry", "parameters", 4)
_field(
    request,
    "inputs",
    5,
    TYPE.TYPE_MESSAGE,
    repeated=True,
    type_name=".inference.ModelInferRequest.InferInputTensor",
)
_field(
    request,
    "outputs",
    6,
    TYPE.TYPE_MESSAGE,
    repeated=True,
    type_name=".inference.ModelInferRequest.InferRequestedOutputTensor",
)
_field(request, "raw_input_contents", 7, TYPE.TYPE_BYTES, repeated=True)

response = _message("ModelInferResponse")
output_tensor = _nested(response, "InferOutputTensor")
_field(output_tensor, "name", 1, TYPE.TYPE_STRING)
_field(output_tensor, "datatype", 2, TYPE.TYPE_STRING)
_field(output_tensor, "shape", 3, TYPE.TYPE_INT64, repeated=True)
response_output_map = _nested(output_tensor, "ParametersEntry", map_entry=True)
_field(response_output_map, "key", 1, TYPE.TYPE_STRING)
_field(response_output_map, "value", 2, TYPE.TYPE_MESSAGE, type_name=".inference.InferParameter")
_field(
    output_tensor,
    "parameters",
    4,
    TYPE.TYPE_MESSAGE,
    repeated=True,
    type_name=".inference.ModelInferResponse.InferOutputTensor.ParametersEntry",
)
_field(output_tensor, "contents", 5, TYPE.TYPE_MESSAGE, type_name=".inference.InferTensorContents")
_field(response, "model_name", 1, TYPE.TYPE_STRING)
_field(response, "model_version", 2, TYPE.TYPE_STRING)
_field(response, "id", 3, TYPE.TYPE_STRING)
_map(response, "ParametersEntry", "parameters", 4)
_field(
    response,
    "outputs",
    5,
    TYPE.TYPE_MESSAGE,
    repeated=True,
    type_name=".inference.ModelInferResponse.InferOutputTensor",
)
_field(response, "raw_output_contents", 6, TYPE.TYPE_BYTES, repeated=True)

service = _file.service.add()
service.name = "GRPCInferenceService"
for name, request_type, response_type in (
    ("ServerLive", "ServerLiveRequest", "ServerLiveResponse"),
    ("ServerReady", "ServerReadyRequest", "ServerReadyResponse"),
    ("ModelReady", "ModelReadyRequest", "ModelReadyResponse"),
    ("ServerMetadata", "ServerMetadataRequest", "ServerMetadataResponse"),
    ("ModelMetadata", "ModelMetadataRequest", "ModelMetadataResponse"),
    ("ModelInfer", "ModelInferRequest", "ModelInferResponse"),
):
    method = service.method.add()
    method.name = name
    method.input_type = f".inference.{request_type}"
    method.output_type = f".inference.{response_type}"

DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(_file.SerializeToString())
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, globals())
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, __name__, globals())
