cmake_minimum_required(VERSION 3.20)

project(CUTriton VERSION 0.1.0 LANGUAGES CXX)

option(CUTRITON_BUILD_TESTS "构建 CUTriton C++ 测试" ON)
option(CUTRITON_BUILD_BENCHMARKS "构建 CUTriton 性能基准" ON)
option(CUTRITON_ENABLE_CUDA "启用 CUDA Runtime 实现钩子" OFF)

option(BUILD_SHARED_LIBS "Build CUTriton as shared libraries" OFF)

add_library(cutriton
  src/backend/backend.cpp
  src/backend/cuda_launcher.cpp
  src/backend/kernel_artifact.cpp
  src/compiler/compiler.cpp
  src/core/buffer.cpp
  src/core/status.cpp
  src/core/tensor.cpp
  src/ir/graph.cpp
  src/ir/fusion.cpp
  src/ir/op_schema.cpp
  src/ir/pass.cpp
  src/runtime/engine.cpp
  src/runtime/memory_planner.cpp
  src/runtime/profiler.cpp
)

target_include_directories(cutriton
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_compile_features(cutriton PUBLIC cxx_std_17)
set_target_properties(cutriton PROPERTIES POSITION_INDEPENDENT_CODE ON)

if(WIN32 AND BUILD_SHARED_LIBS)
  set_target_properties(cutriton PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)
endif()

if(MSVC)
  target_compile_options(cutriton PRIVATE /W4 /permissive-)
else()
  target_compile_options(cutriton PRIVATE -Wall -Wextra -Wpedantic)
endif()

if(CUTRITON_ENABLE_CUDA)
  find_package(CUDAToolkit REQUIRED)
  find_package(Python3 3.10 COMPONENTS Interpreter REQUIRED)
  execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import triton; print(triton.__version__)"
    RESULT_VARIABLE CUTRITON_TRITON_CHECK
    ERROR_VARIABLE CUTRITON_TRITON_ERROR
  )
  if(NOT CUTRITON_TRITON_CHECK EQUAL 0)
    message(FATAL_ERROR "CUTRITON_ENABLE_CUDA requires an importable Triton installation: ${CUTRITON_TRITON_ERROR}")
  endif()
  set(CUTRITON_KERNEL_ARTIFACT_DIR
      "${CMAKE_BINARY_DIR}/triton_kernels" CACHE PATH
      "Directory for generated Triton PTX and manifest")
  add_custom_command(
    OUTPUT "${CUTRITON_KERNEL_ARTIFACT_DIR}/pack.json"
    COMMAND ${Python3_EXECUTABLE}
            "${CMAKE_CURRENT_SOURCE_DIR}/tools/build_triton_kernels.py"
            --output "${CUTRITON_KERNEL_ARTIFACT_DIR}"
    DEPENDS
      "${CMAKE_CURRENT_SOURCE_DIR}/tools/build_triton_kernels.py"
      "${CMAKE_CURRENT_SOURCE_DIR}/python/cutriton/kernel_sdk/spec.py"
      "${CMAKE_CURRENT_SOURCE_DIR}/python/cutriton/kernel_sdk/builder.py"
      "${CMAKE_CURRENT_SOURCE_DIR}/python/cutriton/triton_kernels/resnet.py"
      "${CMAKE_CURRENT_SOURCE_DIR}/python/cutriton/triton_kernels/transformer.py"
    COMMENT "Compiling CUTriton FP32 Triton kernels to PTX"
    VERBATIM
  )
  add_custom_target(cutriton_triton_kernels
    DEPENDS "${CUTRITON_KERNEL_ARTIFACT_DIR}/pack.json")
  add_dependencies(cutriton cutriton_triton_kernels)
  target_link_libraries(cutriton PRIVATE CUDA::cuda_driver)
  target_compile_definitions(cutriton PUBLIC CUTRITON_ENABLE_CUDA=1)
else()
  target_compile_definitions(cutriton PUBLIC CUTRITON_ENABLE_CUDA=0)
endif()

if(CUTRITON_ENABLE_CUDA AND CUTRITON_BUILD_BENCHMARKS)
  add_executable(cutriton_resnet50_benchmark
    benchmarks/resnet50_native.cpp
  )
  target_link_libraries(cutriton_resnet50_benchmark
    PRIVATE cutriton CUDA::cuda_driver)
  target_compile_definitions(cutriton_resnet50_benchmark PRIVATE
    CUTRITON_BENCHMARK_KERNEL_DIR="${CUTRITON_KERNEL_ARTIFACT_DIR}")

  add_executable(cutriton_transformer_ffn_benchmark
    benchmarks/transformer_ffn_native.cpp
  )
  target_link_libraries(cutriton_transformer_ffn_benchmark
    PRIVATE cutriton CUDA::cuda_driver)
  target_compile_definitions(cutriton_transformer_ffn_benchmark PRIVATE
    CUTRITON_BENCHMARK_KERNEL_DIR="${CUTRITON_KERNEL_ARTIFACT_DIR}")
endif()

if(CUTRITON_BUILD_TESTS)
  enable_testing()

  add_executable(cutriton_core_tests
    tests/cpp/test_core.cpp
  )
  target_link_libraries(cutriton_core_tests PRIVATE cutriton)
  add_test(NAME cutriton_core_tests COMMAND cutriton_core_tests)

  if(CUTRITON_ENABLE_CUDA)
    add_executable(cutriton_cuda_tests
      tests/cpp/test_cuda.cpp
    )
    target_link_libraries(cutriton_cuda_tests PRIVATE cutriton CUDA::cuda_driver)
    target_compile_definitions(cutriton_cuda_tests PRIVATE
      CUTRITON_TEST_KERNEL_DIR="${CUTRITON_KERNEL_ARTIFACT_DIR}")
    add_test(NAME cutriton_cuda_tests COMMAND cutriton_cuda_tests)
    set_tests_properties(cutriton_cuda_tests PROPERTIES LABELS "gpu")
    add_test(NAME cutriton_cuda_pytorch_reference
      COMMAND ${Python3_EXECUTABLE}
              "${CMAKE_CURRENT_SOURCE_DIR}/tests/python/test_cuda_reference.py"
              --executable $<TARGET_FILE:cutriton_cuda_tests>)
    set_tests_properties(cutriton_cuda_pytorch_reference PROPERTIES LABELS "gpu")
    if(CUTRITON_BUILD_BENCHMARKS)
      add_test(NAME cutriton_resnet50_pytorch_reference
        COMMAND ${Python3_EXECUTABLE}
                "${CMAKE_CURRENT_SOURCE_DIR}/benchmarks/resnet50_compare.py"
                --executable $<TARGET_FILE:cutriton_resnet50_benchmark>
                --warmup 1 --iterations 1)
      set_tests_properties(cutriton_resnet50_pytorch_reference PROPERTIES
        LABELS "gpu;benchmark"
        TIMEOUT 120)
      add_test(NAME cutriton_transformer_ffn_reference
        COMMAND ${Python3_EXECUTABLE}
                "${CMAKE_CURRENT_SOURCE_DIR}/benchmarks/transformer_ffn_compare.py"
                --executable $<TARGET_FILE:cutriton_transformer_ffn_benchmark>
                --tokens 17 --warmup 2 --iterations 3)
      set_tests_properties(cutriton_transformer_ffn_reference PROPERTIES
        LABELS "gpu;benchmark"
        TIMEOUT 120)
    endif()
  endif()
endif()
