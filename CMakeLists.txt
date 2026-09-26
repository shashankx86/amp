cmake_minimum_required(VERSION 3.22)
project(amp LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
# ---------------------------------------------------------------------------
# ggml: amp reuses the *exact* kernels llama.cpp uses, so the quantized math is
# bit-identical. We link the already-built shared libraries from the local
# llama.cpp tree rather than recompiling ggml.
# ---------------------------------------------------------------------------
set(AMP_LLAMA_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../llama.cpp" CACHE PATH "llama.cpp source root")
set(AMP_GGML_BIN_DIR "${AMP_LLAMA_ROOT}/build/bin" CACHE PATH "directory containing libggml*.so")

# amp links llama.cpp's shared libs, so bake their location into the binaries: the tools must run
# without LD_LIBRARY_PATH. (Must come after AMP_GGML_BIN_DIR is defined.)
set(CMAKE_BUILD_RPATH "${AMP_GGML_BIN_DIR}")
set(CMAKE_INSTALL_RPATH "${AMP_GGML_BIN_DIR}")

find_path(AMP_GGML_INCLUDE_DIR ggml.h
  PATHS "${AMP_LLAMA_ROOT}/ggml/include" "${AMP_LLAMA_ROOT}/ggml/src"
  NO_DEFAULT_PATH)
if(NOT AMP_GGML_INCLUDE_DIR)
  message(FATAL_ERROR "ggml.h not found under ${AMP_LLAMA_ROOT}; set -DAMP_LLAMA_ROOT")
endif()

find_library(AMP_LIBGGML      NAMES ggml      PATHS "${AMP_GGML_BIN_DIR}" NO_DEFAULT_PATH)
find_library(AMP_LIBGGML_BASE NAMES ggml-base PATHS "${AMP_GGML_BIN_DIR}" NO_DEFAULT_PATH)
find_library(AMP_LIBGGML_CPU  NAMES ggml-cpu  PATHS "${AMP_GGML_BIN_DIR}" NO_DEFAULT_PATH)
find_library(AMP_LIBGGML_CUDA NAMES ggml-cuda PATHS "${AMP_GGML_BIN_DIR}" NO_DEFAULT_PATH)
find_library(AMP_LIBLLAMA      NAMES llama      PATHS "${AMP_GGML_BIN_DIR}" NO_DEFAULT_PATH)
if(NOT AMP_LIBGGML OR NOT AMP_LIBGGML_BASE)
  message(FATAL_ERROR "libggml*.so not found in ${AMP_GGML_BIN_DIR}; build llama.cpp first")
endif()

message(STATUS "amp: ggml include = ${AMP_GGML_INCLUDE_DIR}")
message(STATUS "amp: ggml libs    = ${AMP_LIBGGML}, ${AMP_LIBGGML_BASE}"
               "${AMP_LIBGGML_CPU}" "${AMP_LIBGGML_CUDA}")

# ---------------------------------------------------------------------------
# options
# ---------------------------------------------------------------------------
option(AMP_BUILD_TOOLS "build amp command line tools" ON)
option(AMP_BUILD_TESTS "build amp tests" ON)
option(AMP_WITH_URING  "use io_uring for streaming reads" ON)

# ---------------------------------------------------------------------------
# libraries (one per module, so dependencies stay explicit)
# ---------------------------------------------------------------------------
add_library(amp_util STATIC
  src/format.cpp
  src/bytes.cpp
  src/log.cpp
  src/timing.cpp
)
target_include_directories(amp_util PUBLIC include)
target_compile_options(amp_util PRIVATE -Wall -Wextra -Wno-unused-parameter)

# io: mmap + page-cache warming + streaming reads
add_library(amp_io STATIC
  src/io/mapped_file.cpp
  src/io/warmer.cpp
  src/io/stream_reader.cpp
)
target_link_libraries(amp_io PUBLIC amp_util)
target_compile_options(amp_io PRIVATE -Wall -Wextra -Wno-unused-parameter)

# model: GGUF parsing + derived geometry
add_library(amp_model STATIC
  src/model/gguf.cpp
  src/model/geometry.cpp
)
target_include_directories(amp_model PUBLIC ${AMP_GGML_INCLUDE_DIR})
target_link_libraries(amp_model PUBLIC amp_util ${AMP_LIBGGML} ${AMP_LIBGGML_BASE})
target_compile_options(amp_model PRIVATE -Wall -Wextra -Wno-unused-parameter)

# runtime: the forward path, built on llama.cpp's public API so the arithmetic is identical
add_library(amp_runtime STATIC
  src/runtime/model_runtime.cpp
)
target_include_directories(amp_runtime PUBLIC ${AMP_LLAMA_ROOT}/include)
target_link_libraries(amp_runtime PUBLIC amp_io amp_plan ${AMP_LIBLLAMA} ${AMP_LIBGGML} ${AMP_LIBGGML_BASE})
target_compile_options(amp_runtime PRIVATE -Wall -Wextra -Wno-unused-parameter)

# plan: memory/device planning
add_library(amp_plan STATIC
  src/plan/memory_plan.cpp
  src/plan/cost_model.cpp
)
target_link_libraries(amp_plan PUBLIC amp_model amp_io)
target_compile_options(amp_plan PRIVATE -Wall -Wextra -Wno-unused-parameter)

if(AMP_BUILD_TOOLS)
  foreach(tool amp-plan amp-warm amp-infer)
    string(REPLACE "-" "_" tool_src ${tool})
    add_executable(${tool} tools/${tool_src}.cpp)
    if(${tool} STREQUAL "amp-infer")
      target_link_libraries(${tool} PRIVATE amp_runtime)
    else()
      target_link_libraries(${tool} PRIVATE amp_plan)
    endif()
    target_compile_options(${tool} PRIVATE -Wall -Wextra -Wno-unused-parameter)
  endforeach()
endif()

if(AMP_BUILD_TESTS)
  enable_testing()
  add_executable(amp_tests
    tests/test_main.cpp
    tests/test_gguf.cpp
    tests/test_plan.cpp
  )
  target_link_libraries(amp_tests PRIVATE amp_plan)
  target_include_directories(amp_tests PRIVATE tests)
  target_compile_options(amp_tests PRIVATE -Wall -Wextra -Wno-unused-parameter)
  add_test(NAME amp_tests COMMAND amp_tests)
endif()
