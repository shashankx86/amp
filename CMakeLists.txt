cmake_minimum_required(VERSION 3.22)
project(amp LANGUAGES C CXX)

# ---------------------------------------------------------------------------
# amp is standalone by construction: llama.cpp is *vendored* under third_party/ at a pinned
# commit (see third_party/DEPS.lock), built from source here, and linked statically. There is no
# external llama.cpp checkout, no LD_LIBRARY_PATH, and no RPATH into someone else's build tree.
# Everything lands under this build/ directory.
# ---------------------------------------------------------------------------
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(AMP_VENDOR_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/llama.cpp")
if(NOT EXISTS "${AMP_VENDOR_DIR}/CMakeLists.txt")
  message(FATAL_ERROR
    "amp: vendored llama.cpp not found at ${AMP_VENDOR_DIR}\n"
    "      run:  scripts/fetch_deps.sh")
endif()

option(AMP_WITH_CUDA "build amp with CUDA support" ON)
option(AMP_BUILD_TOOLS "build amp command line tools" ON)
option(AMP_BUILD_TESTS "build amp tests" ON)

# ---------------------------------------------------------------------------
# vendored llama.cpp
# ---------------------------------------------------------------------------
# Static, CUDA, and nothing of theirs built except the libraries we link. The flags below mirror
# the working build on this box (Release/-O3, GGML_CUDA_FA_ALL_QUANTS=ON) so amp's arithmetic is
# identical to the llama-server baseline it is compared against.
set(BUILD_SHARED_LIBS        OFF CACHE BOOL "" FORCE)
set(LLAMA_STANDALONE         OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_COMMON       ON  CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS        OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS        OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES     OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER       OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_APP          OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_UI           OFF CACHE BOOL "" FORCE)
set(LLAMA_USE_PREBUILT_UI    OFF CACHE BOOL "" FORCE)
set(LLAMA_ALL_WARNINGS       OFF CACHE BOOL "" FORCE)
set(LLAMA_ALL_WARNINGS_3RD_PARTY OFF CACHE BOOL "" FORCE)
set(LLAMA_FATAL_WARNINGS     OFF CACHE BOOL "" FORCE)
set(LLAMA_SUBPROCESS         OFF CACHE BOOL "" FORCE)
set(LLAMA_OPENSSL            OFF CACHE BOOL "" FORCE)
set(GGML_CUDA_FA_ALL_QUANTS  ON  CACHE BOOL "" FORCE)
if(AMP_WITH_CUDA)
  set(GGML_CUDA ON CACHE BOOL "" FORCE)
endif()

add_subdirectory("${AMP_VENDOR_DIR}" "${CMAKE_BINARY_DIR}/llama.cpp" EXCLUDE_FROM_ALL)

# All of our executables in one place: build/bin
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
foreach(cfg DEBUG RELEASE RELWITHDEBINFO MINSIZEREL)
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_${cfg} "${CMAKE_BINARY_DIR}/bin")
endforeach()

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

# util also owns the JSON layer used by the API
target_sources(amp_util PRIVATE src/util/json.cpp)

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
target_include_directories(amp_model PUBLIC
  "${AMP_VENDOR_DIR}/ggml/include"
  "${AMP_VENDOR_DIR}/include")
target_link_libraries(amp_model PUBLIC amp_util ggml ggml-base)
target_compile_options(amp_model PRIVATE -Wall -Wextra -Wno-unused-parameter)

# plan: memory/device planning
add_library(amp_plan STATIC
  src/plan/memory_plan.cpp
  src/plan/cost_model.cpp
)
target_link_libraries(amp_plan PUBLIC amp_model amp_io)
target_compile_options(amp_plan PRIVATE -Wall -Wextra -Wno-unused-parameter)

# runtime: the forward path, built on llama.cpp's own kernels so the arithmetic is identical
add_library(amp_runtime STATIC
  src/runtime/model_runtime.cpp
)
target_include_directories(amp_runtime PUBLIC "${AMP_VENDOR_DIR}/include")
target_link_libraries(amp_runtime PUBLIC amp_io amp_plan llama llama-common)
target_compile_options(amp_runtime PRIVATE -Wall -Wextra -Wno-unused-parameter)

if(AMP_BUILD_TESTS)
  enable_testing()
  add_executable(amp_tests
    tests/test_main.cpp
    tests/test_gguf.cpp
    tests/test_plan.cpp
    tests/test_json.cpp
  )
  target_link_libraries(amp_tests PRIVATE amp_plan)
  target_include_directories(amp_tests PRIVATE tests)
  target_compile_options(amp_tests PRIVATE -Wall -Wextra -Wno-unused-parameter)
  add_test(NAME amp_tests COMMAND amp_tests)
endif()

message(STATUS "amp: vendored llama.cpp at ${AMP_VENDOR_DIR}")
message(STATUS "amp: CUDA ${AMP_WITH_CUDA}   binaries -> ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")
