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
# ON only to get tools/{ui,server,mtmd} defined: llama.cpp's root CMakeLists gates the whole tools/
# tree on LLAMA_BUILD_TOOLS, not on LLAMA_BUILD_SERVER. We build named targets only, so the other
# tool targets (llama-bench, perplexity, ...) are defined but never compiled.
set(LLAMA_BUILD_TOOLS        ON  CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES     OFF CACHE BOOL "" FORCE)
# ON so llama.cpp's own tools/CMakeLists.txt adds tools/{ui,cli,server,mtmd}. We do not build the
# other tool targets (build.sh names our targets explicitly); we only want the `llama-server-impl`
# and `server-context` libraries, which tools/server/CMakeLists.txt documents as "reusable by app".
# Going through llama.cpp's own tools/CMakeLists.txt matters: those files use ${PROJECT_SOURCE_DIR},
# which is only correct if llama.cpp's root CMakeLists is the one doing the add_subdirectory.
set(LLAMA_BUILD_SERVER       ON  CACHE BOOL "" FORCE)
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

# preflight: the whole of amp's contribution to amp-server. Its own library rather than part of
# amp_plan, because it needs llama + llama-common (common_params, tensor_buft_overrides) and
# amp-plan/amp-warm should not have to link either of those just to print a plan.
add_library(amp_preflight STATIC
  src/plan/preflight.cpp
)
target_include_directories(amp_preflight PUBLIC "${AMP_VENDOR_DIR}/common" "${AMP_VENDOR_DIR}/include")
target_link_libraries(amp_preflight PUBLIC amp_plan llama llama-common)
target_compile_options(amp_preflight PRIVATE -Wall -Wextra -Wno-unused-parameter)

# runtime: the forward path, built on llama.cpp's own kernels so the arithmetic is identical
add_library(amp_runtime STATIC
  src/runtime/model_runtime.cpp
  src/runtime/buft_overrides.cpp
)
target_include_directories(amp_runtime PUBLIC "${AMP_VENDOR_DIR}/include")
target_link_libraries(amp_runtime PUBLIC amp_io amp_plan llama llama-common)
target_compile_options(amp_runtime PRIVATE -Wall -Wextra -Wno-unused-parameter)

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

  # amp-server: llama.cpp's tools/server, with amp's memory plan applied as a preflight.
  # llama-server-impl is the static library tools/server/CMakeLists.txt documents as
  # "reusable by app"; it is where the exported llama_server() lives. amp_plan supplies the
  # planner that becomes common_params fields. The old hand-rolled server library is gone:
  # its sources stay on disk for one commit as a fallback, but nothing builds them.
  add_executable(amp-server tools/amp_server.cpp)
  target_include_directories(amp-server PRIVATE "${AMP_VENDOR_DIR}/common" "${AMP_VENDOR_DIR}/include")
  target_link_libraries(amp-server PRIVATE amp_preflight llama-server-impl)
  target_compile_options(amp-server PRIVATE -Wall -Wextra -Wno-unused-parameter)

  # kernel-bound: a measurement tool, not part of the serving path. It answers whether the MoE
  # expert matvec is limited by memory or by arithmetic, which is what decides whether a
  # prefetcher or a cheaper dequant kernel is the thing worth writing.
  add_executable(amp-kernel-bound tools/kernel_bound.cpp)
  target_include_directories(amp-kernel-bound PRIVATE "${AMP_VENDOR_DIR}/include" "${AMP_VENDOR_DIR}/ggml/include")
  target_link_libraries(amp-kernel-bound PRIVATE llama ggml ggml-cpu)
  target_compile_options(amp-kernel-bound PRIVATE -Wall -Wextra -Wno-unused-parameter)

  # page-walk: a measurement tool, not part of the serving path. It answers whether the CPU
  # expert read is limited by DRAM or by address translation, which decides whether page size
  # is a lever at all on a 12.19 GiB mmap'd working set.
  add_executable(amp-page-walk tools/page_walk.cpp)
  target_link_libraries(amp-page-walk PRIVATE amp_util)
  target_compile_options(amp-page-walk PRIVATE -Wall -Wextra -Wno-unused-parameter)

  # fa-bench: a measurement tool, not part of the serving path. It times one decode
  # attention alone, with nothing else in the graph, so kernel changes can be compared
  # without the page-cache variance that swamps an end-to-end A/B on this box.
  add_executable(amp-fa-bench tools/fa_bench.cpp)
  target_include_directories(amp-fa-bench PRIVATE "${AMP_VENDOR_DIR}/include" "${AMP_VENDOR_DIR}/ggml/include")
  target_link_libraries(amp-fa-bench PRIVATE llama ggml ggml-cuda)
  target_compile_options(amp-fa-bench PRIVATE -Wall -Wextra -Wno-unused-parameter)

  # gdn-bench: a measurement tool, not part of the serving path. Sizes the 30 recurrent
  # layers' per-token state traffic against what the card can deliver, so it is visible
  # whether a fused GDN kernel has anything to win.
  add_executable(amp-gdn-bench tools/gdn_bench.cpp)
  target_link_libraries(amp-gdn-bench PRIVATE amp_model amp_io amp_util llama ggml ggml-cuda)
  target_compile_options(amp-gdn-bench PRIVATE -Wall -Wextra -Wno-unused-parameter)

  # budget: a measurement tool, not part of the serving path. Prints an upper bound on what
  # each remaining candidate could win, so the next one is picked by the size of its prize
  # rather than by how plausible it sounds.
  add_executable(amp-budget tools/budget.cpp)
  target_link_libraries(amp-budget PRIVATE amp_model amp_io amp_util llama ggml ggml-cuda)
  target_compile_options(amp-budget PRIVATE -Wall -Wextra -Wno-unused-parameter)
endif()

if(AMP_BUILD_TESTS)
  enable_testing()
  add_executable(amp_tests
    tests/test_main.cpp
    tests/test_gguf.cpp
    tests/test_plan.cpp
    tests/test_preflight.cpp
  )
  # amp_preflight, not just amp_plan: the "never override an explicit flag" guarantee is a hard
  # constraint and needs a test. It pulls in llama-common, which the test binary needs anyway to
  # construct a common_params the way common_params_parse would leave one.
  target_link_libraries(amp_tests PRIVATE amp_preflight)
  target_include_directories(amp_tests PRIVATE tests)
  target_compile_options(amp_tests PRIVATE -Wall -Wextra -Wno-unused-parameter)
  add_test(NAME amp_tests COMMAND amp_tests)
endif()

# Reported as system_fingerprint on every response, the way llama.cpp stamps llama_build_info().
execute_process(COMMAND git -C "${CMAKE_CURRENT_SOURCE_DIR}" rev-parse --short HEAD
                OUTPUT_VARIABLE AMP_GIT_SHA OUTPUT_STRIP_TRAILING_WHITESPACE
                ERROR_QUIET)
if(NOT AMP_GIT_SHA)
  set(AMP_GIT_SHA "unknown")
endif()
string(TIMESTAMP AMP_BUILD_DATE "%Y-%m-%d" UTC)
target_compile_definitions(amp_util PUBLIC
  AMP_VERSION="${AMP_GIT_SHA}+${AMP_BUILD_DATE}")

message(STATUS "amp: vendored llama.cpp at ${AMP_VENDOR_DIR}")
message(STATUS "amp: version ${AMP_GIT_SHA}+${AMP_BUILD_DATE}")
message(STATUS "amp: CUDA ${AMP_WITH_CUDA}   binaries -> ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")
