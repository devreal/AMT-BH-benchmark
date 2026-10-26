cmake_minimum_required(VERSION 3.19)
project(pepc-ttg CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if (NOT CMAKE_BUILD_TYPE)
  # Default to Release: this project exists partly to measure TTG/PaRSEC
  # per-task overhead, which an unoptimized build would badly distort.
  # Override with -DCMAKE_BUILD_TYPE=Debug if you actually want that.
  set(CMAKE_BUILD_TYPE Release)
endif()

# FetchContent's dependency sources/build dirs live under the build tree by
# default (${CMAKE_BINARY_DIR}/_deps). If you already have a TTG build
# elsewhere with the same deps cloned, point FETCHCONTENT_BASE_DIR at its
# _deps directory (e.g. -DFETCHCONTENT_BASE_DIR=/path/to/ttg/build/_deps)
# so FetchContent reconfigures+rebuilds them instead of re-cloning from
# GitHub - but don't hardcode that path here, since it won't exist on
# other systems.

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake/modules")
include(FindOrFetchTTG)

# add_ttg_executable() is normally made available via ttg-config.cmake
# (install-time consumption). TTG is pulled in here as a live subproject
# instead (via add_subdirectory, under the hood, in FetchContent), so its
# own `list(APPEND CMAKE_MODULE_PATH ...)` only affected ITS OWN directory
# scope, not ours - add its modules dir here too, using the ttg_SOURCE_DIR
# FetchContent_MakeAvailable(ttg) set in our scope.
list(APPEND CMAKE_MODULE_PATH "${ttg_SOURCE_DIR}/cmake/modules")
include(AddTTGExecutable)

# Compile our own targets (not the fetched TTG/PaRSEC/MADNESS, added above)
# for the build machine's ISA: the force kernel's SIMD width and rsqrt path
# (src/simd.h) are picked from the target ISA macros, so a plain x86-64
# build would fall back to SSE2. Turn off for portable binaries.
option(BH_NATIVE_ARCH "Compile with -march=native" ON)
if (BH_NATIVE_ARCH)
  include(CheckCXXCompilerFlag)
  check_cxx_compiler_flag("-march=native" BH_HAVE_MARCH_NATIVE)
  if (BH_HAVE_MARCH_NATIVE)
    add_compile_options(-march=native)
  endif()
endif()

enable_testing()

add_subdirectory(src)
add_subdirectory(tests)
