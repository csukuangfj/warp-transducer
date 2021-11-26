if("x${CMAKE_SOURCE_DIR}" STREQUAL "x${CMAKE_BINARY_DIR}")
  message(FATAL_ERROR "\
In-source build is not a good practice.
Please use:
  mkdir build
  cd build
  cmake ..
to build this project"
  )
endif()

cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

# see https://cmake.org/cmake/help/latest/policy/CMP0111.html
#
# This is to suppress the warnings for importing PyTorch.
if(POLICY CMP0111)
  cmake_policy(SET CMP0111 OLD)
endif()
if(POLICY CMP0074)
  cmake_policy(SET CMP0074 NEW)
endif()

set(languages CXX)
set(_RNNT_WITH_CUDA ON)

find_program(RNNT_HAS_NVCC nvcc)
if(NOT RNNT_HAS_NVCC AND "$ENV{CUDACXX}" STREQUAL "")
  message(STATUS "No NVCC detected. Disable CUDA support")
  set(_RNNT_WITH_CUDA OFF)
endif()

if(APPLE OR (DEFINED RNNT_WITH_CUDA AND NOT RNNT_WITH_CUDA))
  if(_RNNT_WITH_CUDA)
    message(STATUS "Disable CUDA support")
    set(_RNNT_WITH_CUDA OFF)
  endif()
endif()

if(_RNNT_WITH_CUDA)
  set(languages ${languages} CUDA)
endif()

message(STATUS "Enabled languages: ${languages}")

project(warp_rnnt ${languages})

set(RNNT_VERSION "1.0")


if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(RNNT_WITH_CUDA "Whether to build warp-rnnt with CUDA" ${_RNNT_WITH_CUDA})
option(RNNT_USE_NAIVE_KERNEL "use naive alpha-beta kernel" OFF)
option(RNNT_DEBUG_TIME "output kernel time" OFF)
option(RNNT_DEBUG_KERNEL "output alpha beta" OFF)
option(RNNT_WITH_OMP "compile warp-rnnt with openmp." OFF)
option(RNNT_BUILD_TESTS "True to also build tests." OFF)

if(RNNT_USE_NAIVE_KERNEL)
  add_definitions(-DUSE_NAIVE_KERNEL)
endif()

if(RNNT_DEBUG_TIME)
  add_definitions(-DDEBUG_TIME)
endif()

if(RNNT_DEBUG_KERNEL)
  add_definitions(-DDEBUG_KERNEL)
endif()


if(NOT RNNT_WITH_OMP)
  add_definitions(-DRNNT_DISABLE_OMP)
endif()

if(RNNT_WITH_CUDA)
  add_definitions(-DRNNT_WITH_CUDA)
endif()

if(RNNT_WITH_OMP)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
  if(RNNT_WITH_CUDA)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -fopenmp")
  endif()
endif()

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

set(CMAKE_SKIP_BUILD_RPATH FALSE)
set(BUILD_RPATH_USE_ORIGIN TRUE)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
set(CMAKE_INSTALL_RPATH "$ORIGIN")
set(CMAKE_BUILD_RPATH "$ORIGIN")

set(CMAKE_CXX_STANDARD 14 CACHE STRING "The C++ version to be used.")
set(CMAKE_CXX_EXTENSIONS OFF)
message(STATUS "C++ Standard version: ${CMAKE_CXX_STANDARD}")

if(RNNT_WITH_CUDA)
  unset(CMAKE_CUDA_STANDARD CACHE)
  set(CMAKE_CUDA_STANDARD ${CMAKE_CXX_STANDARD})

  include(cmake/select_compute_arch.cmake)
  cuda_select_nvcc_arch_flags(RNNT_COMPUTE_ARCH_FLAGS)
  message(STATUS "RNNT_COMPUTE_ARCH_FLAGS: ${RNNT_COMPUTE_ARCH_FLAGS}")

  # You can use -DRNNT_COMPUTE_ARCHS="37;50;70" when invoking CMake
  if(NOT RNNT_COMPUTE_ARCHS)
    # see https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/
    # https://www.myzhar.com/blog/tutorials/tutorial-nvidia-gpu-cuda-compute-capability/
    set(RNNT_COMPUTE_ARCH_CANDIDATES 35 50 60 61 70 75)
    if(CUDA_VERSION VERSION_GREATER "11.0")
      list(APPEND RNNT_COMPUTE_ARCH_CANDIDATES 80 86)
    endif()
    message(STATUS "RNNT_COMPUTE_ARCH_CANDIDATES ${RNNT_COMPUTE_ARCH_CANDIDATES}")

    set(RNNT_COMPUTE_ARCHS)

    foreach(COMPUTE_ARCH IN LISTS RNNT_COMPUTE_ARCH_CANDIDATES)
      if("${RNNT_COMPUTE_ARCH_FLAGS}" MATCHES ${COMPUTE_ARCH})
        message(STATUS "Adding arch ${COMPUTE_ARCH}")
        list(APPEND RNNT_COMPUTE_ARCHS ${COMPUTE_ARCH})
      else()
        message(STATUS "Skipping arch ${COMPUTE_ARCH}")
      endif()
    endforeach()

    if(NOT RNNT_COMPUTE_ARCHS)
      set(RNNT_COMPUTE_ARCHS ${RNNT_COMPUTE_ARCH_CANDIDATES})
    endif()
  endif()

  message(STATUS "RNNT_COMPUTE_ARCHS: ${RNNT_COMPUTE_ARCHS}")

  foreach(COMPUTE_ARCH IN LISTS RNNT_COMPUTE_ARCHS)
    string(APPEND CMAKE_CUDA_FLAGS " --expt-extended-lambda -gencode arch=compute_${COMPUTE_ARCH},code=sm_${COMPUTE_ARCH}")
    set(CMAKE_CUDA_ARCHITECTURES "${COMPUTE_ARCH}-real;${COMPUTE_ARCH}-virtual;${CMAKE_CUDA_ARCHITECTURES}")
  endforeach()
endif()

list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake/Modules)
list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)

include(pybind11)
include(torch)

set(srcs ./src/rnnt_entrypoint.cu)
if(NOT RNNT_WITH_CUDA)
  set(srcs ./src/rnnt_entrypoint.cpp)
endif()

include_directories(include)
add_library(warprnnt_core SHARED ${srcs})
if(RNNT_WITH_OMP)
  target_link_libraries(warprnnt_core -fopenmp)
endif()

if(RNNT_BUILD_TESTS)
  add_executable(test_cpu tests/test_cpu.cpp tests/random.cpp)
  target_link_libraries(test_cpu warprnnt_core)

  add_executable(test_time tests/test_time.cpp tests/random.cpp )
  target_link_libraries(test_time warprnnt_core)

  if(RNNT_WITH_CUDA)
    add_executable(test_gpu tests/test_gpu.cu tests/random.cu)
    target_link_libraries(test_gpu warprnnt_core)

    add_executable(test_time_gpu tests/test_time.cu tests/random.cu)
    target_link_libraries(test_time_gpu warprnnt_core)
  endif()
endif()


set(warp_rnnt_py_srcs python/warp_rnnt.cu)
if(NOT RNNT_WITH_CUDA)
  set(warp_rnnt_py_srcs python/warp_rnnt.cc)
endif()

pybind11_add_module(_warp_rnnt_py ${warp_rnnt_py_srcs} SHARED)
target_link_libraries(_warp_rnnt_py PRIVATE warprnnt_core)
target_link_libraries(_warp_rnnt_py PRIVATE ${TORCH_LIBRARIES})
target_link_libraries(_warp_rnnt_py PRIVATE ${TORCH_DIR}/lib/libtorch_python.so)

if(RNNT_WITH_CUDA)
  message(STATUS "CMAKE_CUDA_FLAGS: ${CMAKE_CUDA_FLAGS}")
endif()
