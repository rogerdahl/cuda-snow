cmake_minimum_required(VERSION 3.5)
project(cuda-snow)
set(CMAKE_VERBOSE_MAKEFILE ON)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)

set(SRC_DIR ${CMAKE_SOURCE_DIR}/src)

SET(CUDA_VERBOSE_BUILD ON)
SET(CUDA_SEPARABLE_COMPILATION ON)
find_package(CUDA QUIET REQUIRED)
message(STATUS "Found CUDA version: ${CUDA_VERSION}")
#set(BUILD_SHARED_LIBS OFF)

# SDL
find_package(SDL2 REQUIRED)

set(
  CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};
  -gencode=arch=compute_61,code=sm_61
  -gencode arch=compute_50,code=sm_50
  -gencode arch=compute_35,code=sm_35
  -std=c++11;
)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  message(WARNING "Building DEBUG version")
  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -g -G -O0)
else()
  message(WARNING "Building RELEASE version")
endif()

set(CMAKE_CXX_FLAGS
  "${CMAKE_CXX_FLAGS} -Wall -Werror -Wfatal-errors -Wno-deprecated-declarations"
)

set(CMAKE_CXX_STANDARD 14)

# Specify include directories
include_directories(
  ${SRC_DIR}
  ${SDL2_INCLUDE_DIRS}
)

link_directories(
)

link_libraries(
  ${CUDA_LIBRARIES}
  ${SDL2_LIBRARIES}
  GLEW
  glut
  GL
)

cuda_add_executable(
  snow
  ${SRC_DIR}/cuda_snow.cpp
  ${SRC_DIR}/kernel.cu
  ${SRC_DIR}/cuda_utilities.cpp
)

message(WARNING "CUDA_NVCC_FLAGS: ${CUDA_NVCC_FLAGS}")
message(WARNING "CMAKE_CXX_FLAGS: ${CMAKE_CXX_FLAGS}")
