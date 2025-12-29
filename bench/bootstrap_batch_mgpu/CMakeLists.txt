# Multi-GPU Batch Bootstrap Benchmark CMake Configuration

cmake_minimum_required(VERSION 3.18)
project(bootstrap_batch_mgpu LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Set CUDA architectures for VALAR GPUs
set(CMAKE_CUDA_ARCHITECTURES "70;75;80;86;89")

# Find dependencies
find_package(CUDAToolkit REQUIRED)
find_package(OpenMP QUIET)
find_package(OpenFHE CONFIG REQUIRED)

# Get parent FIDESlib from two directories up
get_filename_component(FIDESLIB_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
set(FIDESLIB_INCLUDE_DIR "${FIDESLIB_ROOT}/include")

# Check if FIDESlib was already built
if(TARGET fideslib)
    message(STATUS "Using fideslib from parent project")
else()
    # Try to find the library
    find_library(FIDESLIB_LIBRARY
        NAMES fideslib libfideslib
        PATHS "${FIDESLIB_ROOT}/build"
        NO_DEFAULT_PATH
    )
    if(FIDESLIB_LIBRARY)
        message(STATUS "Found fideslib: ${FIDESLIB_LIBRARY}")
    else()
        message(FATAL_ERROR "fideslib not found. Please build FIDESlib first.")
    endif()
endif()

# Create executable
add_executable(bootstrap_batch_mgpu bootstrap_batch_mgpu.cu)

# Include directories
target_include_directories(bootstrap_batch_mgpu PRIVATE
    ${FIDESLIB_INCLUDE_DIR}
    ${CUDAToolkit_INCLUDE_DIRS}
    ${OpenFHE_INCLUDE}
    ${OpenFHE_INCLUDE}/third-party/include
    ${OpenFHE_INCLUDE}/core
    ${OpenFHE_INCLUDE}/pke
    ${OpenFHE_INCLUDE}/binfhe
)

# Link libraries
if(TARGET fideslib)
    target_link_libraries(bootstrap_batch_mgpu PRIVATE fideslib)
else()
    target_link_libraries(bootstrap_batch_mgpu PRIVATE ${FIDESLIB_LIBRARY})
endif()

target_link_libraries(bootstrap_batch_mgpu PRIVATE
    CUDA::cudart
    ${OpenFHE_LIBRARIES}
)

# OpenMP support for multi-threading
if(OpenMP_CXX_FOUND)
    target_link_libraries(bootstrap_batch_mgpu PRIVATE OpenMP::OpenMP_CXX)
    target_compile_definitions(bootstrap_batch_mgpu PRIVATE _OPENMP)
endif()

# Compile options
target_compile_options(bootstrap_batch_mgpu PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-lineinfo>
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:DEBUG>>:-G -g>
)

set_target_properties(bootstrap_batch_mgpu PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    POSITION_INDEPENDENT_CODE ON
)

# Installation
install(TARGETS bootstrap_batch_mgpu
    RUNTIME DESTINATION bin/bench
)

message(STATUS "")
message(STATUS "===== Bootstrap Batch MGPU Configuration =====")
message(STATUS "CUDA Toolkit: ${CUDAToolkit_VERSION}")
message(STATUS "FIDESlib Include: ${FIDESLIB_INCLUDE_DIR}")
message(STATUS "OpenFHE Include: ${OpenFHE_INCLUDE}")
message(STATUS "OpenMP Support: ${OpenMP_CXX_FOUND}")
message(STATUS "==============================================")
