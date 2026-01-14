cmake_minimum_required (VERSION 3.25.2)

# C++ Config
set (CMAKE_CXX_STANDARD 20)

# Cuda config - L4 GPUs use sm_89
set (CMAKE_CUDA_ARCHITECTURES "89")
set (CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")

# Project
project (multigpu_benchmark LANGUAGES CUDA CXX)

# Packages
find_package (FIDESlib REQUIRED CONFIG)
find_package (OpenFHE CONFIG REQUIRED)
find_package (OpenMP REQUIRED)

# NCCL for multi-GPU communication
find_library(NCCL_LIBRARY nccl HINTS /usr/lib/x86_64-linux-gnu /usr/local/lib)
if(NOT NCCL_LIBRARY)
    message(WARNING "NCCL not found, multi-GPU aggregation features will be limited")
endif()

# OpenFHE configuration.
option (BUILD_STATIC "Set to ON to include static versions of the library" OFF)
if (OpenFHE_FOUND)
    message (STATUS "FOUND PACKAGE OpenFHE")
    message (STATUS "OpenFHE Version: ${BASE_OPENFHE_VERSION}")
    message (STATUS "OpenFHE installed as shared libraries: ${OpenFHE_SHARED}")
    message (STATUS "OpenFHE include files location: ${OpenFHE_INCLUDE}")
    message (STATUS "OpenFHE lib files location: ${OpenFHE_LIBDIR}")
    message (STATUS "OpenFHE Native Backend size: ${OpenFHE_NATIVE_SIZE}")
else ()
    message (FATAL_ERROR "PACKAGE OpenFHE NOT FOUND")
endif ()

set (CMAKE_CXX_FLAGS ${OpenFHE_CXX_FLAGS})

include_directories (${OPENMP_INCLUDES})
include_directories (${OpenFHE_INCLUDE})
include_directories (${OpenFHE_INCLUDE}/third-party/include)
include_directories (${OpenFHE_INCLUDE}/core)
include_directories (${OpenFHE_INCLUDE}/pke)
include_directories (${OpenFHE_INCLUDE}/binfhe)
include_directories (${OpenFHE_INCLUDE}/hook)

link_directories (${OpenFHE_LIBDIR})
link_directories (${OPENMP_LIBRARIES})

if (BUILD_STATIC)
    set (CMAKE_EXE_LINKER_FLAGS "${OpenFHE_EXE_LINKER_FLAGS} -static")
    link_libraries (${OpenFHE_STATIC_LIBRARIES})
else ()
    set (CMAKE_EXE_LINKER_FLAGS ${OpenFHE_EXE_LINKER_FLAGS})
    link_libraries (${OpenFHE_SHARED_LIBRARIES})
endif ()

# FIDESlib configuration.
if (FIDESlib_FOUND)
    message (STATUS "FOUND PACKAGE FIDESlib")
    message (STATUS "FIDESlib Version: ${FIDESlib_VERSION}")
    message (STATUS "FIDESlib header files location: ${FIDESLIB_INCLUDE_PATH}")
    message (STATUS "FIDESlib lib files location: ${FIDESLIB_LIBRARIES_PATH}")
    message (STATUS "FIDESlib binary files location: ${FIDESLIB_BINARY_PATH}")
else ()
    message (FATAL_ERROR "PACKAGE FIDESlib NOT FOUND")
endif ()

# Source files
set (SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)

# Multi-GPU MatMul Benchmark
add_executable (multigpu_matmul_bench ${SOURCE_DIR}/multigpu_matmul_bench.cu)
target_link_libraries (multigpu_matmul_bench PUBLIC OpenMP::OpenMP_CXX)
target_link_libraries (multigpu_matmul_bench PRIVATE FIDESlib::fideslib)
if(NCCL_LIBRARY)
    target_link_libraries(multigpu_matmul_bench PRIVATE ${NCCL_LIBRARY})
    target_compile_definitions(multigpu_matmul_bench PRIVATE HAS_NCCL)
endif()

# Multi-GPU Bootstrap Benchmark
add_executable (multigpu_bootstrap_bench ${SOURCE_DIR}/multigpu_bootstrap_bench.cu)
target_link_libraries (multigpu_bootstrap_bench PUBLIC OpenMP::OpenMP_CXX)
target_link_libraries (multigpu_bootstrap_bench PRIVATE FIDESlib::fideslib)
if(NCCL_LIBRARY)
    target_link_libraries(multigpu_bootstrap_bench PRIVATE ${NCCL_LIBRARY})
    target_compile_definitions(multigpu_bootstrap_bench PRIVATE HAS_NCCL)
endif()

# Dual Context Test (validates multi-GPU works)
add_executable (test_dual_context ${SOURCE_DIR}/test_dual_context.cu)
target_link_libraries (test_dual_context PUBLIC OpenMP::OpenMP_CXX)
target_link_libraries (test_dual_context PRIVATE FIDESlib::fideslib)

# Simple Benchmark (pure GPU compute)
add_executable (multigpu_simple_bench ${SOURCE_DIR}/multigpu_simple_bench.cu)
target_link_libraries (multigpu_simple_bench PUBLIC OpenMP::OpenMP_CXX)
target_link_libraries (multigpu_simple_bench PRIVATE FIDESlib::fideslib)
if(NCCL_LIBRARY)
    target_link_libraries(multigpu_simple_bench PRIVATE ${NCCL_LIBRARY})
    target_compile_definitions(multigpu_simple_bench PRIVATE HAS_NCCL)
endif()
