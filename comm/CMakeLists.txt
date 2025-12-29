# Communication Microbenchmarks CMake Configuration
# Build NCCL and P2P tests for multi-GPU verification

cmake_minimum_required(VERSION 3.18)
project(FIDESlib_comm_tests LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Set CUDA architectures for VALAR GPUs
# 70 = V100, 75 = T4, 80 = A100, 86 = A40/A6000, 89 = L40S
set(CMAKE_CUDA_ARCHITECTURES "70;75;80;86;89")

# Find CUDA
find_package(CUDAToolkit REQUIRED)

# Find NCCL
# NCCL is typically in $NCCL_ROOT when module is loaded
if(DEFINED ENV{NCCL_ROOT})
    set(NCCL_ROOT $ENV{NCCL_ROOT})
    message(STATUS "NCCL_ROOT from environment: ${NCCL_ROOT}")
elseif(DEFINED ENV{NCCL_HOME})
    set(NCCL_ROOT $ENV{NCCL_HOME})
    message(STATUS "NCCL_HOME from environment: ${NCCL_ROOT}")
else()
    message(WARNING "NCCL_ROOT not set. NCCL tests may not build.")
endif()

# NCCL include and library paths
if(NCCL_ROOT)
    set(NCCL_INCLUDE_DIRS ${NCCL_ROOT}/include)
    set(NCCL_LIBRARIES ${NCCL_ROOT}/lib/libnccl.so)
    
    if(NOT EXISTS ${NCCL_INCLUDE_DIRS}/nccl.h)
        message(WARNING "nccl.h not found at ${NCCL_INCLUDE_DIRS}/nccl.h")
    endif()
    if(NOT EXISTS ${NCCL_LIBRARIES})
        message(WARNING "libnccl.so not found at ${NCCL_LIBRARIES}")
    endif()
endif()

# ======== NCCL All-Reduce Test ========
add_executable(nccl_allreduce nccl_allreduce.cu)

target_include_directories(nccl_allreduce PRIVATE 
    ${CUDAToolkit_INCLUDE_DIRS}
    ${NCCL_INCLUDE_DIRS}
)

target_link_libraries(nccl_allreduce PRIVATE 
    CUDA::cudart
    ${NCCL_LIBRARIES}
)

target_compile_options(nccl_allreduce PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
)

set_target_properties(nccl_allreduce PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# ======== P2P Bandwidth Test ========
add_executable(p2p_bw p2p_bw.cu)

target_include_directories(p2p_bw PRIVATE 
    ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(p2p_bw PRIVATE 
    CUDA::cudart
)

target_compile_options(p2p_bw PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
)

set_target_properties(p2p_bw PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# ======== Installation ========
install(TARGETS nccl_allreduce p2p_bw
    RUNTIME DESTINATION bin/comm
)

message(STATUS "")
message(STATUS "========== Comm Tests Configuration ==========")
message(STATUS "CUDA Toolkit: ${CUDAToolkit_VERSION}")
message(STATUS "CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "NCCL Root: ${NCCL_ROOT}")
message(STATUS "==============================================")
