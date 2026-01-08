cmake_minimum_required(VERSION 3.25.2)

# C++ Config
set(CMAKE_CXX_STANDARD 20)

# CUDA config
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "89-real")
endif()
set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")

# Project
project(he_matmul LANGUAGES CUDA CXX)

# Packages
find_package(FIDESlib REQUIRED CONFIG)
find_package(OpenFHE CONFIG REQUIRED)
find_package(OpenMP REQUIRED)

# OpenFHE configuration
option(BUILD_STATIC "Set to ON to include static versions of the library" OFF)
if(OpenFHE_FOUND)
    message(STATUS "FOUND PACKAGE OpenFHE")
    message(STATUS "OpenFHE Version: ${BASE_OPENFHE_VERSION}")
    message(STATUS "OpenFHE installed as shared libraries: ${OpenFHE_SHARED}")
    message(STATUS "OpenFHE include files location: ${OpenFHE_INCLUDE}")
    message(STATUS "OpenFHE lib files location: ${OpenFHE_LIBDIR}")
    message(STATUS "OpenFHE Native Backend size: ${OpenFHE_NATIVE_SIZE}")
else()
    message(FATAL_ERROR "PACKAGE OpenFHE NOT FOUND")
endif()

set(CMAKE_CXX_FLAGS ${OpenFHE_CXX_FLAGS})

include_directories(${OPENMP_INCLUDES})
include_directories(${OpenFHE_INCLUDE})
include_directories(${OpenFHE_INCLUDE}/third-party/include)
include_directories(${OpenFHE_INCLUDE}/core)
include_directories(${OpenFHE_INCLUDE}/pke)
include_directories(${OpenFHE_INCLUDE}/binfhe)
include_directories(${OpenFHE_INCLUDE}/hook)

link_directories(${OpenFHE_LIBDIR})
link_directories(${OPENMP_LIBRARIES})

if(BUILD_STATIC)
    set(CMAKE_EXE_LINKER_FLAGS "${OpenFHE_EXE_LINKER_FLAGS} -static")
    link_libraries(${OpenFHE_STATIC_LIBRARIES})
else()
    link_libraries(${OpenFHE_SHARED_LIBRARIES})
endif()

# Executable
add_executable(he_matmul src/main.cu)

target_link_libraries(he_matmul 
    FIDESlib::fideslib
    OpenMP::OpenMP_CXX
)

set_target_properties(he_matmul PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)
