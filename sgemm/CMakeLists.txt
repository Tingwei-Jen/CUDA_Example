
cmake_minimum_required(VERSION 3.10)

# Project name 
project(sgemm CXX CUDA)

# Specify the C++ standard
enable_language(CXX)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# set CUDA path
set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda-11.8")
set(CMAKE_CUDA_ARCHITECTURES 75 86) 

# Specify the CUDA standard
enable_language(CUDA)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED True)

# set nvcc flags
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} --resource-usage -gencode arch=compute_86,code=sm_86")
#GDB
set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -G -g") 

# Find CUDA package
find_package(CUDA REQUIRED)

find_package(Eigen3 REQUIRED)

# 找到 cuBLAS 庫
find_library(CUBLAS_LIB cublas HINTS ${CUDA_TOOLKIT_ROOT_DIR}/lib64)

add_executable(${PROJECT_NAME} main.cpp 
    sgemm_naive.cu 
    sgemm_shared.cu 
    sgemm_shared_bk.cu 
    sgemm_tile2d.cu
    sgemm_tile2d_padding.cu
    sgemm_tile2d_transpose.cu
    sgemm_tile2d_float4.cu)

target_include_directories(${PROJECT_NAME} PRIVATE 
    ${CUDA_INCLUDE_DIRS} 
    /usr/local/cuda-samples/Common/)

target_link_libraries(${PROJECT_NAME} 
    ${CUDA_LIBRARIES} 
    ${CUBLAS_LIB}
    Eigen3::Eigen)

# 設定其他編譯選項和連結庫
set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
