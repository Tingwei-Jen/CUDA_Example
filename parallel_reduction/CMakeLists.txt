
cmake_minimum_required(VERSION 3.10)

# Project name 
project(reduction CXX CUDA)

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

# add executable
file(GLOB CUDA_KERNEL_SOURCES "src/*.cu")
add_executable(${PROJECT_NAME} main.cpp ${CUDA_KERNEL_SOURCES})

target_include_directories(${PROJECT_NAME} PRIVATE 
    ${CUDA_INCLUDE_DIRS} 
    /usr/local/cuda-samples/Common/
    include)

target_link_libraries(${PROJECT_NAME} 
    ${CUDA_LIBRARIES})

# 設定其他編譯選項和連結庫
set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)