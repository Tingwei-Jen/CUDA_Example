
cmake_minimum_required(VERSION 3.10)

# Project name 
project(convolutionSeprable CXX CUDA)

# 设置 CUDA 相关变量
set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda-11.8")
set(CMAKE_CUDA_ARCHITECTURES 75) 

# 設定 CUDA 編譯器標誌
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} --resource-usage -gencode arch=compute_86,code=sm_86" )
# list(APPEND CUDA_NVCC_FLAGS "-rdc=true")
# list(APPEND CUDA_NVCC_FLAGS "-Xcompiler -fopenmp")
# set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -G -g")

# Find CUDA package
find_package(CUDA REQUIRED)

add_executable(${PROJECT_NAME} main.cpp convolutionSeparable_cpu.cpp convolutionSeparable.cu)

target_include_directories(${PROJECT_NAME} PRIVATE ${CUDA_INCLUDE_DIRS} /usr/local/cuda-samples/Common/)

target_link_libraries(${PROJECT_NAME} ${CUDA_LIBRARIES})

# 設定其他編譯選項和連結庫
set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
