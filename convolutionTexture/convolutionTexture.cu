#include <assert.h>
#include <helper_cuda.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;
#include "convolutionSeparable_common.h"

#define MAX_KERNEL_LENGTH 128
#define BLOCKDIM_X 16
#define BLOCKDIM_Y 16
#define RESULT_STEPS 4

////////////////////////////////////////////////////////////////////////////////
// Convolution kernel storage
////////////////////////////////////////////////////////////////////////////////

__constant__ float c_Kernel[MAX_KERNEL_LENGTH];

extern "C" void setConvolutionKernel(float *h_Kernel, int kernelSize) {
    cudaMemcpyToSymbol(c_Kernel, h_Kernel, kernelSize * sizeof(float));
}

////////////////////////////////////////////////////////////////////////////////
// Row convolution filter
////////////////////////////////////////////////////////////////////////////////
__global__ void convolutionRowsKernel(float *d_Dst, int imageW, int imageH, int kernelRadius, cudaTextureObject_t texObj) {

    const int idx_x = blockDim.x * blockIdx.x + threadIdx.x;
    const int idx_y = blockDim.y * blockIdx.y + threadIdx.y;
    
    // 在 CUDA 和 OpenGL 以及大多数图像处理库中，纹理坐标通常被定义为位于像素的左上角。
    // 例如，对于一个整数坐标 (ix, iy)，默认情况下，这个坐标指的是像素左上角的位置，而不是像素的中心。
    const float x = (float)idx_x + 0.5f;
    const float y = (float)idx_y + 0.5f;

    if (x >= imageW || y >= imageH) {
        return;
    }

    float sum = 0;

    #pragma unroll
    for (int j = -kernelRadius; j <= kernelRadius; j++) {
        sum += tex2D<float>(texObj, x + (float)j, y) * c_Kernel[kernelRadius - j];
    }

    d_Dst[idx_y * imageW + idx_x] = sum;
}

extern "C" void convolutionRowsGPU(float *d_Dst, int imageW, int imageH, int kernelSize, cudaTextureObject_t texObj) {
    dim3 dimBlock(BLOCKDIM_X, BLOCKDIM_Y);
    int stride = BLOCKDIM_X * RESULT_STEPS;
    dim3 dimGrid((imageW + BLOCKDIM_X - 1) / BLOCKDIM_X, 
                 (imageH + BLOCKDIM_Y - 1) / BLOCKDIM_Y);
    int kernelRadius = kernelSize / 2;
    convolutionRowsKernel<<<dimGrid, dimBlock>>>(d_Dst, imageW, imageH, kernelRadius, texObj);
    getLastCudaError("convolutionRowsKernel() execution failed\n");
}

////////////////////////////////////////////////////////////////////////////////
// Column convolution filter
////////////////////////////////////////////////////////////////////////////////
__global__ void convolutionColsKernel(float *d_Dst, int imageW, int imageH, int kernelRadius, cudaTextureObject_t texObj) {

    const int idx_x = blockDim.x * blockIdx.x + threadIdx.x;
    const int idx_y = blockDim.y * blockIdx.y + threadIdx.y;
    
    // 在 CUDA 和 OpenGL 以及大多数图像处理库中，纹理坐标通常被定义为位于像素的左上角。
    // 例如，对于一个整数坐标 (ix, iy)，默认情况下，这个坐标指的是像素左上角的位置，而不是像素的中心。
    const float x = (float)idx_x + 0.5f;
    const float y = (float)idx_y + 0.5f;

    if (x >= imageW || y >= imageH) {
        return;
    }

    float sum = 0;

    #pragma unroll
    for (int j = -kernelRadius; j <= kernelRadius; j++) {
        sum += tex2D<float>(texObj, x, y + (float)j) * c_Kernel[kernelRadius - j];
    }

    d_Dst[idx_y * imageW + idx_x] = sum;
}

extern "C" void convolutionColumnsGPU(float *d_Dst, int imageW, int imageH, int kernelSize, cudaTextureObject_t texObj) {
    dim3 dimBlock(BLOCKDIM_X, BLOCKDIM_Y);
    dim3 dimGrid((imageW + BLOCKDIM_X - 1) / BLOCKDIM_X, 
                 (imageH + BLOCKDIM_Y - 1) / BLOCKDIM_Y);
    int kernelRadius = kernelSize / 2;
    convolutionColsKernel<<<dimGrid, dimBlock>>>(d_Dst, imageW, imageH, kernelRadius, texObj);
    getLastCudaError("convolutionColsKernel() execution failed\n");
}
