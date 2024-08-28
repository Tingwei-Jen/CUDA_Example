
/* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

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
__global__ void convolutionRowsKernel(float *d_Dst, float *d_Src, int imageW, int imageH, int rowsHaloSteps, int kernelRadius) {
    // Handle to thread block group
    cg::thread_block cta = cg::this_thread_block();

    extern __shared__ float s_Data[];
    int idx_x = blockDim.x * (blockIdx.x * RESULT_STEPS) + threadIdx.x; 
    int idx_y = blockDim.y * blockIdx.y + threadIdx.y;

    // Load main data
    #pragma unroll
    for (int i = 0; i < RESULT_STEPS; i++) {
        int localIdx_x = threadIdx.x + (i + rowsHaloSteps) * blockDim.x;
        int localIdx_y = threadIdx.y;
        int localIdx = localIdx_y * (RESULT_STEPS + 2 * rowsHaloSteps) * blockDim.x + localIdx_x;
        
        int dataIdx_x = idx_x + i * blockDim.x;
        int dataIdx_y = idx_y;
        int dataIdx = dataIdx_y * imageW + dataIdx_x;

        s_Data[localIdx] = (dataIdx_x < imageW && dataIdx_y < imageH) ? d_Src[dataIdx] : 0;
    }

    // Load left halo
    #pragma unroll
    for (int i = 0; i < rowsHaloSteps; i++) {

        int localIdx_x = threadIdx.x + i * blockDim.x;
        int localIdx_y = threadIdx.y;
        int localIdx = localIdx_y * (RESULT_STEPS + 2 * rowsHaloSteps) * blockDim.x + localIdx_x;

        int dataIdx_x = idx_x - (rowsHaloSteps-i) * blockDim.x;
        int dataIdx_y = idx_y;
        int dataIdx = dataIdx_y * imageW + dataIdx_x;

        s_Data[localIdx] = (dataIdx_x >= 0 && dataIdx_y < imageH) ? d_Src[dataIdx] : 0;
    }
    
    // Load right halo
    #pragma unroll
    for (int i = 0; i < rowsHaloSteps; i++) {
        int localIdx_x = threadIdx.x + (RESULT_STEPS + rowsHaloSteps + i) * blockDim.x;
        int localIdx_y = threadIdx.y;
        int localIdx = localIdx_y * (RESULT_STEPS + 2 * rowsHaloSteps) * blockDim.x + localIdx_x;
        
        int dataIdx_x = idx_x + (RESULT_STEPS + i) * blockDim.x;
        int dataIdx_y = idx_y;
        int dataIdx = dataIdx_y * imageW + dataIdx_x;

        s_Data[localIdx] = (dataIdx_x < imageW && dataIdx_y < imageH) ? d_Src[dataIdx] : 0;
    }

    // Compute convolution and store results
    cg::sync(cta);

    #pragma unroll
    for (int i = 0; i < RESULT_STEPS; i++) {
        float sum = 0;

        #pragma unroll
        for (int j = -kernelRadius; j <= kernelRadius; j++) {
            int localIdx_x = threadIdx.x + (i + rowsHaloSteps) * blockDim.x + j;
            int localIdx_y = threadIdx.y;
            int localIdx = localIdx_y * (RESULT_STEPS + 2 * rowsHaloSteps) * blockDim.x + localIdx_x;
            sum += c_Kernel[kernelRadius - j] * s_Data[localIdx];
        }

        int dataIdx_x = idx_x + i * blockDim.x;
        int dataIdx_y = idx_y;
        int dataIdx = dataIdx_y * imageW + dataIdx_x;
        if (dataIdx_x < imageW && dataIdx_y < imageH)
            d_Dst[dataIdx] = sum;
    }
}

extern "C" void convolutionRowsGPU(float *d_Dst, float *d_Src, int imageW, int imageH, int kernelSize) {

    dim3 dimBlock(BLOCKDIM_X, BLOCKDIM_Y);
    int stride = BLOCKDIM_X * RESULT_STEPS;
    dim3 dimGrid((imageW + stride - 1) / (stride), 
                 (imageH + BLOCKDIM_Y - 1) / BLOCKDIM_Y);

    int kernelRadius = kernelSize / 2;
    int rowsHaloSteps = kernelRadius / BLOCKDIM_X + 1; 

    int sharedMemSize = BLOCKDIM_Y * (RESULT_STEPS + 2 * rowsHaloSteps) * BLOCKDIM_X * sizeof(float);
    convolutionRowsKernel<<<dimGrid, dimBlock, sharedMemSize, 0>>>(d_Dst, d_Src, imageW, imageH, rowsHaloSteps, kernelRadius);
    getLastCudaError("convolutionRowsKernel() execution failed\n");
}

////////////////////////////////////////////////////////////////////////////////
// Column convolution filter
////////////////////////////////////////////////////////////////////////////////
__global__ void convolutionColumnsKernel(float *d_Dst, float *d_Src, int imageW, int imageH, int colsHaloSteps, int kernelRadius) {
    // Handle to thread block group
    cg::thread_block cta = cg::this_thread_block();

    extern __shared__ float s_Data[];
    int idx_x = blockDim.x * blockIdx.x + threadIdx.x; 
    int idx_y = blockDim.y * (blockIdx.y * RESULT_STEPS) + threadIdx.y;

    // Load main data
    #pragma unroll
    for (int i = 0; i < RESULT_STEPS; i++) {
        int localIdx_x = threadIdx.x;
        int localIdx_y = threadIdx.y + (i + colsHaloSteps) * blockDim.y;
        int localIdx = localIdx_y * blockDim.x + localIdx_x;
        
        int dataIdx_x = idx_x;
        int dataIdx_y = idx_y + i * blockDim.y;
        int dataIdx = dataIdx_y * imageW + dataIdx_x;

        s_Data[localIdx] = (dataIdx_x < imageW && dataIdx_y < imageH) ? d_Src[dataIdx] : 0;
    }

    // Load top halo
    #pragma unroll
    for (int i = 0; i < colsHaloSteps; i++) {

        int localIdx_x = threadIdx.x;
        int localIdx_y = threadIdx.y + i * blockDim.y;
        int localIdx = localIdx_y * blockDim.x + localIdx_x;

        int dataIdx_x = idx_x;
        int dataIdx_y = idx_y - (colsHaloSteps-i) * blockDim.y;
        int dataIdx = dataIdx_y * imageW + dataIdx_x;

        s_Data[localIdx] = (dataIdx_y >= 0 && dataIdx_x < imageW) ? d_Src[dataIdx] : 0;
    }

    // Load buttom halo
    #pragma unroll
    for (int i = 0; i < colsHaloSteps; i++) {
        int localIdx_x = threadIdx.x;
        int localIdx_y = threadIdx.y + (RESULT_STEPS + colsHaloSteps + i) * blockDim.y;
        int localIdx = localIdx_y * blockDim.x + localIdx_x;

        int dataIdx_x = idx_x;
        int dataIdx_y = idx_y + (RESULT_STEPS + i) * blockDim.y;
        int dataIdx = dataIdx_y * imageW + dataIdx_x;
        
        s_Data[localIdx] = (dataIdx_x < imageW && dataIdx_y < imageH) ? d_Src[dataIdx] : 0;
    }

    // Compute convolution and store results
    cg::sync(cta);

    #pragma unroll
    for (int i = 0; i < RESULT_STEPS; i++) {
        float sum = 0;

        #pragma unroll
        for (int j = -kernelRadius; j <= kernelRadius; j++) {
            int localIdx_x = threadIdx.x;
            int localIdx_y = threadIdx.y + (i + colsHaloSteps) * blockDim.y + j;
            int localIdx = localIdx_y * blockDim.x + localIdx_x;
            sum += c_Kernel[kernelRadius - j] * s_Data[localIdx];
        }

        int dataIdx_x = idx_x;
        int dataIdx_y = idx_y + i * blockDim.y;
        int dataIdx = dataIdx_y * imageW + dataIdx_x;
        if (dataIdx_x < imageW && dataIdx_y < imageH) {
            d_Dst[dataIdx] = sum;
        }
    }
}

extern "C" void convolutionColumnsGPU(float *d_Dst, float *d_Src, int imageW, int imageH, int kernelSize) {

    dim3 dimBlock(BLOCKDIM_X, BLOCKDIM_Y);
    int stride = BLOCKDIM_Y * RESULT_STEPS;
    dim3 dimGrid((imageW + BLOCKDIM_X - 1) / (BLOCKDIM_X), 
                 (imageH + stride - 1) / (stride));

    int kernelRadius = kernelSize / 2;
    int colsHaloSteps = kernelRadius / BLOCKDIM_Y + 1; 

    int sharedMemSize = (RESULT_STEPS + 2 * colsHaloSteps) * BLOCKDIM_Y * BLOCKDIM_X * sizeof(float);
    convolutionColumnsKernel<<<dimGrid, dimBlock, sharedMemSize, 0>>>(d_Dst, d_Src, imageW, imageH, colsHaloSteps, kernelRadius);
    getLastCudaError("convolutionColumnsGPU() execution failed\n");
}
