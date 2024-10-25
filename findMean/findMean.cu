#include "findMean.h"
#include <stdio.h>

#define BLOCK_SIZE 256 // must be power of 2

// unroll warp reduction
__device__ void warpReduce(volatile float* s_data, int localIdx) {
    s_data[localIdx] += s_data[localIdx + 32]; 
    s_data[localIdx] += s_data[localIdx + 16]; 
    s_data[localIdx] += s_data[localIdx + 8]; 
    s_data[localIdx] += s_data[localIdx + 4]; 
    s_data[localIdx] += s_data[localIdx + 2]; 
    s_data[localIdx] += s_data[localIdx + 1]; 
}

__global__ void reduction_kernel(float *d_output, const float *d_input, const int numElements) {

    unsigned int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int localIdx = threadIdx.x;
    
    extern __shared__ float s_data[];

    // cumulates input with grid-stride loop and save to share memory
    float input = 0.f;
    #pragma unroll
    for (int i = globalIdx; i < numElements; i += blockDim.x * gridDim.x) {
        input += d_input[i];
    }
    s_data[localIdx] = input;

    __syncthreads();

    #pragma unroll
    for (unsigned int stride = blockDim.x/2; stride > 32; stride >>= 1) {
        if (localIdx < stride) {
            s_data[localIdx] += s_data[localIdx + stride];
        }
        __syncthreads();
    }

    if (localIdx < 32) {
        warpReduce(s_data, localIdx);
    }

    if (localIdx == 0) {
        d_output[blockIdx.x] = s_data[0];
    }
}

__global__ void multiply(float *d_meanValue, const float invSize) {
    unsigned int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (globalIdx == 0) {
        d_meanValue[0] *= invSize;
    }
}

void findMean(float *d_meanValue, const float *d_input, const int numElements) {
    // printf("Calculating mean value...\n");
    
    // calculate number of blocks
    int numSMs;
    int numBlocksPerSM;
    cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, 0);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSM, reduction_kernel, BLOCK_SIZE, 0);

    // parallel reduction, calculate sum of input 
    int blockSize = BLOCK_SIZE;
    int gridSize = min(numBlocksPerSM * numSMs, (numElements + BLOCK_SIZE - 1) / BLOCK_SIZE);    
    int sharedMemSize = BLOCK_SIZE * sizeof(float);
    reduction_kernel<<<gridSize, blockSize, sharedMemSize, 0>>>(d_meanValue, d_input, numElements);
    reduction_kernel<<<1, blockSize, sharedMemSize, 0>>>(d_meanValue, d_meanValue, gridSize);
    
    // calculate mean
    float invSize = 1.f / numElements;
    multiply<<<1, BLOCK_SIZE>>>(d_meanValue, invSize);
}