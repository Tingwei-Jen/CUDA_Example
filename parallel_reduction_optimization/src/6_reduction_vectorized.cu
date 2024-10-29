#include "reduction.h"
#include <stdio.h>
#define BLOCK_SIZE 256 // must be power of 2
#define TN 8          // number of elements per thread

/*
    Parallel sum reduction using shared memory
    - takes log(n) steps for n input elements
    - uses n threads
    - only works for power-of-2 arrays
*/
// unroll warp reduction
__device__ void warpReduce3(volatile float* s_data, int localIdx) {
    s_data[localIdx] += s_data[localIdx + 32]; 
    s_data[localIdx] += s_data[localIdx + 16]; 
    s_data[localIdx] += s_data[localIdx + 8]; 
    s_data[localIdx] += s_data[localIdx + 4]; 
    s_data[localIdx] += s_data[localIdx + 2]; 
    s_data[localIdx] += s_data[localIdx + 1]; 
}

__global__ void reduction_vectorized_kernel(float *d_output, const float *d_input, const int numElements) {
    unsigned int localIdx = threadIdx.x;
    unsigned int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;

    // shared memory
    __shared__ float s_data[BLOCK_SIZE];
    int size = (numElements + TN - 1) / TN;

    float4 input = {0.f, 0.f, 0.f, 0.f};
    #pragma unroll
    for (int i = globalIdx; i < size; i += blockDim.x * gridDim.x) {
        #pragma unroll
        for (int j = 0; j < TN; j+=4) {
            float4 temp = reinterpret_cast<const float4*>(&d_input[i * TN + j])[0];
            input.x += temp.x;
            input.y += temp.y;
            input.z += temp.z;
            input.w += temp.w;
        }
    }
    float inputSum = input.x + input.y + input.z + input.w;
    s_data[localIdx] = inputSum;

    __syncthreads();


    // do reduction in shared mem, interleave addressing
    #pragma unroll
    for (unsigned int stride = blockDim.x/2; stride > 32; stride >>= 1) {
        if (localIdx < stride) {
            s_data[localIdx] += s_data[localIdx + stride];
        }
        __syncthreads();
    }

    // unroll warp reduction
    if (localIdx < 32) {
        warpReduce3(s_data, localIdx);
    }

    // output 
    if (localIdx == 0) {
        d_output[blockIdx.x] = s_data[0];
    }
}

void reduction_vectorized(float *d_output, const float *d_input, const int numElements) {

    // calculate number of blocks
    int numSMs;
    int numBlocksPerSM;
    cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, 0);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSM, reduction_vectorized_kernel, BLOCK_SIZE, 0);
    int tileSize = BLOCK_SIZE * TN;
    int gridSize = min(numBlocksPerSM * numSMs, (numElements + (tileSize) - 1) / (tileSize));

    // launch kernel
    reduction_vectorized_kernel<<<gridSize, BLOCK_SIZE>>>(d_output, d_input, numElements);
    reduction_vectorized_kernel<<<1, BLOCK_SIZE>>>(d_output, d_output, gridSize);
}
