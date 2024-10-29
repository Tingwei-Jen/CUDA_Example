#include "reduction.h"
#include <stdio.h>
#define BLOCK_SIZE 256 // must be power of 2

/*
    Parallel sum reduction using shared memory
    - takes log(n) steps for n input elements
    - uses n threads
    - only works for power-of-2 arrays
*/
// unroll warp reduction
__device__ void warpReduce2(volatile float* s_data, int localIdx) {
    s_data[localIdx] += s_data[localIdx + 32]; 
    s_data[localIdx] += s_data[localIdx + 16]; 
    s_data[localIdx] += s_data[localIdx + 8]; 
    s_data[localIdx] += s_data[localIdx + 4]; 
    s_data[localIdx] += s_data[localIdx + 2]; 
    s_data[localIdx] += s_data[localIdx + 1]; 
}

__global__ void reduction_unroll_loop_kernel(float *d_output, const float *d_input, const int numElements) {
    unsigned int localIdx = threadIdx.x;
    unsigned int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;

    // shared memory
    extern __shared__ float s_data[];

    // cumulates input with grid-stride loop and save to share memory
    float input = 0.f;
    #pragma unroll
    for (int i = globalIdx; i < numElements; i += blockDim.x * gridDim.x)
        input += d_input[i];
    s_data[localIdx] = input;

    __syncthreads();

    // do reduction in shared mem, interleave addressing
    #pragma unroll
    for (unsigned int stride = blockDim.x/2; stride > 32; stride >>= 1) {
        if (localIdx < stride) {
            s_data[localIdx] += s_data[localIdx + stride];
        }
        __syncthreads();
    }

    if (localIdx < 32) {
        warpReduce2(s_data, localIdx);
    }

    // output 
    if (localIdx == 0) {
        d_output[blockIdx.x] = s_data[0];
    }
}

void reduction_unroll_loop(float *d_output, const float *d_input, const int numElements) {

    // calculate number of blocks
    int numSMs;
    int numBlocksPerSM;
    cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, 0);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSM, reduction_unroll_loop_kernel, BLOCK_SIZE, 0);

    // parallel reduction, calculate sum of input 
    int blockSize = BLOCK_SIZE;
    int gridSize = min(numBlocksPerSM * numSMs, (numElements + (BLOCK_SIZE) - 1) / (BLOCK_SIZE));

    int sharedMemSize = BLOCK_SIZE * sizeof(float);
    reduction_unroll_loop_kernel<<<gridSize, blockSize, sharedMemSize, 0>>>(d_output, d_input, numElements);
    reduction_unroll_loop_kernel<<<1, blockSize, sharedMemSize, 0>>>(d_output, d_output, gridSize);
}
