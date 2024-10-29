#include "reduction.h"
#define BLOCK_SIZE 256 // must be power of 2

/*
    Parallel sum reduction using shared memory
    - takes log(n) steps for n input elements
    - uses n threads
    - only works for power-of-2 arrays
*/

__global__ void reduction_grid_stride_loop_kernel(float *d_output, const float *d_input, const int numElements) {
    unsigned int localIdx = threadIdx.x;
    unsigned int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;

    // shared memory
    extern __shared__ float s_data[];

    // cumulates input with grid-stride loop and save to share memory
    float input = 0.f;
    for (int i = globalIdx; i < numElements; i += blockDim.x * gridDim.x)
        input += d_input[i];
    s_data[localIdx] = input;

    __syncthreads();

    // do reduction in shared mem, interleave addressing
    for (unsigned int stride = blockDim.x/2; stride > 0; stride >>= 1) {
        if (localIdx < stride) {
            s_data[localIdx] += s_data[localIdx + stride];
        }
        __syncthreads();
    }

    // output 
    if (localIdx == 0) {
        d_output[blockIdx.x] = s_data[0];
    }
}

void reduction_grid_stride_loop(float *d_output, const float *d_input, const int numElements) {

    // calculate number of blocks
    int numSMs;
    int numBlocksPerSM;
    cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, 0);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSM, reduction_grid_stride_loop_kernel, BLOCK_SIZE, 0);

    // parallel reduction, calculate sum of input 
    int blockSize = BLOCK_SIZE;
    int gridSize = min(numBlocksPerSM * numSMs, (numElements + (BLOCK_SIZE) - 1) / (BLOCK_SIZE));
    int sharedMemSize = BLOCK_SIZE * sizeof(float);
    reduction_grid_stride_loop_kernel<<<gridSize, blockSize, sharedMemSize, 0>>>(d_output, d_input, numElements);
    reduction_grid_stride_loop_kernel<<<1, blockSize, sharedMemSize, 0>>>(d_output, d_output, gridSize);
}
