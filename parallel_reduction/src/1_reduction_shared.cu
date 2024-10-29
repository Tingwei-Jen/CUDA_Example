#include "reduction.h"
#define BLOCK_SIZE 256 // must be power of 2

/*
    Parallel sum reduction using shared memory
    - takes log(n) steps for n input elements
    - uses n threads
    - only works for power-of-2 arrays
*/

__global__ void reduction_shared_kernel(float *d_output, const float *d_input, const int numElements) {
    unsigned int localIdx = threadIdx.x;
    unsigned int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;

    // shared memory
    extern __shared__ float s_data[];
    s_data[localIdx] = (globalIdx < numElements) ? d_input[globalIdx] : 0.f;

    __syncthreads();

    // do reduction in shared mem, interleave addressing
    for (unsigned int stride = 1; stride < blockDim.x; stride *= 2) {
        
        if (localIdx % (2 * stride) == 0)
            s_data[localIdx] += s_data[localIdx + stride];
        
        __syncthreads();
    }

    // output 
    if (localIdx == 0) {
        d_output[blockIdx.x] = s_data[0];
    }
}

void reduction_shared(float *d_output, const float *d_input, const int numElements) {

    int size = numElements;
    cudaMemcpy(d_output, d_input, size * sizeof(float), cudaMemcpyDeviceToDevice);
    while(size > 1) {
        int n_blocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
        reduction_shared_kernel<<<n_blocks, BLOCK_SIZE, BLOCK_SIZE * sizeof(float)>>>(d_output, d_output, size);
        size = n_blocks;
    }
}
