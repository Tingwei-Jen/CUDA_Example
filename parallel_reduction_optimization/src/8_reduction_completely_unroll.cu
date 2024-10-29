#include "reduction.h"
#include <stdio.h>
#define BLOCK_SIZE 256 // must be power of 2
#define TN 8           // number of elements per thread
#define K 1            // whole data to seperate K parts

/*
    Parallel sum reduction using shared memory
    - takes log(n) steps for n input elements
    - uses n threads
    - only works for power-of-2 arrays
*/
// unroll warp reduction
__device__ void warpReduce5(volatile float* s_data, int localIdx, int current) {
    int index = current * BLOCK_SIZE + localIdx;
    if (BLOCK_SIZE >= 64) s_data[index] += s_data[index + 32]; 
    if (BLOCK_SIZE >= 32) s_data[index] += s_data[index + 16]; 
    if (BLOCK_SIZE >= 16) s_data[index] += s_data[index + 8]; 
    if (BLOCK_SIZE >= 8) s_data[index] += s_data[index + 4]; 
    if (BLOCK_SIZE >= 4) s_data[index] += s_data[index + 2]; 
    if (BLOCK_SIZE >= 2) s_data[index] += s_data[index + 1]; 
}

__device__ void blockReduce1(volatile float* s_data, int localIdx, int current) {

    // do reduction in shared mem, interleave addressing
    int index = current * BLOCK_SIZE + localIdx;

    if (BLOCK_SIZE >= 1024) { if (localIdx < 512) { s_data[index] += s_data[index + 512]; } __syncthreads(); }
    if (BLOCK_SIZE >= 512) { if (localIdx < 256) { s_data[index] += s_data[index + 256]; } __syncthreads(); }
    if (BLOCK_SIZE >= 256) { if (localIdx < 128) { s_data[index] += s_data[index + 128]; } __syncthreads(); }
    if (BLOCK_SIZE >= 128) { if (localIdx < 64) { s_data[index] += s_data[index + 64]; } __syncthreads(); }

    // unroll warp reduction
    if (localIdx < 32) warpReduce5(s_data, localIdx, current);
}

__global__ void reduction_completely_unroll_kernel(float *d_output, const float *d_input, const int numElements) {
    unsigned int localIdx = threadIdx.x;
    unsigned int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;

    // shared memory double buffering
    __shared__ float s_data[2 * BLOCK_SIZE];

    // Determine which shared memory bank to use for double buffering
    int current = 0;

    // seperate data to K parts
    int data_stride = numElements / K;
    int size = (data_stride + TN - 1) / TN;

    // Load the first data_stride into shared memory
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
    s_data[current * BLOCK_SIZE + localIdx] = inputSum;

    __syncthreads();

    // move data to next stride
    d_input += data_stride;

    // save the result of each data_stride
    float result[K] = {0.0f};
    int count = 0;

    #pragma unroll
    for (int bk = data_stride; bk < numElements; bk += data_stride, count++) {
        // Switch to the next shared memory bank for prefetching
        int next = 1 - current;

        // reset input
        input = {0.f, 0.f, 0.f, 0.f};

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
        s_data[next * BLOCK_SIZE + localIdx] = inputSum;

        // move data to next stride
        d_input += data_stride;

        // perform block reduction
        blockReduce1(s_data, localIdx, current);
        
        // save the result
        result[count] = s_data[current * BLOCK_SIZE];

        __syncthreads();

        // Move to the next data stride (next becomes current)
        current = next;
    }

    // Perform the last block reduction
    blockReduce1(s_data, localIdx, current);
    result[count] = s_data[current * BLOCK_SIZE];

    // output 
    if (localIdx == 0) {
        int sum = 0;
        #pragma unroll
        for (int i = 0; i < K; i++) {
            sum += result[i];
        }
        d_output[blockIdx.x] = sum;
    }
}

void reduction_completely_unroll(float *d_output, const float *d_input, const int numElements) {

    // calculate number of blocks
    int numSMs;
    int numBlocksPerSM;
    cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, 0);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSM, reduction_completely_unroll_kernel, BLOCK_SIZE, 0);
    int tileSize = BLOCK_SIZE * TN;
    int gridSize = min(numBlocksPerSM * numSMs, (numElements + (tileSize) - 1) / (tileSize));

    // launch kernel
    reduction_completely_unroll_kernel<<<gridSize, BLOCK_SIZE>>>(d_output, d_input, numElements);
    // reduction_completely_unroll_kernel<<<1, BLOCK_SIZE>>>(d_output, d_output, gridSize);
}
