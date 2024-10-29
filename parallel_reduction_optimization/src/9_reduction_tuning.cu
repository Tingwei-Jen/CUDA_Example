#include "reduction.h"
#include <stdio.h>
// #define BLOCK_SIZE 512 // must be power of 2
// #define TN 4           // number of elements per thread
// #define K 2            // whole data to seperate K parts

/*
    Parallel sum reduction using shared memory
    - takes log(n) steps for n input elements
    - uses n threads
    - only works for power-of-2 arrays
*/

// unroll warp reduction
template<unsigned int blockSize>
__device__ void warpReduce6(volatile float* s_data, int localIdx, int current) {
    int index = current * blockSize + localIdx;
    if (blockSize >= 64) s_data[index] += s_data[index + 32]; 
    if (blockSize >= 32) s_data[index] += s_data[index + 16]; 
    if (blockSize >= 16) s_data[index] += s_data[index + 8]; 
    if (blockSize >= 8) s_data[index] += s_data[index + 4]; 
    if (blockSize >= 4) s_data[index] += s_data[index + 2]; 
    if (blockSize >= 2) s_data[index] += s_data[index + 1]; 
}

template<unsigned int blockSize>
__device__ void blockReduce2(volatile float* s_data, int localIdx, int current) {

    // do reduction in shared mem, interleave addressing
    int index = current * blockSize + localIdx;

    if (blockSize >= 1024) { if (localIdx < 512) { s_data[index] += s_data[index + 512]; } __syncthreads(); }
    if (blockSize >= 512) { if (localIdx < 256) { s_data[index] += s_data[index + 256]; } __syncthreads(); }
    if (blockSize >= 256) { if (localIdx < 128) { s_data[index] += s_data[index + 128]; } __syncthreads(); }
    if (blockSize >= 128) { if (localIdx < 64) { s_data[index] += s_data[index + 64]; } __syncthreads(); }

    // unroll warp reduction
    if (localIdx < 32) warpReduce6<blockSize>(s_data, localIdx, current);
}

template<unsigned int blockSize, int TN, int K>
__global__ void reduction_tuning_kernel(float *d_output, const float *d_input, const int numElements) {
    unsigned int localIdx = threadIdx.x;
    unsigned int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;

    // shared memory double buffering
    __shared__ float s_data[2 * blockSize];

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
    s_data[current * blockSize + localIdx] = inputSum;

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

        // Load the next data_stride into shared memory
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
        s_data[next * blockSize + localIdx] = inputSum;

        // move data to next stride
        d_input += data_stride;

        // perform block reduction
        blockReduce2<blockSize>(s_data, localIdx, current);
        
        // save the result
        result[count] = s_data[current * blockSize];

        __syncthreads();

        // Move to the next data stride (next becomes current)
        current = next;
    }

    // Perform the last block reduction
    blockReduce2<blockSize>(s_data, localIdx, current);
    result[count] = s_data[current * blockSize];

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


void reduction_tuning(float *d_output, const float *d_input, const int numElements) {

    // calculate number of blocks
    int blockSize = 64;
    int TN = 4;
    int numSMs;
    int numBlocksPerSM;
    cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, 0);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSM, reduction_tuning_kernel<64,4,1>, blockSize, 0);
    int tileSize = blockSize * TN;
    int gridSize = min(numBlocksPerSM * numSMs, (numElements + (tileSize) - 1) / (tileSize));

    // launch kernel
    reduction_tuning_kernel<64,4,1><<<gridSize, blockSize>>>(d_output, d_input, numElements);
    reduction_tuning_kernel<64,4,1><<<1, blockSize>>>(d_output, d_output, gridSize);
}


// blockSize = 512, 256, 128, 64, 32
// TN = 8, 4
// K = 4, 2, 1

// blockSize = 512, TN = 8, K = 2   552
// blockSize = 512, TN = 8, K = 1   557
// blockSize = 512, TN = 4, K = 4   545
// blockSize = 512, TN = 4, K = 2   551
// blockSize = 512, TN = 4, K = 1   556

// blockSize = 256, TN = 8, K = 4   548
// blockSize = 256, TN = 8, K = 2   553
// blockSize = 256, TN = 8, K = 1   558.5
// blockSize = 256, TN = 4, K = 4   549
// blockSize = 256, TN = 4, K = 2   553
// blockSize = 256, TN = 4, K = 1   557

// blockSize = 128, TN = 8, K = 4   550
// blockSize = 128, TN = 8, K = 2   555
// blockSize = 128, TN = 8, K = 1   558.6
// blockSize = 128, TN = 4, K = 4   550
// blockSize = 128, TN = 4, K = 2   554
// blockSize = 128, TN = 4, K = 1   556.8

// blockSize = 64, TN = 8, K = 2   555
// blockSize = 64, TN = 8, K = 1   557.8
// blockSize = 64, TN = 4, K = 2
// blockSize = 64, TN = 4, K = 1   555.10