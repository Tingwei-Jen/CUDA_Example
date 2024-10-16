#include "sgemm.h"
#include <stdio.h>
#define BLOCK_DIM 16

__global__ void sgemm_shared_kernel(const float *A, const float *B, float *C, int M, int N, int K, float alpha, float beta) {

    // local thread index
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    // global index
    int col = blockIdx.x * blockDim.x + tx;
    int row = blockIdx.y * blockDim.y + ty;

    // global index out of range
    if (col >= N || row >= M) {
        return;
    }

    __shared__ float s_tile_A[BLOCK_DIM][BLOCK_DIM];
    __shared__ float s_tile_B[BLOCK_DIM][BLOCK_DIM];

    float sum = 0.f;

    #pragma unroll
    for (int bk = 0; bk < K; bk += BLOCK_DIM) {

        s_tile_A[ty][tx] = A[row * K + bk + tx];    // load A to shared memory
        s_tile_B[ty][tx] = B[(bk + ty) * N + col];  // load B to shared memory

        __syncthreads();

        #pragma unroll
        for (int e = 0; e < BLOCK_DIM; ++e) {
            float a = s_tile_A[ty][e];
            float b = s_tile_B[e][tx];
            sum += a * b;
        }

	    __syncthreads();
    }

    C[row * N + col] = alpha * sum + beta * C[row * N + col];
}

void sgemm_shared(const float *A, const float *B, float *C, int M, int N, int K, float alpha, float beta) {

    dim3 dimBlock(BLOCK_DIM, BLOCK_DIM);
    dim3 dimGrid((N + BLOCK_DIM - 1) / BLOCK_DIM, (M + BLOCK_DIM - 1) / BLOCK_DIM);

    sgemm_shared_kernel << < dimGrid, dimBlock >> > (A, B, C, M, N, K, alpha, beta);
}