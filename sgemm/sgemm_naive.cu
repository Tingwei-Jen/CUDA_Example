#include "sgemm.h"
#include <stdio.h>

#define BLOCK_DIM 16

__global__ void sgemm_naive_kernel(const float *A, const float *B, float *C, int M, int N, int K, float alpha, float beta) {

    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col >= N || row >= M) {
        return;
    }

    float sum = 0.f;

    #pragma unroll
    for (int i = 0; i < K; ++i) {
        sum += A[row * K + i] * B[i * N + col];  // two times global memory access and one time multiplication
    }

    C[row * N + col] = alpha * sum + beta * C[row * N + col];
}

void sgemm_naive(const float *A, const float *B, float *C, int M, int N, int K, float alpha, float beta) {

    dim3 dimBlock(BLOCK_DIM, BLOCK_DIM);
    dim3 dimGrid((N + BLOCK_DIM - 1) / BLOCK_DIM, (M + BLOCK_DIM - 1) / BLOCK_DIM);
    sgemm_naive_kernel << < dimGrid, dimBlock >> > (A, B, C, M, N, K, alpha, beta);
}