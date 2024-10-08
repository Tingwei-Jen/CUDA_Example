#include "sgemm.h"
#include <stdio.h>

#define BLOCK_DIM 24

__global__ void sgemm_gpu_kernel(const float *A, const float *B, float *C, int M, int N, int K, float alpha, float beta) {

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

__global__ void sgemm_gpu_kernel_shared(const float *A, const float *B, float *C, int M, int N, int K, float alpha, float beta) {

    int localIdx_x = threadIdx.x;
    int localIdx_y = threadIdx.y;
    int globalIdx_x = blockIdx.x * blockDim.x + localIdx_x;
    int globalIdx_y = blockIdx.y * blockDim.y + localIdx_y;

    if (globalIdx_x >= N || globalIdx_y >= M) {
        return;
    }

    __shared__ float s_tile_A[BLOCK_DIM][BLOCK_DIM];
    __shared__ float s_tile_B[BLOCK_DIM][BLOCK_DIM];

    float sum = 0.f;

    #pragma unroll
    for (int i = 0; i < K; i += BLOCK_DIM) {

        s_tile_A[localIdx_y][localIdx_x] = A[globalIdx_y * K + i + localIdx_x];  // load A to shared memory
        s_tile_B[localIdx_y][localIdx_x] = B[(i + localIdx_y) * N + globalIdx_x];  // load B to shared memory

        __syncthreads();

        #pragma unroll
        for (int e = 0; e < BLOCK_DIM; ++e) {
            sum += s_tile_A[localIdx_y][e] * s_tile_B[e][localIdx_x];  // two times shared memory access and one time multiplication
        }

	    __syncthreads();
    }

    C[globalIdx_y * N + globalIdx_x] = alpha * sum + beta * C[globalIdx_y * N + globalIdx_x];
}

#define TILE_SIZE 32  // tile 的大小
#define ELEMS_PER_THREAD_X 4  // 每个线程处理的元素个数
#define ELEMS_PER_THREAD_Y 4  // 每个线程处理的元素个数

__global__ void sgemm_gpu_kernel_shared_2d_tile(const float *A, const float *B, float *C, int M, int N, int K, float alpha, float beta) {

    // local thread index
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // 計算當前 thread 處理的 C 矩陣塊的位置
    int row = blockIdx.y * TILE_SIZE + ty * ELEMS_PER_THREAD_Y;
    int col = blockIdx.x * TILE_SIZE + tx * ELEMS_PER_THREAD_X;

    // 使用共享內存來存儲 A 和 B 的子矩陣（tile）
    __shared__ float sA[TILE_SIZE][TILE_SIZE];
    __shared__ float sB[TILE_SIZE][TILE_SIZE];

    // 初始化 Cvalue 為每個 thread 處理的多個結果
    float cvalue[ELEMS_PER_THREAD_Y][ELEMS_PER_THREAD_X] = {0.0f};

    // 計算 Cvalue
    for (int k = 0; k < K; k += TILE_SIZE) {

        // 加載 A 的子矩陣到共享內存
        for (int i = 0; i < ELEMS_PER_THREAD_Y; i++) {
            for (int j = 0; j < ELEMS_PER_THREAD_X; j++) {
                // check bounrary
                if (row + i < M && k + tx * ELEMS_PER_THREAD_X + j < K) {
                    sA[ty * ELEMS_PER_THREAD_Y + i][tx * ELEMS_PER_THREAD_X + j] = A[(row + i) * K + k + tx * ELEMS_PER_THREAD_X + j];
                } else {
                    sA[ty * ELEMS_PER_THREAD_Y + i][tx * ELEMS_PER_THREAD_X + j] = 0.0f;
                }
            }
        }

        // 加載 B 的子矩陣到共享內存
        for (int i = 0; i < ELEMS_PER_THREAD_Y; i++) {
            for (int j = 0; j < ELEMS_PER_THREAD_X; j++) {
                if (k + ty * ELEMS_PER_THREAD_Y + i < K && col + j < N) {
                    sB[ty * ELEMS_PER_THREAD_Y + i][tx * ELEMS_PER_THREAD_X + j] = B[(k + ty * ELEMS_PER_THREAD_Y + i) * N + col + j];
                } else {
                    sB[ty * ELEMS_PER_THREAD_Y + i][tx * ELEMS_PER_THREAD_X + j] = 0.0f;
                }
            }
        }

        __syncthreads();

        for (int e = 0; e < TILE_SIZE; ++e) {
            for (int i = 0; i < ELEMS_PER_THREAD_Y; i++) {
                for (int j = 0; j < ELEMS_PER_THREAD_X; j++) {
                    cvalue[i][j] += sA[ty * ELEMS_PER_THREAD_Y + i][e] * sB[e][tx * ELEMS_PER_THREAD_X + j];
                }
            }
        }

        __syncthreads();

    }

    // 將結果寫回 C 矩陣
    for (int i = 0; i < ELEMS_PER_THREAD_Y; i++) {
        for (int j = 0; j < ELEMS_PER_THREAD_X; j++) {
            if (row + i < M && col + j < N) {
                C[(row + i) * N + col + j] = alpha * cvalue[i][j] + beta * C[(row + i) * N + col + j];
            }
        }
    }
}

void sgemm_gpu(const float *A, const float *B, float *C, int M, int N, int K, float alpha, float beta) {

    dim3 dimBlock(TILE_SIZE / ELEMS_PER_THREAD_X, TILE_SIZE / ELEMS_PER_THREAD_Y);
    dim3 dimGrid((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);
    sgemm_gpu_kernel_shared_2d_tile << < dimGrid, dimBlock >> > (A, B, C, M, N, K, alpha, beta);
}