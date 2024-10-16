#include "sgemm.h"
#include <stdio.h>
#define BLOCK_DIM 16
#define ELEMS_PER_THREAD 4  // 每个线程处理的元素个数

__global__ void sgemm_tile2d_kernel(const float *A, const float *B, float *C, int M, int N, int K, float alpha, float beta) {

    // local thread index
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // 計算當前 thread 處理的 C 矩陣塊的位置, left top corner
    int tileSize = BLOCK_DIM * ELEMS_PER_THREAD;
    int ltx = blockIdx.x * tileSize;
    int lty = blockIdx.y * tileSize;

    // 使用共享內存來存儲 A 和 B 的子矩陣（tile）
    __shared__ float s_tile_A[BLOCK_DIM * ELEMS_PER_THREAD][BLOCK_DIM * ELEMS_PER_THREAD];
    __shared__ float s_tile_B[BLOCK_DIM * ELEMS_PER_THREAD][BLOCK_DIM * ELEMS_PER_THREAD];

    // 初始化 Cvalue 為每個 thread 處理的多個結果
    float cvalue[ELEMS_PER_THREAD][ELEMS_PER_THREAD] = {0.0f};

    // 計算 Cvalue
    #pragma unroll
    for (int bk = 0; bk < K; bk += tileSize) {

        // load A to shared memory
        #pragma unroll
        for (int i = 0; i < ELEMS_PER_THREAD; i++) {    // row
            #pragma unroll
            for (int j = 0; j < ELEMS_PER_THREAD; j++) {   // col
                int localRow = ty + i * BLOCK_DIM;
                int localCol = tx + j * BLOCK_DIM;
                int globalRow = lty + localRow;
                int globalCol = bk + localCol;
                if (globalRow < M && globalCol < K) {
                    s_tile_A[localRow][localCol] = A[globalRow * K + globalCol];
                } else {
                    s_tile_A[localRow][localCol] = 0.0f;
                }
            }
        }

        // load B to shared memory
        #pragma unroll
        for (int i = 0; i < ELEMS_PER_THREAD; i++) {    // row
            #pragma unroll
            for (int j = 0; j < ELEMS_PER_THREAD; j++) {   // col
                int localRow = ty + i * BLOCK_DIM;
                int localCol = tx + j * BLOCK_DIM;
                int globalRow = bk + localRow;
                int globalCol = ltx + localCol;
                if (globalRow < K && globalCol < N) {
                    s_tile_B[localRow][localCol] = B[globalRow * N + globalCol];
                } else {
                    s_tile_B[localRow][localCol] = 0.0f;
                }
            }
        }

        __syncthreads();

        #pragma unroll
        for (int e = 0; e < tileSize; ++e) {
            #pragma unroll
            for (int i = 0; i < ELEMS_PER_THREAD; i++) {    // row
                #pragma unroll
                for (int j = 0; j < ELEMS_PER_THREAD; j++) {   // col
                    int localRow = ty + i * BLOCK_DIM;
                    int localCol = tx + j * BLOCK_DIM;
                    cvalue[i][j] += s_tile_A[localRow][e] * s_tile_B[e][localCol];
                }
            }
        }
        __syncthreads();
    }

    // 將結果寫回 C 矩陣
    #pragma unroll
    for (int i = 0; i < ELEMS_PER_THREAD; i++) {  // row
        #pragma unroll 
        for (int j = 0; j < ELEMS_PER_THREAD; j++) {  // col
            int row = lty + ty + i * BLOCK_DIM;
            int col = ltx + tx + j * BLOCK_DIM;
            if (row < M && col < N) {
                C[row * N + col] = alpha * cvalue[i][j] + beta * C[row * N + col];
            }
        }
    }
}


void sgemm_tile2d(const float *A, const float *B, float *C, int M, int N, int K, float alpha, float beta) {

    dim3 dimBlock(BLOCK_DIM, BLOCK_DIM);
    int tileSize = BLOCK_DIM * ELEMS_PER_THREAD;
    dim3 dimGrid((N + tileSize - 1) / tileSize, (M + tileSize - 1) / tileSize);
    sgemm_tile2d_kernel << < dimGrid, dimBlock >> > (A, B, C, M, N, K, alpha, beta);
}
