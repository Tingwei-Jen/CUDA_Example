#include "sgemm.h"
#include <stdio.h>

#define TILE_SIZE 64
#define ELEMS_PER_THREAD_X 1
#define ELEMS_PER_THREAD_Y 4

__global__ void sgemm_tile2d_float4_kernel(float *A, float *B, float *C, int M, int N, int K, float alpha, float beta) {

    // float4 type
    float4 *a4 = reinterpret_cast<float4*>(A);
    float4 *b4 = reinterpret_cast<float4*>(B);
    float4 *c4 = reinterpret_cast<float4*>(C);

    // local thread index
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // left top corner of each tile on C 
    int ltx = blockIdx.x * TILE_SIZE;
    int lty = blockIdx.y * TILE_SIZE;

    // shared memory for tile, float type
    __shared__ float s_tile_A[TILE_SIZE][TILE_SIZE];
    __shared__ float s_tile_B[TILE_SIZE][TILE_SIZE];

    // 初始化 Cvalue 為每個 thread 處理的多個結果, float4 type
    float4 cvalue[ELEMS_PER_THREAD_Y][ELEMS_PER_THREAD_X] = {0.0f};

    // 計算 Cvalue
    #pragma unroll
    for (int bk = 0; bk < K; bk += TILE_SIZE) {
        
        // load A to shared memory
        #pragma unroll
        for (int i = 0; i < ELEMS_PER_THREAD_Y; i++) {    // row
            #pragma unroll
            for (int j = 0; j < ELEMS_PER_THREAD_X; j++) {   // col
                int localRow = ty + i * blockDim.y;
                int localCol = (tx + j * blockDim.x) * 4;
                int globalRow = lty + localRow;
                int globalCol = bk + localCol;
                int globalIdx = globalRow * K / 4 + globalCol / 4;
                float4 a = a4[globalIdx];
                if (globalRow < M && globalCol < K) {
                    s_tile_A[localRow][localCol] = a.x;
                    s_tile_A[localRow][localCol + 1] = a.y;
                    s_tile_A[localRow][localCol + 2] = a.z;
                    s_tile_A[localRow][localCol + 3] = a.w;
                } else {
                    s_tile_A[localRow][localCol] = 0.0f;
                    s_tile_A[localRow][localCol + 1] = 0.0f;
                    s_tile_A[localRow][localCol + 2] = 0.0f;
                    s_tile_A[localRow][localCol + 3] = 0.0f;
                }
            }
        }

        // load B to shared memory
        #pragma unroll
        for (int i = 0; i < ELEMS_PER_THREAD_Y; i++) {    // row
            #pragma unroll
            for (int j = 0; j < ELEMS_PER_THREAD_X; j++) {   // col
                int localRow = ty + i * blockDim.y;
                int localCol = (tx + j * blockDim.x) * 4;
                int globalRow = bk + localRow;
                int globalCol = ltx + localCol;
                int globalIdx = globalRow * N / 4 + globalCol / 4;
                float4 b = b4[globalIdx];
                if (globalRow < K && globalCol < N) {
                    s_tile_B[localRow][localCol] = b.x;
                    s_tile_B[localRow][localCol + 1] = b.y;
                    s_tile_B[localRow][localCol + 2] = b.z;
                    s_tile_B[localRow][localCol + 3] = b.w;
                } else {
                    s_tile_B[localRow][localCol] = 0.0f;
                    s_tile_B[localRow][localCol + 1] = 0.0f;
                    s_tile_B[localRow][localCol + 2] = 0.0f;
                    s_tile_B[localRow][localCol + 3] = 0.0f;
                }

            }
        }

        __syncthreads();

        #pragma unroll
        for (int i = 0; i < ELEMS_PER_THREAD_Y; i++) {    // row
            #pragma unroll
            for (int j = 0; j < ELEMS_PER_THREAD_X; j++) {   // col
                int localRow = ty + i * blockDim.y;
                int localCol = (tx + j * blockDim.x) * 4;
                #pragma unroll
                for (int e = 0; e < TILE_SIZE; ++e) {
                    cvalue[i][j].x += s_tile_A[localRow][e] * s_tile_B[e][localCol];
                    cvalue[i][j].y += s_tile_A[localRow][e] * s_tile_B[e][localCol + 1];
                    cvalue[i][j].z += s_tile_A[localRow][e] * s_tile_B[e][localCol + 2];
                    cvalue[i][j].w += s_tile_A[localRow][e] * s_tile_B[e][localCol + 3];
                }
            }
        }
        __syncthreads();
    }

    // 將結果寫回 C 矩陣
    #pragma unroll
    for (int i = 0; i < ELEMS_PER_THREAD_Y; i++) {  // row
        #pragma unroll 
        for (int j = 0; j < ELEMS_PER_THREAD_X; j++) {  // col
            int globalRow = lty + ty + i * blockDim.y;
            int globalCol = ltx + (tx + j * blockDim.x) * 4;
            int globalIdx = globalRow * N / 4 + globalCol / 4;
            float4 c = c4[globalIdx];
            if (globalRow < M && globalCol < N) {
                c.x = alpha * cvalue[i][j].x + beta * c.x;
                c.y = alpha * cvalue[i][j].y + beta * c.y;
                c.z = alpha * cvalue[i][j].z + beta * c.z;
                c.w = alpha * cvalue[i][j].w + beta * c.w;
                c4[globalIdx] = c;
            }
        }
    }
}

void sgemm_tile2d_float4(float *A, float *B, float *C, int M, int N, int K, float alpha, float beta) {

    int blockDimx = TILE_SIZE / ELEMS_PER_THREAD_X / 4;
    int blockDimy = TILE_SIZE / ELEMS_PER_THREAD_Y;
    dim3 dimBlock(blockDimx, blockDimy);
    dim3 dimGrid((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);
    sgemm_tile2d_float4_kernel << < dimGrid, dimBlock >> > (A, B, C, M, N, K, alpha, beta);
}