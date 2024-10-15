#include "sgemm.h"
#include <stdio.h>

#define BLOCK_DIM 16
#define BLOCK_DIM_BK 8
#define ELEMS_PER_THREAD 4  // 每个线程处理的元素个数

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

void sgemm_gpu(const float *A, const float *B, float *C, int M, int N, int K, float alpha, float beta) {

    dim3 dimBlock(BLOCK_DIM, BLOCK_DIM);
    dim3 dimGrid((N + BLOCK_DIM - 1) / BLOCK_DIM, (M + BLOCK_DIM - 1) / BLOCK_DIM);
    sgemm_gpu_kernel << < dimGrid, dimBlock >> > (A, B, C, M, N, K, alpha, beta);
}

__global__ void sgemm_gpu_shared_kernel(const float *A, const float *B, float *C, int M, int N, int K, float alpha, float beta) {

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

void sgemm_gpu_shared(const float *A, const float *B, float *C, int M, int N, int K, float alpha, float beta) {

    dim3 dimBlock(BLOCK_DIM, BLOCK_DIM);
    dim3 dimGrid((N + BLOCK_DIM - 1) / BLOCK_DIM, (M + BLOCK_DIM - 1) / BLOCK_DIM);

    sgemm_gpu_shared_kernel << < dimGrid, dimBlock >> > (A, B, C, M, N, K, alpha, beta);
}

__global__ void sgemm_gpu_shared2_kernel(const float *A, const float *B, float *C, int M, int N, int K, float alpha, float beta) {

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

    __shared__ float s_tile_A[BLOCK_DIM][BLOCK_DIM_BK];
    __shared__ float s_tile_B[BLOCK_DIM_BK][BLOCK_DIM];

    float sum = 0.f;
    int BK = BLOCK_DIM_BK;

    #pragma unroll
    for (int bk = 0; bk < K; bk += BK) {

        if (tx < BK) {
            s_tile_A[ty][tx] = A[row * K + bk + tx];    // load A to shared memory
        }

        if (ty < BK) {
            s_tile_B[ty][tx] = B[(bk + ty) * N + col];  // load B to shared memory
        } 

        __syncthreads();

        #pragma unroll
        for (int e = 0; e < BLOCK_DIM_BK; ++e) {
            sum += s_tile_A[ty][e] * s_tile_B[e][tx];
        }

	    __syncthreads();
    }

    C[row * N + col] = alpha * sum + beta * C[row * N + col];
}

void sgemm_gpu_shared2(const float *A, const float *B, float *C, int M, int N, int K, float alpha, float beta) {

    dim3 dimBlock(BLOCK_DIM, BLOCK_DIM);
    dim3 dimGrid((N + BLOCK_DIM - 1) / BLOCK_DIM, (M + BLOCK_DIM - 1) / BLOCK_DIM);
    sgemm_gpu_shared2_kernel << < dimGrid, dimBlock >> > (A, B, C, M, N, K, alpha, beta);
}

__global__ void sgemm_gpu_tile2d_kernel(const float *A, const float *B, float *C, int M, int N, int K, float alpha, float beta) {

    // local thread index
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // 計算當前 thread 處理的 C 矩陣塊的位置, left top corner
    int tileSize = BLOCK_DIM * ELEMS_PER_THREAD;
    int ltx = blockIdx.x * tileSize;
    int lty = blockIdx.y * tileSize;

    // 使用共享內存來存儲 A 和 B 的子矩陣（tile）
    __shared__ float s_tile_A[BLOCK_DIM * ELEMS_PER_THREAD][BLOCK_DIM * ELEMS_PER_THREAD+1];
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


void sgemm_gpu_tile2d(const float *A, const float *B, float *C, int M, int N, int K, float alpha, float beta) {

    dim3 dimBlock(BLOCK_DIM, BLOCK_DIM);
    int tileSize = BLOCK_DIM * ELEMS_PER_THREAD;
    dim3 dimGrid((N + tileSize - 1) / tileSize, (M + tileSize - 1) / tileSize);
    sgemm_gpu_tile2d_kernel << < dimGrid, dimBlock >> > (A, B, C, M, N, K, alpha, beta);
}

__global__ void sgemm_gpu_tile2d_solveBC_kernel(const float *A, const float *B, float *C, int M, int N, int K, float alpha, float beta) {

    // local thread index
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // 計算當前 thread 處理的 C 矩陣塊的位置, left top corner
    int tileSize = BLOCK_DIM * ELEMS_PER_THREAD;
    int ltx = blockIdx.x * tileSize;
    int lty = blockIdx.y * tileSize;

    // 使用共享內存來存儲 A 和 B 的子矩陣（tile）
    __shared__ float s_tile_A_trans[BLOCK_DIM * ELEMS_PER_THREAD][BLOCK_DIM * ELEMS_PER_THREAD];
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
                    s_tile_A_trans[localCol][localRow] = A[globalRow * K + globalCol];
                } else {
                    s_tile_A_trans[localCol][localRow] = 0.0f;
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
                    cvalue[i][j] += s_tile_A_trans[e][localRow] * s_tile_B[e][localCol];
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

void sgemm_gpu_tile2d_solveBC(const float *A, const float *B, float *C, int M, int N, int K, float alpha, float beta) {

    dim3 dimBlock(BLOCK_DIM, BLOCK_DIM);
    int tileSize = BLOCK_DIM * ELEMS_PER_THREAD;
    dim3 dimGrid((N + tileSize - 1) / tileSize, (M + tileSize - 1) / tileSize);
    sgemm_gpu_tile2d_solveBC_kernel << < dimGrid, dimBlock >> > (A, B, C, M, N, K, alpha, beta);
}

#define TILE_SIZE 64
#define ELEMS_PER_THREAD_X 1
#define ELEMS_PER_THREAD_Y 4

__global__ void sgemm_gpu_tile2d_float4_kernel(float *A, float *B, float *C, int M, int N, int K, float alpha, float beta) {

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

void sgemm_gpu_tile2d_float4(float *A, float *B, float *C, int M, int N, int K, float alpha, float beta) {

    int blockDimx = TILE_SIZE / ELEMS_PER_THREAD_X / 4;
    int blockDimy = TILE_SIZE / ELEMS_PER_THREAD_Y;
    dim3 dimBlock(blockDimx, blockDimy);
    dim3 dimGrid((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);
    sgemm_gpu_tile2d_float4_kernel << < dimGrid, dimBlock >> > (A, B, C, M, N, K, alpha, beta);
}