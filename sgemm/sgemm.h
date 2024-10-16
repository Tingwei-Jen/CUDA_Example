#ifndef SGEMM_H
#define SGEMM_H
#include <cuda_fp16.h>

/**
 * @brief Performs single-precision general matrix multiplication (SGEMM) on the GPU.
 *
 * This function computes the matrix product of two matrices A and B, and adds the result to matrix C,
 * scaled by the factors alpha and beta respectively. The operation performed is:
 * 
 *     C = alpha * (A * B) + beta * C
 *
 * @param A Pointer to the first input matrix (of size M x K).
 * @param B Pointer to the second input matrix (of size K x N).
 * @param C Pointer to the output matrix (of size M x N).
 * @param N Number of rows in matrices A and C.
 * @param M Number of columns in matrices B and C.
 * @param K Number of columns in matrix A and number of rows in matrix B.
 * @param alpha Scalar multiplier for the product of matrices A and B.
 * @param beta Scalar multiplier for the matrix C.
 */
void sgemm_naive(const float *A, const float *B, float *C, int M, int N, int K, float alpha, float beta);
void sgemm_shared(const float *A, const float *B, float *C, int M, int N, int K, float alpha, float beta);
void sgemm_shared_bk(const float *A, const float *B, float *C, int M, int N, int K, float alpha, float beta);
void sgemm_tile2d(const float *A, const float *B, float *C, int M, int N, int K, float alpha, float beta);
void sgemm_tile2d_padding(const float *A, const float *B, float *C, int M, int N, int K, float alpha, float beta);
void sgemm_tile2d_transpose(const float *A, const float *B, float *C, int M, int N, int K, float alpha, float beta);
void sgemm_tile2d_float4(float *A, float *B, float *C, int M, int N, int K, float alpha, float beta);

#endif // SGEMM_H