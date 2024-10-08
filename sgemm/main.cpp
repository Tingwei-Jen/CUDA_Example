#include <iostream>
#include <Eigen/Dense>

// CUDA runtime
#include <cuda_runtime.h>
#include <cublas_v2.h>

// Utilities and system includes
#include <helper_functions.h>
#include <helper_cuda.h>

#include "sgemm.h"

void random_init(float *data, int size)
{
	for (int i = 0; i < size; ++i) {
		// random value between -1 and 1
        data[i] = 2.f * (rand() / (float)RAND_MAX) - 1.f;
	}
}

int main(int argc, char* argv[]) {

    // start logs
    printf("[%s] - Starting...\n", argv[0]);

    // initialize timer
    StopWatchInterface *hTimer;
    sdkCreateTimer(&hTimer);

    // initialize cuda event
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int test_iter = 100;
	float *h_A, *h_B, *h_C;
	float *d_A, *d_B, *d_C;
	int N = 2048;
    int M = 2048;
    int K = 2048;
	float alpha = 1.f;
	float beta = 0.f;

	// allocation of host memory
	h_A = (float *)malloc(M * K * sizeof(float));
	h_B = (float *)malloc(K * N * sizeof(float));
	h_C = (float *)malloc(M * N * sizeof(float));

	// allocation of gpu linear memory space
	cudaMalloc((void **)&d_A, M * K * sizeof(float));
	cudaMalloc((void **)&d_B, K * N * sizeof(float));
	cudaMalloc((void **)&d_C, M * N * sizeof(float));

	// initialize randomized values for memory space
	random_init(h_A, M * K);
	random_init(h_B, K * N);
	random_init(h_C, M * N);

    // copy data from host to device
    cudaMemcpy(d_A, h_A, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, K * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C, M * N * sizeof(float), cudaMemcpyHostToDevice);
    
    // warm-up
    sgemm_gpu(d_A, d_B, d_C, M, N, K, alpha, beta);

    // reset timer
    checkCudaErrors(cudaDeviceSynchronize());
    sdkResetTimer(&hTimer);
    sdkStartTimer(&hTimer);

    // start cuda event
    cudaEventRecord(start);

    for (int i = 0; i < test_iter; i++) {
        // Launch kernel
        sgemm_gpu(d_A, d_B, d_C, M, N, K, alpha, beta);
    }

    // event record
    cudaEventRecord(stop);
    checkCudaErrors(cudaEventSynchronize(stop));

    // calculate elapsed time
    checkCudaErrors(cudaDeviceSynchronize());
    sdkStopTimer(&hTimer);

    // print elapsed time by cuda event
    float elapsed_time_msed_event = 0.f;
    cudaEventElapsedTime(&elapsed_time_msed_event, start, stop);
    elapsed_time_msed_event /= (float)test_iter;
    printf("CUDA event estimated - elapsed %.6f ms \n", elapsed_time_msed_event);

    // print elapsed time by sdk timer
    float elapsed_timer_gpu = sdkGetTimerValue(&hTimer) / (float)test_iter;
    printf("Processing Time by timer: %.6f ms\n", elapsed_timer_gpu);

    // copy data from device to host
    cudaMemcpy(h_C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    // cublas comparison  ///////////////////////////////////////////////////////////
    float *h_C_cublas;
    float *d_C_cublas;
    h_C_cublas = (float *)malloc(M * N * sizeof(float));
    cudaMalloc((void **)&d_C_cublas, M * N * sizeof(float));

    // Create a handle for CUBLAS
    cublasHandle_t handle;
    cublasCreate(&handle);

    // Reset the cublas timer
    sdkResetTimer(&hTimer);
    sdkStartTimer(&hTimer);

    // Start the cublas timer
    cudaEventRecord(start);

    // Perform matrix multiplication with CUBLAS
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, &alpha, d_B, M, d_A, K, &beta, d_C_cublas, M);

    // event record
    cudaEventRecord(stop);
    checkCudaErrors(cudaEventSynchronize(stop));

    // calculate elapsed time
    checkCudaErrors(cudaDeviceSynchronize());
    sdkStopTimer(&hTimer);

    // Calculate the elapsed time
    float elapsed_time_msed_event_cublas = 0.f;
    cudaEventElapsedTime(&elapsed_time_msed_event_cublas, start, stop);
    printf("CUBLAS event estimated - elapsed %.6f ms \n", elapsed_time_msed_event_cublas);

    // print elapsed time by sdk timer
    float elapsed_timer_cublas = sdkGetTimerValue(&hTimer);
    printf("Processing Time by CUBLAS timer: %.6f ms\n", elapsed_timer_cublas);

    // copy data from device to host
    cudaMemcpy(h_C_cublas, d_C_cublas, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    // compare the results
    float error = 0.f;

    for (int i = 0; i < M * N; i++) {
        // printf("h_C[%d]: %.6f, h_C_cublas[%d]: %.6f\n", i, h_C[i], i, h_C_cublas[i]);
        error += h_C[i] - h_C_cublas[i];
    }

    printf("Error: %.6f\n", error);

    // // eigen comparison  ///////////////////////////////////////////////////////////
    // Eigen::MatrixXf A(M, K);
    // Eigen::MatrixXf B(K, N);
    // Eigen::MatrixXf C(M, N);

    // for (int i = 0; i < M; i++) {
    //     for (int j = 0; j < K; j++) {
    //         A(i, j) = h_A[i * K + j];
    //     }
    // }

    // for (int i = 0; i < K; i++) {
    //     for (int j = 0; j < N; j++) {
    //         B(i, j) = h_B[i * N + j];
    //     }
    // }

    // C = A * B;

    // float error_eigen = 0.f;

    // for (int i = 0; i < M; i++) {
    //     for (int j = 0; j < N; j++) {
    //         // printf("C(%d, %d): %.6f, h_C[%d]: %.6f\n", i, j, C(i, j), i * N + j, h_C[i * N + j]);
    //         error_eigen += C(i, j) - h_C[i * N + j];
    //     }
    // }

    // printf("Error Eigen: %.6f\n", error_eigen);

    // free memory
    free(h_A);
    free(h_B);
    free(h_C);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // finalize timer
    sdkDeleteTimer(&hTimer);

    // finalize cuda event
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // finalize cublas
    cublasDestroy(handle);

    return 0;
}