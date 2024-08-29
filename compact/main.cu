// CUDA runtime
#include <cuda_runtime.h>

// Utilities and system includes
#include <helper_functions.h>
#include <helper_cuda.h>

__global__ void compactKernel(int* input, int* output, int* counter, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        if (input[idx] != 0) {  // 篩選出非零元素
            int pos = atomicAdd(counter, 1);  // 使用atomicAdd獲取下一個可用位置
            output[pos] = input[idx];  // 將非零元素壓縮到output數組
        }
    }
}

int main(int argc, char **argv) {
    // start logs
    printf("[%s] - Starting...\n", argv[0]);

    int n = 1024;
    int *h_input = (int *)malloc(n * sizeof(int));
    int *h_output = (int *)malloc(n * sizeof(int));
    int *h_counter = (int *)malloc(sizeof(int));

    int *d_input, *d_output, *d_counter;
    checkCudaErrors(cudaMalloc((void **)&d_input, n * sizeof(int)));
    checkCudaErrors(cudaMalloc((void **)&d_output, n * sizeof(int)));
    checkCudaErrors(cudaMalloc((void **)&d_counter, sizeof(int)));

    for (int i = 0; i < n; i++) {
        h_input[i] = (rand() % 2) * (i + 1);  // 隨機生成0或非零元素
    }

    checkCudaErrors(cudaMemcpy(d_input, h_input, n * sizeof(int), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemset(d_counter, 0, sizeof(int)));

    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;
    compactKernel<<<gridSize, blockSize>>>(d_input, d_output, d_counter, n);

    checkCudaErrors(cudaMemcpy(h_output, d_output, n * sizeof(int), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(h_counter, d_counter, sizeof(int), cudaMemcpyDeviceToHost));
    
    printf("Counter: %d\n", *h_counter);
    // printf("Output array after compact:\n");
    // for (int i = 0; i < *h_counter; i++) {
    //     printf("%d ", h_output[i]);
    // }

    printf("\n");

    // cleanup
    free(h_input);
    free(h_output);
    free(h_counter);
    checkCudaErrors(cudaFree(d_input));
    checkCudaErrors(cudaFree(d_output));
    checkCudaErrors(cudaFree(d_counter));
    
    return 0;
}