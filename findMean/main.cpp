#include <iostream>
// Utilities and system includes
#include <helper_functions.h>
#include <helper_cuda.h>
#include "findMean.h"

int main(int argc, char* argv[]) {
    // start logs
    printf("[%s] - Starting...\n", argv[0]);

    // initialize cuda event
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // data size
    int dataSize = 1e6;
    int test_iter = 1000;

    // buffers
    float *h_input = (float *)malloc(dataSize * sizeof(float));
    float *h_meanValue = (float *)malloc(sizeof(float));

    // random init input from 0 to 1
    for (int i = 0; i < dataSize; i++) {
        h_input[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    }

    float *d_input, *d_meanValue;
    cudaMalloc((void **)&d_input, dataSize * sizeof(float));
    cudaMalloc((void **)&d_meanValue, dataSize * sizeof(float));    

    // Copy data from host to device
    cudaMemcpy(d_input, h_input, dataSize * sizeof(float), cudaMemcpyHostToDevice);

    // warm-up
    findMean(d_meanValue, d_input, dataSize);

    // start cuda event
    cudaEventRecord(start);

    for (int i = 0; i < test_iter; i++) {
        findMean(d_meanValue, d_input, dataSize);
    }

    // event record
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // print elapsed time by cuda event
    float elapsed_time_msed_event = 0.f;
    cudaEventElapsedTime(&elapsed_time_msed_event, start, stop);
    elapsed_time_msed_event /= (float)test_iter;
    printf("CUDA event estimated - elapsed %.6f ms \n", elapsed_time_msed_event);

    // Copy data from device to host
    cudaMemcpy(h_meanValue, d_meanValue, sizeof(float), cudaMemcpyDeviceToHost);

    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(e));
        return 1;
    }

    // CPU Test
    float sum = 0;
    for (int i = 0; i < dataSize; i++) {
        sum += h_input[i];
    }
    float mean = sum / dataSize;

    printf("GPU Mean = %f\n", *h_meanValue);
    printf("CPU Mean = %f\n", mean);

    // cleanup
    free(h_input);
    free(h_meanValue);
    cudaFree(d_input);
    cudaFree(d_meanValue);

    // finalize cuda event
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}