// CUDA runtime
#include <cuda_runtime.h>

// Utilities and system includes
#include <helper_functions.h>
#include <helper_cuda.h>

#include "convolutionSeparable_common.h"

void generate_data(float *h_buffer, int imageH, int imageW)
{
    for (int row = 0; row < imageH; row++) {
        for (int col = 0; col < imageW; col++) {

            h_buffer[row * imageW + col] = static_cast<float>(rand()) / RAND_MAX * 10.0f;
            // h_buffer[row * imageW + col] = 1.f;
        }
    }
}

/* Generates Bi-symetric Gaussian Filter */
void generate_1D_kernel(float *h_kernel, int kernelSize, float sigma)
{
    float sumKernel = 0.f; //for normalization
    for (int i = -kernelSize / 2; i <= kernelSize / 2; i++)
    {
        float kernelValue = expf(-(float)(i * i) / (2.f * sigma * sigma));
        h_kernel[i + kernelSize / 2] = kernelValue;
        sumKernel += kernelValue;
    }

    // normalization
    float normalizationFactor = 1.f / sumKernel;
    for (int i = -kernelSize / 2; i <= kernelSize / 2; i++)
        h_kernel[i + kernelSize / 2] *= normalizationFactor;
}

int main(int argc, char **argv) {
    // start logs
    printf("[%s] - Starting...\n", argv[0]);

    // buffers
    float *h_input, *h_outputGPU;
    float *d_input, *d_buffer, *d_output;  // d_buffer is used for intermediate result after row convolution
    float *h_kernel;

    // image size and kernel size
    const int imageW = 3072;
    const int imageH = 3072;
    const int kernelSize = 35;
    int bufferSize = imageW * imageH * sizeof(float);

    // initialize timer
    StopWatchInterface *hTimer;
    sdkCreateTimer(&hTimer);

    // initialize cuda event
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // allocate host memories
    h_input = (float *)malloc(bufferSize);
    h_outputGPU = (float *)malloc(bufferSize);
    h_kernel = (float *)malloc(kernelSize * sizeof(float));

    // allocate gpu memories
    checkCudaErrors(cudaMalloc((void **)&d_input, bufferSize));
    checkCudaErrors(cudaMalloc((void **)&d_output, bufferSize));
    checkCudaErrors(cudaMalloc((void **)&d_buffer, bufferSize));

    // generate data and filter
    generate_data(h_input, imageH, imageW);
    generate_1D_kernel(h_kernel, kernelSize, 1.f);

    // set kernel into global memory of GPU
    setConvolutionKernel(h_kernel, kernelSize);

    // gpu memory copy
    checkCudaErrors(cudaMemcpy(d_input, h_input, bufferSize, cudaMemcpyHostToDevice));

    // reset timer
    checkCudaErrors(cudaDeviceSynchronize());
    sdkResetTimer(&hTimer);
    sdkStartTimer(&hTimer);
    cudaEventRecord(start);

    // convolution
    convolutionRowsGPU(d_buffer, d_input, imageW, imageH, kernelSize);
    convolutionColumnsGPU(d_output, d_buffer, imageW, imageH, kernelSize);

    // event record
    cudaEventRecord(stop);
    checkCudaErrors(cudaEventSynchronize(stop));

    // calculate elapsed time
    checkCudaErrors(cudaDeviceSynchronize());
    sdkStopTimer(&hTimer);

    // print elapsed time by cuda event
    float elapsed_time_msed = 0.f;
    cudaEventElapsedTime(&elapsed_time_msed, start, stop);
    printf("CUDA event estimated - elapsed %.3f ms \n", elapsed_time_msed);

    // print elapsed time by sdk timer
    float elapsed_timer_gpu = sdkGetTimerValue(&hTimer);
    printf("Processing Time: %.6f ms\n", elapsed_timer_gpu);

    // back to host
    printf("\nReading back GPU results...\n\n");
    checkCudaErrors(cudaMemcpy(h_outputGPU, d_output, bufferSize, cudaMemcpyDeviceToHost));

    printf("Checking the results...\n");

    float *h_buffer, *h_outputCPU;
    h_buffer = (float *)malloc(bufferSize);
    h_outputCPU = (float *)malloc(bufferSize);

    printf(" ...running convolutionRowCPU()\n");
    convolutionRowCPU(h_buffer, h_input, h_kernel, imageW, imageH, kernelSize);
    printf(" ...running convolutionColumnCPU()\n");
    convolutionColumnCPU(h_outputCPU, h_buffer, h_kernel, imageW, imageH, kernelSize);

    printf(" ...comparing the results\n");
    double sum = 0, delta = 0;

    for (unsigned i = 0; i < imageW * imageH; i++) {
        delta +=
            (h_outputGPU[i] - h_outputCPU[i]) * (h_outputGPU[i] - h_outputCPU[i]);
        sum += h_outputCPU[i] * h_outputCPU[i];
    }

    double L2norm = sqrt(delta / sum);
    printf(" ...Relative L2 norm: %E\n\n", L2norm);
    printf("Shutting down...\n");

    checkCudaErrors(cudaFree(d_input));
    checkCudaErrors(cudaFree(d_buffer));
    checkCudaErrors(cudaFree(d_output));
    free(h_input);
    free(h_outputGPU);
    free(h_kernel);
    free(h_buffer);
    free(h_outputCPU);
    sdkDeleteTimer(&hTimer);

    return 0;
}