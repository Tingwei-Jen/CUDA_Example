#include <cuda_runtime.h>

#include <helper_functions.h>
#include <helper_cuda.h>

#include "convolutionSeparable_common.h"

// create texture object
void createTextureObject(cudaArray* cuArray, cudaTextureObject_t& texObj) {

    // specify resource descriptor
    cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = cuArray;

    // specify texture descriptor
    cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
	texDesc.addressMode[0] = cudaAddressModeBorder; // 超出边界时返回一个固定的边界值（通常为 0）。
	texDesc.addressMode[1] = cudaAddressModeBorder; // 若改成cudaAddressModeClamp: 即超出边界的坐标将会被钳制到最近的有效像素位置。
    texDesc.filterMode = cudaFilterModeLinear; // 线性插值
    texDesc.readMode = cudaReadModeElementType; // 直接读取数据类型值
    texDesc.normalizedCoords = 0; // 设置为 0 表示不使用标准化坐标（坐标以像素为单位），设置为 1 表示使用标准化坐标（坐标范围为 [0, 1]）。

    // create texture object
    checkCudaErrors(cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL));
}

void generate_data(float *h_buffer, int numElements) {

    for (int i = 0; i < numElements; i++) {
        h_buffer[i] = static_cast<float>(rand()) / RAND_MAX * 10.0f;
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
    float *h_input, *h_outputGPU, *h_kernel;
    float *d_input, *d_output;

    // image size and kernel size
    const int imageW = 1920;
    const int imageH = 1080;
    const int kernelSize = 7;
    int bufferSize = imageW * imageH * sizeof(float);
    int numElements = imageW * imageH;

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

    // allocate cuda array
    cudaArray *cuArray;
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);  // 等價於 cudaCreateChannelDesc<float>();
    checkCudaErrors(cudaMallocArray(&cuArray, &channelDesc, imageW, imageH));

    // create texture object
    cudaTextureObject_t texObj;
    createTextureObject(cuArray, texObj);

    // generate data and filter
    generate_data(h_input, numElements);
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

    // update intput data in cuda array
    checkCudaErrors(cudaMemcpyToArray(cuArray, 0, 0, d_input, bufferSize, cudaMemcpyDeviceToDevice));

    // convolution
    convolutionRowsGPU(d_output, imageW, imageH, kernelSize, texObj);

    // update row convolution result data in cuda array
    checkCudaErrors(cudaMemcpyToArray(cuArray, 0, 0, d_output, bufferSize, cudaMemcpyDeviceToDevice));
    
    // convolution
    convolutionColumnsGPU(d_output, imageW, imageH, kernelSize, texObj);

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

    // check the results by CPU
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
        delta += (h_outputGPU[i] - h_outputCPU[i]) * (h_outputGPU[i] - h_outputCPU[i]);
        sum += h_outputCPU[i] * h_outputCPU[i];
    }

    double L2norm = sqrt(delta / sum);
    printf(" ...Relative L2 norm: %E\n\n", L2norm);
    printf("Shutting down...\n");

    // free memories
    checkCudaErrors(cudaDestroyTextureObject(texObj));
    checkCudaErrors(cudaFreeArray(cuArray));
    checkCudaErrors(cudaFree(d_output));
    free(h_input);
    free(h_outputGPU);
    free(h_kernel);
    free(h_buffer);
    free(h_outputCPU);

    return 0;
}