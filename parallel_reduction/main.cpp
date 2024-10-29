#include <iostream>
#include <cuda_runtime.h>

// Utilities and system includes
#include <helper_functions.h>
#include <helper_cuda.h>
#include <cuda_profiler_api.h>
#include "reduction.h"

class ReductionProfiler {
public:
    ReductionProfiler(const int dataSize, const int testIter);
    ~ReductionProfiler();

    typedef void (*reduction_function)(float *d_output, const float *d_input, const int numElements);
    void profiling(reduction_function reduction_impl);

private:
    cudaEvent_t m_start, m_stop;
    int m_dataSize;
    int m_testIter;
    float *m_hInput, *m_hOutput;
    float *m_dInput, *m_dOutput;
};

ReductionProfiler::ReductionProfiler(const int dataSize, const int testIter) 
    : m_dataSize(dataSize), m_testIter(testIter) {
    // initialize cuda event
    checkCudaErrors(cudaEventCreate(&m_start));
    checkCudaErrors(cudaEventCreate(&m_stop));

    m_hInput = (float *)malloc(m_dataSize * sizeof(float));
    m_hOutput = (float *)malloc(sizeof(float));
    checkCudaErrors(cudaMalloc((void **)&m_dInput, m_dataSize * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&m_dOutput, m_dataSize * sizeof(float)));

    // random init input from 0 to 1
    for (int i = 0; i < m_dataSize; i++) {
        // m_hInput[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
        m_hInput[i] = 1.0f;
    }

    // Copy data from host to device
    cudaMemcpy(m_dInput, m_hInput, m_dataSize * sizeof(float), cudaMemcpyHostToDevice);
}

ReductionProfiler::~ReductionProfiler() {
    free(m_hInput);
    free(m_hOutput);
    checkCudaErrors(cudaFree(m_dInput));
    checkCudaErrors(cudaFree(m_dOutput));
    checkCudaErrors(cudaEventDestroy(m_start));
    checkCudaErrors(cudaEventDestroy(m_stop));
}

void ReductionProfiler::profiling(reduction_function reduction_impl) {
    // warm-up
    reduction_impl(m_dOutput, m_dInput, m_dataSize);

    // start cuda event
    cudaEventRecord(m_start);
    for (int i = 0; i < m_testIter; i++) {
        reduction_impl(m_dOutput, m_dInput, m_dataSize);
    }

    // event record
    cudaEventRecord(m_stop);
    checkCudaErrors(cudaEventSynchronize(m_stop));

    // print elapsed time by cuda event
    float elapsed_time_msed_event = 0.f;
    cudaEventElapsedTime(&elapsed_time_msed_event, m_start, m_stop);
    elapsed_time_msed_event /= (float)m_testIter;
    printf("CUDA event estimated - elapsed %.6f ms \n", elapsed_time_msed_event);
    float bandwidth = m_dataSize * sizeof(float) / (elapsed_time_msed_event / 1000.f) / 1e9;
    printf("Bandwidth: %.6f GB/s\n", bandwidth);

    // Copy data from device to host
    cudaMemcpy(m_hOutput, m_dOutput, sizeof(float), cudaMemcpyDeviceToHost);

    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(e));
        return;
    }

    // CPU Test
    float sum = 0;
    for (int i = 0; i < m_dataSize; i++) {
        sum += m_hInput[i];
    }

    printf("CPU Sum: %f\n", sum);
    printf("GPU Sum: %f\n", *m_hOutput);
}

int main(int argc, char* argv[]) {
    // start logs
    printf("[%s] - Starting...\n", argv[0]);

    int start = 24;
    int end = 24;

    for (int i = start; i <= end; i++) {
        int dataSize = 1 << i;
        printf("Data size: %d\n", dataSize);
        ReductionProfiler profiler(dataSize, 0);

        // printf("Reduction Shared: \n");
        // profiler.profiling(reduction_shared);
        // printf("\n");

        // printf("Reduction Sequential: \n");
        // profiler.profiling(reduction_sequential);
        // printf("\n");

        // printf("Reduction Grid stride loop: \n");
        // profiler.profiling(reduction_grid_stride_loop);
        // printf("\n");
        
        // printf("Reduction unroll last warp: \n");
        // profiler.profiling(reduction_unroll_last_warp);
        // printf("\n");
        
        // printf("Reduction unroll loop: \n");
        // profiler.profiling(reduction_unroll_loop);
        // printf("\n");
        
        // printf("Reduction vectorized: \n");
        // profiler.profiling(reduction_vectorized);
        // printf("\n");
        
        // printf("Reduction prefetech: \n");
        // profiler.profiling(reduction_prefetech);
        // printf("\n");
        
        printf("Reduction unroll: \n");
        profiler.profiling(reduction_completely_unroll);
        printf("\n");

        // printf("Reduction tuning: \n");
        // profiler.profiling(reduction_tuning);
        // printf("\n");
    }


}