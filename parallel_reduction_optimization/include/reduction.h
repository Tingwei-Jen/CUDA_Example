#ifndef REDUCTION_H
#define REDUCTION_H
#include <cuda_runtime.h>

void reduction_shared(float *d_output, const float *d_input, const int numElements);
void reduction_sequential(float *d_output, const float *d_input, const int numElements);
void reduction_grid_stride_loop(float *d_output, const float *d_input, const int numElements);
void reduction_unroll_last_warp(float *d_output, const float *d_input, const int numElements);
void reduction_unroll_loop(float *d_output, const float *d_input, const int numElements);
void reduction_vectorized(float *d_output, const float *d_input, const int numElements);
void reduction_prefetech(float *d_output, const float *d_input, const int numElements);
void reduction_completely_unroll(float *d_output, const float *d_input, const int numElements);
void reduction_tuning(float *d_output, const float *d_input, const int numElements);
#endif // REDUCTION_H