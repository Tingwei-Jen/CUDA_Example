#ifndef FINDMEAN_H
#define FINDMEAN_H
#include <cuda_runtime.h>

/**
 * @brief Computes the mean value of an array of floats.
 * 
 * This function calculates the mean value of the elements in the input array
 * and stores the result in the provided mean value pointer.
 * 
 * @param d_meanValue Pointer to the memory location where the computed mean value will be stored.
 * @param d_input Pointer to the input array of floats.
 * @param numElements The number of elements in the input array.
 */
void findMean(float *d_meanValue, const float *d_input, const int numElements);

#endif // FINDMEAN_H