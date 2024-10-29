# Cuda Parallel Reduction Optimization

## Optimization Goal

- Choose the right metric:
    - GFLOP/s: for compute-bound kernels
    - Bandwidth: for memory-bound kernels
        - Reductions have very low arithmetic intensity, with 1 flop per element loaded (bandwidth-optimal)
- In this project, achieve 92% bandwidth of NVIDIA 3070Ti, which has a bandwidth of 608 GB/s.

### GB/s at 16M Array

<!-- benchmark_results -->
| Kernels                              |  GB/s  | Performance Relative to Theoretical Value |
|:-------------------------------------|---------:|:-----------------------------------------|
| Shared                               | `85.9` | 14.1%                                     |
| Sequential                           | `110.6` | 18.2%                                     |
| Grid Stride Loop                     | `536.7`  | 88.2%                                    |
| Unroll Last Warp                     | `540.1`  | 88.8%                                    |
| Unroll Loop                          | `539.7`   | 88.7%                                    |
| Vectorized                           | `558.5`  | 91.8%                                    |
| Prefetch                             | `552.5`   | 90.8%                                    |
| Completely Unroll                    | `559.3`  | 91.9%                                    |
| Theoretical Bandwidth                | `608.3`  | 100.0%                                   |
<!-- benchmark_results -->


## Build the Project

To build the project, follow these steps:

1. Open a terminal and navigate to the `parallel_reduction_optimization` directory.
2. Create a `build` directory by running the following command:
    ```bash
    mkdir build
    ```
3. Navigate to the `build` directory:
    ```bash
    cd build
    ```
4. Run the following command to configure the project:
    ```bash
    cmake ..
    ```
5. Build the project using the following command:
    ```bash
    make -j4
    ```
6. Run the reduction:
    ```bash
    ./reduction
    ```

