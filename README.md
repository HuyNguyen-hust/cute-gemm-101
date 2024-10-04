# cute-gemm-101

## Introduction

This project is for self-learning purposes, aiming to understand and implement GEMM optimizations using Cutlass CuTe. Through a series of optimizations, this project has successfully achieved performance levels matching or exceeding cuBLAS for both single-precision (float) and half-precision (half) computations in specific configurations and on certain devices.

## Detail Implementation
Utilizes Cutlass CuTe for efficient GEMM implementation, providing a step-by-step approach to GEMM optimization

0. Official CuTe GEMM implementations: [sgemm_1.cu](https://github.com/NVIDIA/cutlass/blob/5c447dd84f8ae0e1d48ff9a2eae26ce8c4958101/examples/cute/tutorial/sgemm_1.cu) and [sgemm_2.cu](https://github.com/NVIDIA/cutlass/blob/5c447dd84f8ae0e1d48ff9a2eae26ce8c4958101/examples/cute/tutorial/sgemm_2.cu)
1. MMA Atom without Shared Memory (V00)
2. Adding Async Copy with Swizzle Shared Memory (V01)
3. Adding Pipelining (V02)
4. Adding Epilouge (V03)

## Building and Running

```
git submodule init
git submodule update
cmake -B build
cmake --build build
./build/src/profile_cuda_gemm_fp32
./build/src/profile_cuda_gemm_fp16
```

## Performance

Performance tests were conducted on an NVIDIA A100 80GB PCIe MIG 1g.10gb GPU with 9 GB of global memory and a peak memory bandwidth of 241.92 GB/s. The problem size used for testing was MxNxK = 16384 x 16384 x 16384.

### Half Precision (FP16) Performance

| Kernel Version | Latency (ms) | Effective Bandwidth (GB/s) | Effective TFLOPs | % of cuBLAS |
|----------------|--------------|----------------------------|------------------|-------------|
| cuBLAS         | 368.753      | 8.73546                    | 23.8536          | 100%        |
| Official V01   | 5210.12      | 0.618263                   | 1.68827          | 7.09%       |
| Official V02   | 2801.76      | 1.14971                    | 3.13949          | 13.17%      |
| Custom V00     | 3496.99      | 0.921142                   | 2.51533          | 10.56%      |
| Custom V01     | 472.99       | 6.81035                    | 18.5968          | 77.99%      |
| Custom V02     | 370.066      | 8.70445                    | 23.769           | 99.65%      |
| Custom V03     | 395.636      | 8.1419                     | 22.2328          | 92.79%      |

### Single Precision (FP32) Performance

| Kernel Version | Latency (ms) | Effective Bandwidth (GB/s) | Effective TFLOPs | % of cuBLAS |
|----------------|--------------|----------------------------|------------------|-------------|
| cuBLAS         | 3924.38      | 0.820825                   | 2.2414           | 100%        |
| Official V01   | 9090.04      | 0.354369                   | 0.967663         | 43.17%      |
| Official V02   | 8742.43      | 0.368459                   | 1.00614          | 44.91%      |
| Custom V00     | 3604.79      | 0.893596                   | 2.44011          | 108.87%     |

Note: Performance may vary based on hardware and specific implementation details. These results are for comparative purposes within this learning project.

## Acknowledgements

This project is based on the work of Reed, as described in the article [CUDA 矩阵乘法终极优化指南](https://zhuanlan.zhihu.com/p/675308830). The implementation heavily relies on the Cutlass CuTe library, and I encourage readers to refer to the [official Cutlass CuTe documentation](https://github.com/NVIDIA/cutlass/tree/master/media/docs/cute) for more detailed information on the techniques used.
