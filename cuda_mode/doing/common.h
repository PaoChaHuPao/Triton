#ifndef COMMON
#define COMMON

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>

using namespace std;

#define BLOCK_SIZE  1024
#define WARP_SIZE 32

#define FLOAT4(value) (reinterpret_cast<float4*>(&(value))[0])
#define CEIL(value,align_num) ((value+align_num-1)/(align_num))
#define cudaCheckSize(kernel) _cudaCheckSize(kernel);
template<class T>
void _cudaCheckSize(T kernel)
{
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, device);

    // 获取最大驻留块数
    int maxBlocksPerSM;
    cudaDeviceGetAttribute(&maxBlocksPerSM, cudaDevAttrMaxBlocksPerMultiprocessor, device);
    std::cout << "Maximum number of resident blocks per SM: " << maxBlocksPerSM << std::endl;

    // 获取最大驻留线程数
    int maxThreadsPerSM = deviceProp.maxThreadsPerMultiProcessor;
    std::cout << "Maximum number of resident threads per SM: " << maxThreadsPerSM << std::endl;


    //获取grid_size上限
    int maxGridX;
    cudaDeviceGetAttribute(&maxGridX, cudaDevAttrMaxGridDimX, device);
    std::cout << "Maximum x-dimension of a grid of thread blocks: " << maxGridX << std::endl;
    
    // 获取 SM 数量
    std::cout << "Maximum number of threads per block: " << deviceProp.maxThreadsPerBlock << std::endl;
    std::cout << "Maximum x-dimension of a block: " << deviceProp.maxThreadsDim[0] << std::endl;
    std::cout << "Maximum y-dimension of a block: " << deviceProp.maxThreadsDim[1] << std::endl;
    std::cout << "Maximum z-dimension of a block: " << deviceProp.maxThreadsDim[2] << std::endl;
    std::cout << "GPU SM Count: " << deviceProp.multiProcessorCount << std::endl;


    int blockSize, minGridSize; 
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, kernel, 0, 0);

    // Print suggested block size and minimum grid size
    std::cout << "Recommended block size: " << blockSize
            << ", Minimum grid size: " << minGridSize << std::endl;
}

#define cudaCheck(err) _cudaCheck(err, __FILE__, __LINE__)
void _cudaCheck(cudaError_t error, const char *file, int line) {
    if (error != cudaSuccess) {
        printf("[CUDA ERROR] at file %s(line %d):\n%s\n", file, line, cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }
    return;
};

#endif
