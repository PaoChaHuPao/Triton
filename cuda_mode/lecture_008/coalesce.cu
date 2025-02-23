#include <iostream>
#include <cuda_runtime.h>


#define N 1024*1024
#define THREADS_PER_BLOCK 1024 // This is just an example block size

__global__ void copyDataNonCoalesced(float *in, float *out, int n) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n) {
        out[index] = in[(index * 2) % n]*2;
    }
}

__global__ void copyDataCoalesced(float *in, float *out, int n) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n) {
        out[index] = in[index]*2;
    }
}

void initializeArray(float *arr, int n) {
    for(int i = 0; i < n; ++i) {
        arr[i] = static_cast<float>(i);
    }
}

int main(int argc, char* argv[]) {
    long long n, blockSize;

    if(argc != 3)
    {
        std::cout<<"This program need two params n , blockSize"<<std::endl;
        return 0;
    }
    n = 1 << atoi(argv[1]);
    blockSize = atoi(argv[2]);
    float *in, *out;

    cudaMallocManaged(&in, N * sizeof(float));
    cudaMallocManaged(&out, N * sizeof(float));

    initializeArray(in, N);

    // int blockSize = 128; // Define block size
    // int blockSize = 1024; // change this when talking about occupancy
    // int blockSize = 4096; // change this when talking about occupancy
    int numBlocks = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK; // Ensure there are enough blocks to cover all elements

    // Launch non-coalesced kernel
    copyDataNonCoalesced<<<numBlocks, THREADS_PER_BLOCK>>>(in, out, n);
    cudaDeviceSynchronize();

    initializeArray(out, n); // Reset output array

    // Launch coalesced kernel
    copyDataCoalesced<<<numBlocks, blockSize>>>(in, out, n);
    cudaDeviceSynchronize();

    cudaFree(in);
    cudaFree(out);

    return 0;
}
