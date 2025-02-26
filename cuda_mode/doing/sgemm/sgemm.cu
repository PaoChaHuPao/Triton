#include "common.h"

__global__ void naive_segmm(float* in1, float* in2, float* out, int M, int N, int K)
{
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    out[row * N + col] = 0;
    for(int i = 0; i < K; i++)
    {
        out[row * N + col] += in1[row * K + i] * in2[i * N + col];
    }
}

__global__ void naive_segmm_v1(float* in1, float* in2, float* out, int M, int N, int K)
{
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    if (row >= M || col >= N) return;
    
    float accum = 0;
    for(int i = 0; i < K; i++)
    {
        accum += in1[row * K + i] * in2[i * N + col];
    }
    out[row * N + col] = accum;
}