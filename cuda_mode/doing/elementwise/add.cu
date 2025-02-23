#include "common.h"

__global__ void naive_add(float* in1, float* in2, float* out, int N)
{
    int index = threadIdx.x + blockIdx.x*blockDim.x;
    if(index < N)
    {
        out[index] = in1[index] + in2[index];
    }
}

__global__ void add_float4(float* in1, float* in2, float* out, int N)
{
    int index = (threadIdx.x + blockIdx.x*blockDim.x)*4;
    float4 tmp_in1 = FLOAT4(in1[index]);
    float4 tmp_in2 = FLOAT4(in2[index]);
    float4 tmp_out;
    if(index < N)
    {
        tmp_out.x = tmp_in1.x + tmp_in2.x;
        tmp_out.y = tmp_in1.y + tmp_in2.y;
        tmp_out.z = tmp_in1.z + tmp_in2.z;
        tmp_out.w = tmp_in1.w + tmp_in2.w;
        // printf("this is out[%d]:%f\n", index+3, tmp_out.w);
        FLOAT4(out[index]) = tmp_out;
    }

    return;
}


int main()
{
    // int size = (1 << 20);
    int size = 1584* 4 * BLOCK_SIZE;

    cudaCheckSize(naive_add);

    float* in_h = (float*) (malloc(sizeof(float) * size));
    float* out_h = (float*) (malloc(sizeof(float) * size));
    for(int i = 0; i < size; i++)
    {
        in_h[i] = 1;
    }

    float* in1_d,* in2_d,* out_d;
    cudaMalloc((void**)&in1_d, size * sizeof(float));
    cudaMalloc((void**)&in2_d, size * sizeof(float));
    cudaMalloc((void**)&out_d, size * sizeof(float));
    cudaMemcpy(in1_d, in_h, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(in2_d, in_h, size * sizeof(float), cudaMemcpyHostToDevice);

    int naive_grid_size = CEIL(size, BLOCK_SIZE);
    naive_add<<<naive_grid_size, BLOCK_SIZE>>>(in1_d, in2_d, out_d, size);
    cudaMemcpy(out_h, out_d, size * sizeof(float), cudaMemcpyDeviceToHost);
    std::cout<<out_h[size-1]<<std::endl;


    int float4_grid_size = CEIL(CEIL(size, BLOCK_SIZE),4);
    add_float4<<<float4_grid_size, BLOCK_SIZE>>>(in1_d, in2_d, out_d, size);
    cudaMemcpy(out_h, out_d, size * sizeof(float), cudaMemcpyDeviceToHost);
    std::cout<<out_h[size-1]<<std::endl;

    cudaFree(in1_d); cudaFree(in2_d); cudaFree(out_d);
    free(in_h); free(out_h);
    return 0;
}
