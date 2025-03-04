#include "common.h"

__global__ void naive_sum(float* in, float* out, int n)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if(index < n)
    {
        atomicAdd(out, in[index]);
    }
}

__global__ void stride_sum(float* in, float* out, int n)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int thridx = threadIdx.x;
    
    __shared__ float inputs[BLOCK_SIZE];
    inputs[thridx] = index < n ? in[index] : 0.0;
    __syncthreads();
    
    for(int stride = BLOCK_SIZE/2; stride >= 1; stride /= 2)
    {
        if(thridx < stride) {inputs[thridx] += inputs[thridx+stride];}
        __syncthreads();
    }

    if(thridx == 0)
    {
        // printf("block:%d sum is %d\n", blockIdx.x * blockDim.x, in[blockIdx.x * blockDim.x]);
        atomicAdd(out, inputs[thridx]);
    }
}

__global__ void warp_sum(float* in, float* out, int n)
{
    __shared__ float inputs[BLOCK_SIZE / WARP_SIZE];

    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int thridx = threadIdx.x;
    int warp_id = thridx / WARP_SIZE;
    int warp_offset = thridx % WARP_SIZE;

    int val = index < n ? in[index] : 0.0;
    for(int offset = WARP_SIZE >> 1; offset >= 1; offset >>= 1)
    {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    __syncthreads();
    if(warp_offset == 0)
    {
        inputs[warp_id] = val;
    }
    __syncthreads();
    if(warp_id == 0)
    {
        for(int stride = BLOCK_SIZE / WARP_SIZE / 2; stride >= 1; stride /= 2)
        {
            if(thridx < stride) {inputs[thridx] += inputs[thridx+stride];}
            __syncthreads();
        }
    }

    if(thridx == 0)
    {
        // printf("block:%d sum is %d\n", blockIdx.x * blockDim.x, in[blockIdx.x * blockDim.x]);
        atomicAdd(out, inputs[thridx]);
    }
}

__global__ void warp_sum_v2(float* in, float* out, int n)
{
    __shared__ float inputs[32];

    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int thridx = threadIdx.x;
    int warp_id = thridx / WARP_SIZE;
    int warp_offset = thridx % WARP_SIZE;

    int val = index < n ? in[index] : 0.0;
    for(int offset = WARP_SIZE >> 1; offset >= 1; offset >>= 1)
    {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    //no need sync cause in one warp all threads are sync
    // __syncthreads();
    
    if(warp_offset == 0)
    {
        inputs[warp_id] = val;
    }
    __syncthreads();
    if(warp_id == 0)
    {
        for(int offset = WARP_SIZE >> 1; offset >= 1; offset >>= 1)
        {
            inputs[warp_offset] += warp_offset+offset < BLOCK_SIZE / WARP_SIZE ? inputs[warp_offset+offset] : 0.0;
            __syncwarp();
        }

        // int warpNum = BLOCK_SIZE / WARP_SIZE;  // 每个block中的warp数量
        // val = (warp_offset < warpNum) ? inputs[warp_offset] : 0.0f;
        // for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
        //     val += __shfl_down_sync(0xFFFFFFFF, val, offset);
        // }
    }

    if(thridx == 0) // which equls warp_id == 0 && warp_offset == 0
    {
        // printf("block:%d sum is %d\n", blockIdx.x * blockDim.x, in[blockIdx.x * blockDim.x]);
        atomicAdd(out, inputs[thridx]);
    }
}

__global__ void warp_sum_v3(float* in, float* out, int n)
{
    __shared__ float inputs[32];

    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int thridx = threadIdx.x;
    int warp_id = thridx / WARP_SIZE;
    int warp_offset = thridx % WARP_SIZE;

    int val = index < n ? in[index] : 0.0;
    for(int offset = WARP_SIZE >> 1; offset >= 1; offset >>= 1)
    {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    //no need sync cause in one warp all threads are sync
    // __syncthreads();
    
    if(warp_offset == 0)
    {
        inputs[warp_id] = val;
    }
    __syncthreads();
    if(warp_id == 0)
    {
        // for(int offset = WARP_SIZE >> 1; offset >= 1; offset >>= 1)
        // {
        //     inputs[warp_offset] += warp_offset+offset < BLOCK_SIZE / WARP_SIZE ? inputs[warp_offset+offset] : 0.0;
        //     __syncwarp();
        // }

        int warpNum = BLOCK_SIZE / WARP_SIZE;  // 每个block中的warp数量
        val = (warp_offset < warpNum) ? inputs[warp_offset] : 0.0f;
        for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
            val += __shfl_down_sync(0xFFFFFFFF, val, offset);
        }
    }

    if(thridx == 0) // which equls warp_id == 0 && warp_offset == 0
    {
        // printf("block:%d sum is %d\n", blockIdx.x * blockDim.x, in[blockIdx.x * blockDim.x]);
        atomicAdd(out, inputs[thridx]);
    }
}


int main()
{
    atomicMax()
    int size = 1 << 20;
    float* in_h = new float[size];
    float* out_h = new float;
    for(int i = 0; i < size; i++)
    {
        in_h[i] = 0;
    }

    float* in_d, * out_d;
    cudaMalloc((void**) &in_d, size * sizeof(float));
    cudaMalloc((void**) &out_d, sizeof(float));
    cudaMemcpy(in_d, in_h, size * sizeof(float), cudaMemcpyHostToDevice);

    int naive_grid_size = CEIL(size, BLOCK_SIZE);
    naive_sum<<<naive_grid_size, BLOCK_SIZE>>>(in_d, out_d, size);
    cudaMemcpy(out_h, out_d, sizeof(float), cudaMemcpyDeviceToHost);
    cout<<"naive_sum:"<<out_h[0]<<endl;

    int stride_grid_size = CEIL(size, BLOCK_SIZE);
    stride_sum<<<stride_grid_size, BLOCK_SIZE>>>(in_d, out_d, size);
    cudaMemcpy(out_h, out_d, sizeof(float), cudaMemcpyDeviceToHost);
    cout<<"stride_sum:"<<out_h[0]<<endl;

    int warp_grid_size = CEIL(size, BLOCK_SIZE);
    warp_sum_v2<<<warp_grid_size, BLOCK_SIZE>>>(in_d, out_d, size);
    cudaMemcpy(out_h, out_d, sizeof(float), cudaMemcpyDeviceToHost);
    cout<<"warp_sum:"<<out_h[0]<<endl;

    warp_sum_v3<<<warp_grid_size, BLOCK_SIZE>>>(in_d, out_d, size);
    cudaMemcpy(out_h, out_d, sizeof(float), cudaMemcpyDeviceToHost);
    cout<<"warp_sum:"<<out_h[0]<<endl;

    return 0;
}