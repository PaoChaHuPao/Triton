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

template<int BM, int BN, int BK, int TM, int TN> 
// BM BN BK means shared mem size, TM TN means one thread needs process TM*TN. One block process blocksize*TM*TN  
// pay attention: we think one block share same shared mem u`d better BM BN equals blockDim.y*TM blockDim.x*TN
// which i think is important to understand tile algorithm
// suppose 128 128 8 8 8
//这个整体是对着照抄一边实现的
__global__ void segmm_tile_v1(float* in1, float* in2, float* out, int M, int N, int K)
{
    //这段程序难写的点在于每个维度缩放下的索引
    int block_x = blockIdx.x;
    int block_y = blockIdx.y;
    int thread_x = threadIdx.x;
    int thread_y = threadIdx.y;

    __shared__ float in1_s[BM * BK];
    __shared__ float in2_s[BK * BN];

    float tmp[TM * TN] = {0.};
    //移动in1 in2 out 到当前block内方便后续
    float* in1_ = &in1[by * BM * K]; //in1移动到对应的行
    float* in2_ = &in2[bx * BN];    //in2移动到列
    float* out_ = &out[by * BM * N + bx * BN];  //out需要移动到具体某一块
    
    //下面的tile索引是为了搬运global 到 shared中所作
    int a_tile_row = (threadIdx.x + threadIdx.y * blockDim.x) / BK;
    int a_tile_col = (threadIdx.x + threadIdx.y * blockDim.x) % BK;
    int b_tile_row = (threadIdx.x + threadIdx.y * blockDim.x) / BN;
    int b_tile_col = (threadIdx.x + threadIdx.y * blockDim.x) % BN;


    for(int k = 0; k < K; k += BK)
    {
        //加载shared mem
        //需要注意这里在加载shared mem的时候按照thread只处理一个数据, 否则导致负载不均衡一个block中有16*16个thread的话 那么至少要搬运 128*8 / 256 = 4轮的搬运
        //也就是下面的两个搬运的循环次数,这里面有很多暗含的=关系可能会导致在第一次手动索引的时候迷惑
        //同时注意由于block并不和单纯shared mem版本是一一对齐的 需要对加载shared mem时单独进行切块处理
        for(int i = 0; i < BM * BK / blockDim.x / blockDim.y ; i++)
        {
            in1_s[i * blockDim.x * blockDim.y + a_tile_row * BK + a_tile_col] = in1[((i * blockDim.x * blockDim.y) / BK + a_tile_row) * K  + a_tile_col];
        }
        for(int i = 0; i < BK * BN / blockDim.x / blockDim.y ; i++)
        {
            in2_s[i * blockDim.x * blockDim.y + b_tile_row * BN + b_tile_col] = in2[((i * blockDim.x * blockDim.y) / BN + b_tile_row) * N  + b_tile_col];
        }
        __syncthreads();
        in1_ += BK;
        in2_ += BK * N;
        for (int j = 0; j < TM; j++) 
        {
            for (int l = 0; l < TN; l++)
            {
                for (int i = 0; i < BK; i++) 
                    tmp[j][l] += As[thread_y * TM * BK + j * BK + i] * Bs[i * BN + thread_x * TN + l];
            }
        }
        __syncthreads();
    }

    for(int tm = 0; tm < TM; tm++)
    {
        for(int tn = 0; tn < TN; tn++)
        {
            out[ (thread_y *  blockDim.y + thread_x) * TM * TN  + tm * TM + tn] = tmp[tm * TM + tn]; //注意这里的thread也是一块一块的
        }
    }
   
}

