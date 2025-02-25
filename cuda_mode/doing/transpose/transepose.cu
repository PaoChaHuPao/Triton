#include "common.h"

__global__ void transpose(int* in, int* out, int M, int N) {
    // input的row和col
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;

    if (row < M && col < N) {
        out[col * M + row] = in[row * N + col];
    }
}


template <const int TILE_DIM>
__global__ void matrix_trans_swizzling(int* in, int* out, int M, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
  
    __shared__ int s_data[32][32];
  
    if (row < M && col < N) {
      // 从全局内存读取数据写入共享内存的逻辑坐标(row=x,col=y)
      // 其映射的物理存储位置位置(row=x,col=x^y)
      s_data[threadIdx.x][threadIdx.x ^ threadIdx.y] = in[row * N + col];
      __syncthreads();
      int n_col = blockIdx.y * blockDim.y + threadIdx.x;
      int n_row = blockIdx.x * blockDim.x + threadIdx.y;
      if (n_row < N && n_col < M) {
        // 从共享内存的逻辑坐标(row=y,col=x)读取数据
        // 其映射的物理存储位置(row=y,col=x^y)
        out[n_row * M + n_col] = s_data[threadIdx.y][threadIdx.x ^ threadIdx.y];
      }
    }
  }




int main()
{
    int size = 1 << 20;

    return 0;
}