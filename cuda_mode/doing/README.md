# common.h

该头文件中主要定义一些常用的宏,以及blocksize的选择
其中对于cudaCheckSize中使用template来达到编译时指定函数类型

这些是通过cuda_runtime获取到的有关本卡(4070TS的相关数据)
Maximum number of resident blocks per SM: 24
Maximum number of resident threads per SM: 1536
Maximum x-dimension of a grid of thread blocks: 2147483647()
Maximum number of threads per block: 1024
Maximum x-dimension of a block: 1024
Maximum y-dimension of a block: 1024
Maximum z-dimension of a block: 64
GPU SM Count: 66

那么对于block_size的选择至少是threads per SM/blocks per SM 1536/24 = 64 上限为1024 以及warp(32倍数)

> 与 block 对应的硬件级别为 SM，SM 为同一个 block 中的线程提供通信和同步等所需的硬件资源，跨 SM 不支持对应的通信，所以一个 block 中的所有线程都是执行在同一个 SM 上的，而且因为线程之间可能同步，所以一旦 block 开始在 SM 上执行，block 中的所有线程同时在同一个 SM 中执行（并发，不是并行），也就是说 block 调度到 SM 的过程是原子的。SM 允许多于一个 block 在其上并发执行，如果一个 SM 空闲的资源满足一个 block 的执行，那么这个 block 就可以被立即调度到该 SM 上执行，具体的硬件资源一般包括寄存器、shared memory、以及各种调度相关的资源，这里的调度相关的资源一般会表现为两个具体的限制，Maximum number of resident blocks per SM 和 Maximum number of resident threads per SM ，也就是 SM 上最大同时执行的 block 数量和线程数量。因为 GPU 的特点是高吞吐高延迟，就像一个自动扶梯一分钟可以运送六十个人到另一层楼，但是一个人一秒钟无法通过自动扶梯到另一层楼，要达到自动扶梯可以运送足够多的人的目标，就要保证扶梯上同一时间有足够多的人，对应到 GPU，就是要尽量保证同一时间流水线上有足够多的指令。
> 要到达这个目的有多种方法，其中一个最简单的方法是让尽量多的线程同时在 SM 上执行，SM 上并发执行的线程数和SM 上最大支持的线程数的比值，被称为 Occupancy，更高的 Occupancy 代表潜在更高的性能。显然，一个 kernel 的 block_size 应大于 SM 上最大线程数和最大 block 数量的比值，否则就无法达到 100% 的 Occupancy，对应不同的架构，这个比值不相同，对于 V100 、 A100、 GTX 1080 Ti 是 2048 / 32 = 64，对于 RTX 3090 是 1536 / 16 = 96，所以为了适配主流架构，如果静态设置 block_size 不应小于 96。考虑到 block 调度的原子性，那么 block_size 应为 SM 最大线程数的约数，否则也无法达到 100% 的 Occupancy，主流架构的 GPU 的 SM 最大线程数的公约是 512，96 以上的约数还包括 128 和 256，也就是到目前为止，block_size 的可选值仅剩下 128 / 256 / 512 三个值
> 我们可以想象，GPU 一次可以调度 SM 数量 * 每个 SM 最大 block 数个 block，因为每个 block 的计算量相等，所以所有 SM 应几乎同时完成这些 block 的计算，然后处理下一批，这其中的每一批被称之为一个 wave。想象如果 grid_size 恰好比一个 wave 多出一个 block，因为 stream 上的下个 kernel 要等这个 kernel 完全执行完成后才能开始执行，所以第一个 wave 完成后，GPU 上将只有一个 block 在执行，GPU 的实际利用率会很低，这种情况被称之为 tail effect，我们应尽量避免这种情况。将 grid_size 设置为精确的一个 wave 可能也无法避免 tail effect，因为 GPU 可能不是被当前 stream 独占的，常见的如 NCCL 执行时会占用一些 SM。所以无特殊情况，可以将 grid_size 设置为数量足够多的整数个 wave，往往会取得比较理想的结果，如果数量足够多，不是整数个 wave 往往影响也不大。
> 综上所述，普通的 elementwise kernel 或者近似的情形中，block_size 设置为 128，grid_size 设置为可以满足足够多的 wave 就可以得到一个比较好的结果了。但更复杂的情况还要具体问题具体分析，比如如果因为 shared_memory 的限制导致一个 SM 只能同时执行很少的 block，那么增加 block_size 有机会提高性能，如果 kernel 中有线程间同步，那么过大的 block_size 会导致实际的 SM 利用率降低，这些我们有机会单独讨论。

------[https://zhuanlan.zhihu.com/p/442304996]()

**但是请注意如果选择blocksize为1024  ncu中给出的理论occupancy为66.7% 并非为100%**

# add.cu

问题1:  add.cu中 使用float4访存版本的kernel的occupancy反而比naive的低这个问题还没有什么头绪.
想法是提高访存带宽

# sum.cu

```
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
```

这是自己实现的warp版本

1. 优化空间1:

```
for(int offset = WARP_SIZE >> 1; offset >= 1; offset >>= 1)
{
val += __shfl_down_sync(0xFFFFFFFF, val, offset);
}
__syncthreads();
```

对于一个warp中的thread是完全同步的,for结束之后是可以不需要sync

2. 优化空间2:

```
if(warp_id == 0)
{
        for(int stride = BLOCK_SIZE / WARP_SIZE / 2; stride >= 1; stride /= 2)
        {
        if(thridx < stride) {inputs[thridx] += inputs[thridx+stride];}
        __syncthreads();
        }
}
```

这里在一个warp中间也应该使用warp来做减少sync次数

3. 后续优化时产生的问题

```
for(int offset = WARP_SIZE >> 1; offset >= 1; offset >>= 1)
{
        inputs[warp_offset] += warp_offset+offset < BLOCK_SIZE / WARP_SIZE ? inputs[warp_offset+offset] : 0.0;
        __syncwarp();
}
```

```
int warpNum = BLOCK_SIZE / WARP_SIZE;
val = (warp_offset < warpNum) ? inputs[warp_offset] : 0.0f;
for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
}
```

这两段代码都是对一个warp中的thread同步操作

通过ncu检测上述三个版本的代码
(
naive版本
自己第一遍
__syncwarp()
__shfl_down_sync(0xFFFFFFFF, val, offset)
)

```
#size = 1 << 20 , BLOCK_SIZE = 1024
ncu ./sum | grep Duration
    Duration                      msecond         1.58
    Duration                      usecond        26.98
    Duration                      usecond        25.92
    Duration                      usecond        19.04
ncu ./sum | grep Occupancy
    Theoretical Occupancy                     %        66.67
    Achieved Occupancy                        %        38.00
    Theoretical Occupancy                     %        66.67
    Achieved Occupancy                        %        65.12
    Theoretical Occupancy                     %        66.67
    Achieved Occupancy                        %        42.49
    Theoretical Occupancy                     %        66.67
    Achieved Occupancy                        %        60.02
```

```
#size = 1 << 20 , BLOCK_SIZE = 512
ncu ./sum | grep Duration
    Duration                      msecond         1.58
    Duration                      usecond        16.93
    Duration                      usecond        20.61
    Duration                      usecond        13.92

ncu ./sum | grep Occupancy
    Theoretical Occupancy                     %          100
    Achieved Occupancy                        %        70.01
    Theoretical Occupancy                     %          100
    Achieved Occupancy                        %        87.68
    Theoretical Occupancy                     %          100
    Achieved Occupancy                        %        61.55
    Theoretical Occupancy                     %          100
    Achieved Occupancy                        %        81.23
```

```
#size = 1 << 20 , BLOCK_SIZE = 256
ncu ./sum | grep Duration
    Duration                      msecond         1.58
    Duration                      usecond        14.98
    Duration                      usecond        25.02
    Duration                      usecond        12.83 
ncu ./sum | grep Occupancy
    Theoretical Occupancy                     %          100
    Achieved Occupancy                        %        79.01
    Theoretical Occupancy                     %          100
    Achieved Occupancy                        %        85.66
    Theoretical Occupancy                     %          100
    Achieved Occupancy                        %        53.10
    Theoretical Occupancy                     %          100
    Achieved Occupancy                        %        79.65
```

```
#size = 1 << 20 , BLOCK_SIZE = 128
ncu ./sum | grep Duration
    Duration                      msecond         1.90
    Duration                      usecond        14.53
    Duration                      usecond        34.24
    Duration                      usecond        14.91
ncu ./sum | grep Occupancy
    Theoretical Occupancy                     %          100
    Achieved Occupancy                        %        80.28
    Theoretical Occupancy                     %          100
    Achieved Occupancy                        %        83.24
    Theoretical Occupancy                     %          100
    Achieved Occupancy                        %        76.51
    Theoretical Occupancy                     %          100
    Achieved Occupancy                        %        74.05
```

可以看出的确是__shfl_down_sync(0xFFFFFFFF, val, offset)更好,并且使用__syncwarp()对性能反向优化了,Duration项时间明显降低,Occupancy也负优化

# transpose.cu

之前看cuda_mode的时候遇到了Swizzle这个概念 以及bank confict![](https://pic1.zhimg.com/v2-7cb74504a5c7b81130bc6ad4d562220c_1440w.jpg)
**在访问shared memory时，因多个线程读写*同一个Bank中的不同数据地址*时，导致shared memory 并发读写退化成顺序读写的现象叫做Bank Conflict；**

与之对应的就是***同一个Bank中的相同数据地址时并不会导致***bank conflict
这里主要想要知道

另外关于读写同一地址时的行为，在[NVProfGuide](https://link.zhihu.com/?target=https%3A//docs.nvidia.com/nsight-compute/ProfilingGuide/index.html)里，给出了更明确的流程：

> When multiple threads make the same read access, one thread receives the data and then broadcasts it to the other threads. When multiple threads write to the same location, only one thread succeeds in the write; which thread that succeeds is undefined.

即，

* 多个线程读同一个数据时，仅有一个线程读，然后broadcast到其他线程
* 多个线程写同一个数据时，仅会有一个线程写成功（不过这里没有提及是否会将写操作执行多次（即a. 多个线程写入，最后一个线程随机写完; or b. 随机挑选一个线程执行写入），具体流程存疑）

```
__shared__ float s_mem[BLOCK_SIZE][BLOCK_SIZE + 1];  // 避免bank conflict
```

为什么能解决bank conflict?
shared mem会被划分为32个bank,并且按照连续的0~31个bank连续的将一块shared mem给划分,如果按照32x32那么每一列都属于一个bank

| 0 | 1 | 2 | 3 | ... | 31 |
| - | - | - | - | --- | -- |
| 0 | 1 | 2 | 3 |     | 31 |
| 0 | 1 | 2 | 3 | ... | 31 |

如果改为32*33 能够使得每一列的bank错位

| 0 | 1 | 2 | ,,, | 31 | 0 |
| - | - | - | --- | -- | - |
| 1 | 2 | 3 | ... | 0  | 1 |
| 2 | 3 | 4 | ... | 1  | 2 |

对于即将被按列读取的这块shared mem来说解决了bank conflict
