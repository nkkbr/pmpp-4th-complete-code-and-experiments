#ifndef SORT_KERNELS_2_CUH
#define SORT_KERNELS_2_CUH

#include <cuda_runtime.h>

////////////////////////////////////////////////////////
/////// 13.4 Optimizing for memory coalescing
////////////////////////////////////////////////////////

// .step 1
template <unsigned int SECTION_SIZE>
__global__ void radix_sort_count_block(
    const unsigned int* __restrict__ input,
    unsigned int* __restrict__ block_counts, // 长度是2*GridDim.x，前 GridDim.x 个数字是各个block中0的个数，后面 GridDim.x 个数字是各个block中1的个数。
    unsigned int N,
    unsigned int iter
){
    __shared__ unsigned int s_block_ones;
    if (threadIdx.x == 0){
        s_block_ones = 0;
    }
    __syncthreads();

    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;

    unsigned int is_one = 0;
    if (i<N){
        is_one = (input[i] >> iter) & 1;
    }

    unsigned int mask = 0xffffffff;

    #pragma unroll
    for(int offset = 16; offset > 0; offset /= 2){
        is_one += __shfl_down_sync(mask, is_one, offset);
    }

    if ((threadIdx.x % 32) == 0) {
        atomicAdd(&s_block_ones, is_one);
    }

    __syncthreads();

    if(threadIdx.x==0){
        // 严格来说，s_block_ones 在shared memory上，要拉到寄存器上才能参与下面的运算
        // 而不这样显式地拉到寄存器上，其实也会被拉到寄存器上
        // 只是，显式地拉取，确保只拉取一次
        // 事实上应该没有那么蠢的编译器会拉取两次
        // 但这里，是一种习惯
        unsigned int total_ones = s_block_ones;

        // 0的个数要能够这样计算，要求N的数量，是BlockDim.x的倍数。我们假设我们的N了满足这个要求。
        // 例如我们保证，在将N填充成最近的BlockDim.x的倍数（使用unsigned int 可以表示的最大的数字）
        block_counts[blockIdx.x] = SECTION_SIZE - total_ones;
        block_counts[gridDim.x+blockIdx.x] = total_ones;
    }
}


// .step2
// 我们使用 thrust::exclusive_scan


// .step 3
// 我们的kernel约定，N 是 BlockDim.x的倍数，所以我们就不做 i＜N 的判断了
// SECTION_SIZE 就是 BlockDim.x
template <unsigned int SECTION_SIZE>
__global__ void radix_sort_coalesced_scatter(
    const unsigned int* __restrict__ input,
    unsigned int* __restrict__ output,
    const unsigned int* __restrict__ block_offsets,
    unsigned int iter
){
    __shared__ unsigned int s_data[SECTION_SIZE];
    __shared__ unsigned int s_scan[SECTION_SIZE];

    // 1. exclusive scan
    unsigned int i = threadIdx.x;
    unsigned int idx = blockDim.x*blockIdx.x + threadIdx.x;

    unsigned int key = input[idx];
    unsigned int bit = (key >> iter) & 1;

    s_scan[i] = bit;

    for(unsigned int stride = 1; stride<blockDim.x; stride *= 2){
        __syncthreads();
        unsigned int temp;
        if(threadIdx.x >= stride){
            temp = s_scan[threadIdx.x] + s_scan[threadIdx.x-stride];
        }
        __syncthreads();
        if(threadIdx.x>=stride){
            s_scan[threadIdx.x] = temp;
        }
    }

    s_scan[threadIdx.x] -= bit; // 因为我们要做的是 exclusive scan

    // 2. 计算出每一个元素，在block内部（即shared memory上）移动到哪个位置
    __syncthreads();
    unsigned int lastBit = (input[blockDim.x*(blockIdx.x+1)-1] >> iter ) & 1;
    unsigned int numOnesTotal = s_scan[SECTION_SIZE-1] + lastBit;

    unsigned int numOnesBefore = s_scan[i];
    unsigned int dst = (bit == 0) ? (i-numOnesBefore):(SECTION_SIZE-numOnesTotal+numOnesBefore);

    s_data[dst] = key;
    __syncthreads(); // 等待所有的线程都好了
    // 事实上，我们似乎也可以不要这个 s_data 直接写入global memory。
    // 在正确性上，是没有问题的。因为每一个元素被写入的位置是固定且唯一的。
    // 但是，移动到s_data上，再写入global memory的话，这个对 global memory 的写入就是合并的（coalesced）
    // 代价是，我们要有一块s_data，先把数据写在上面。

    unsigned int total_zeros = blockDim.x - numOnesTotal;
    if (i<total_zeros){
        output[block_offsets[blockIdx.x]+i] = s_data[i];
    } else {
        output[block_offsets[blockIdx.x+gridDim.x]+i-total_zeros] = s_data[i];
    }
}


#endif // SORT_KERNELS_2_CUH