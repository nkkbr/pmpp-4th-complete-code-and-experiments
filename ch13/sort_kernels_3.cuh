#ifndef SORT_KERNELS_3_CUH
#define SORT_KERNELS_3_CUH

#include <cuda_runtime.h>

////////////////////////////////////////////////////////
/////// 13.5 Choice of radix value 
/////// radix = 4
////////////////////////////////////////////////////////

// .step 1
template <unsigned int SECTION_SIZE>
__global__ void radix_sort_count_block(
    const unsigned int* __restrict__ input,
    unsigned int* __restrict__ block_counts, // 长度是4*GridDim.x，前 GridDim.x 个数字是各个block中0的个数，第二个 GridDim.x 个数字是各个block中1的个数，然后是2和3的个数。
    unsigned int N,
    unsigned int iter // iter = 0,2,4,...
){
    __shared__ unsigned int s_block_ones;
    __shared__ unsigned int s_block_twos;
    __shared__ unsigned int s_block_threes;
    if (threadIdx.x == 0){
        s_block_ones = 0;
        s_block_twos = 0;
        s_block_threes = 0;
    }
    __syncthreads();

    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;

    unsigned int is_one = 0;
    unsigned int is_two = 0;
    unsigned int is_three = 0;

    if (i<N){
        is_one = (((input[i] >> iter) & 3) == 1) ? 1:0;
        is_two = (((input[i] >> iter) & 3) == 2) ? 1:0;
        is_three = (((input[i] >> iter) & 3) == 3) ? 1:0;
    }

    unsigned int mask = 0xffffffff;

    #pragma unroll
    for(int offset = 16; offset > 0; offset /= 2){
        is_one += __shfl_down_sync(mask, is_one, offset);
        is_two += __shfl_down_sync(mask, is_two, offset);
        is_three += __shfl_down_sync(mask, is_three, offset);
    }

    if ((threadIdx.x % 32) == 0) {
        atomicAdd(&s_block_ones, is_one);
        atomicAdd(&s_block_twos, is_two);
        atomicAdd(&s_block_threes, is_three);
    }

    __syncthreads();

    if(threadIdx.x==0){
        // 严格来说，s_block_ones 在shared memory上，要拉到寄存器上才能参与下面的运算
        // 而不这样显式地拉到寄存器上，其实也会被拉到寄存器上
        // 只是，显式地拉取，确保只拉取一次
        // 事实上应该没有那么蠢的编译器会拉取两次
        // 但这里，是一种习惯
        unsigned int total_ones = s_block_ones;
        unsigned int total_twos = s_block_twos;
        unsigned int total_threes = s_block_threes;

        // 0的个数要能够这样计算，要求N的数量，是BlockDim.x的倍数。我们假设我们的N了满足这个要求。
        // 例如我们保证，在将N填充成最近的BlockDim.x的倍数（使用unsigned int 可以表示的最大的数字）
        block_counts[blockIdx.x] = SECTION_SIZE - total_ones - total_twos - total_threes;
        block_counts[gridDim.x+blockIdx.x] = total_ones;
        block_counts[gridDim.x*2+blockIdx.x] = total_twos;
        block_counts[gridDim.x*3+blockIdx.x] = total_threes;
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
    __shared__ unsigned int s_scan_1[SECTION_SIZE];
    __shared__ unsigned int s_scan_2[SECTION_SIZE];
    __shared__ unsigned int s_scan_3[SECTION_SIZE];

    // 1. exclusive scan
    unsigned int i = threadIdx.x;
    unsigned int idx = blockDim.x*blockIdx.x + threadIdx.x;

    unsigned int key = input[idx];
    unsigned int bit = (key >> iter) & 3;

    s_scan_1[i] = (bit==1)?1:0;
    s_scan_2[i] = (bit==2)?1:0;
    s_scan_3[i] = (bit==3)?1:0;

    for(unsigned int stride = 1; stride<blockDim.x; stride *= 2){
        __syncthreads();
        unsigned int temp1,temp2,temp3;
        if(threadIdx.x >= stride){
            temp1 = s_scan_1[threadIdx.x] + s_scan_1[threadIdx.x-stride];
            temp2 = s_scan_2[threadIdx.x] + s_scan_2[threadIdx.x-stride];
            temp3 = s_scan_3[threadIdx.x] + s_scan_3[threadIdx.x-stride];
        }
        __syncthreads();
        if(threadIdx.x>=stride){
            s_scan_1[threadIdx.x] = temp1;
            s_scan_2[threadIdx.x] = temp2;
            s_scan_3[threadIdx.x] = temp3;
        }
    }

    __syncthreads();
    unsigned int numOnesTotal = s_scan_1[blockDim.x-1];
    unsigned int numTwosTotal = s_scan_2[blockDim.x-1];
    unsigned int numThreesTotal = s_scan_3[blockDim.x-1];

    __syncthreads();
    // 因为我们要做的是 exclusive scan
    s_scan_1[threadIdx.x] -= (bit==1)?1:0; 
    s_scan_2[threadIdx.x] -= (bit==2)?1:0;
    s_scan_3[threadIdx.x] -= (bit==3)?1:0;

    // 2. 计算出每一个元素，在block内部（即shared memory上）移动到哪个位置
    __syncthreads();

    unsigned int numOnesBefore = s_scan_1[i];
    unsigned int numTwosBefore = s_scan_2[i];
    unsigned int numThreesBefore = s_scan_3[i];
    unsigned int dst;

    switch (bit) {
    case 0:
        dst = i- numOnesBefore - numTwosBefore - numThreesBefore;
        break;
    case 1:
        dst = SECTION_SIZE - numOnesTotal - numTwosTotal - numThreesTotal + numOnesBefore;
        break;
    case 2:
        dst = SECTION_SIZE - numTwosTotal - numThreesTotal + numTwosBefore;
        break;
    case 3:
        dst = SECTION_SIZE - numThreesTotal + numThreesBefore;
        break; 
    }

    s_data[dst] = key;
    __syncthreads(); 

    unsigned int total_zeros = blockDim.x - numOnesTotal - numTwosTotal - numThreesTotal; 
    unsigned int total_zeros_ones = blockDim.x - numTwosTotal - numThreesTotal; 
    unsigned int total_zeros_ones_twos = blockDim.x - numThreesTotal; 
    if (i<total_zeros){
        output[block_offsets[blockIdx.x]+i] = s_data[i];
    } 
    else  if (i<total_zeros_ones){
        // 这里的 i 肯定 >= total_zeros
        output[block_offsets[blockIdx.x+gridDim.x]+i-total_zeros] = s_data[i];
    }
    else  if (i<total_zeros_ones_twos){
        // 这里的 i 肯定 >= total_zeros_ones
        output[block_offsets[blockIdx.x+2*gridDim.x]+i-total_zeros_ones] = s_data[i];
    }
    else {
        output[block_offsets[blockIdx.x+3*gridDim.x]+i-total_zeros_ones_twos] = s_data[i];
    }
}

#endif // SORT_KERNELS_3_CUH