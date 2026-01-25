#ifndef SORT_KERNELS_5_CUH
#define SORT_KERNELS_5_CUH

#include <cuda_runtime.h>

////////////////////////////////////////////////////////
/////// 13.5 Choice of radix value (optimized version)
/////// radix = 4
////////////////////////////////////////////////////////

// .step 1
template <unsigned int SECTION_SIZE>
__global__ void radix_sort_count_block(
    // 我们不传入数据总量 N，我们确保它是blockDim.x的倍数，并确保gridDim.x正好覆盖所有的数据。
    const unsigned int* __restrict__ input,
    unsigned int* __restrict__ block_counts, // 长度是4*GridDim.x，前 GridDim.x 个数字是各个block中0的个数，第二个 GridDim.x 个数字是各个block中1的个数，然后是2和3的个数。
    unsigned int iter // iter = 0,2,4,...
){
    const unsigned int i = blockDim.x*blockIdx.x + threadIdx.x;

    unsigned int digit = (input[i] >> iter) & 3;
    unsigned int m1 = __ballot_sync(0xffffffff, digit == 1);
    unsigned int m2 = __ballot_sync(0xffffffff, digit == 2);
    unsigned int m3 = __ballot_sync(0xffffffff, digit == 3);

    unsigned int c1 = __popc(m1);
    unsigned int c2 = __popc(m2);
    unsigned int c3 = __popc(m3);

    // 私有槽位，每个warp一个
    __shared__ unsigned int warp_c1[32];
    __shared__ unsigned int warp_c2[32];
    __shared__ unsigned int warp_c3[32];

    const unsigned int lane = threadIdx.x & 31;
    const unsigned int warpId = threadIdx.x >> 5;

    if (lane == 0) {
        warp_c1[warpId] = c1;
        warp_c2[warpId] = c2;
        warp_c3[warpId] = c3;
    }

    __syncthreads();

    if (threadIdx.x == 0){
        unsigned int t1=0, t2=0, t3=0;
        int numWarps = (blockDim.x + 31) >> 5;

        #pragma unroll
        for(unsigned int w=0; w<32; ++w){
            if (w < numWarps) {
                t1 += warp_c1[w];
                t2 += warp_c2[w];
                t3 += warp_c3[w];
            }
        }

        block_counts[blockIdx.x] = SECTION_SIZE - t1 - t2 - t3; // 需要保证 SECTION_SIZE == blockDim.x
        block_counts[gridDim.x+blockIdx.x] = t1;
        block_counts[gridDim.x*2+blockIdx.x] = t2;
        block_counts[gridDim.x*3+blockIdx.x] = t3;
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

#endif // SORT_KERNELS_5_CUH