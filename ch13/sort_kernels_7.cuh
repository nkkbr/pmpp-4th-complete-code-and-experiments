#ifndef SORT_KERNELS_7_CUH
#define SORT_KERNELS_7_CUH

#include <cuda_runtime.h>

////////////////////////////////////////////////////////
/////// 13.5 Choice of radix value (optimized version 2)
/////// radix = 8
////////////////////////////////////////////////////////

// .step 1
template <unsigned int SECTION_SIZE>
__global__ void radix_sort_count_block(
    // 我们不传入数据总量 N，我们确保它是blockDim.x的倍数，并确保gridDim.x正好覆盖所有的数据。
    const unsigned int* __restrict__ input,
    unsigned int* __restrict__ block_counts, // 长度是8*GridDim.x，前 GridDim.x 个数字是各个block中0的个数，第二个 GridDim.x 个数字是各个block中1的个数，然后是2,3,4,5,6,7的个数。
    unsigned int iter // iter = 0,3,6,...
){
    const unsigned int i = blockDim.x*blockIdx.x + threadIdx.x;

    unsigned int digit = (input[i] >> iter) & 7;
    unsigned int m1 = __ballot_sync(0xffffffff, digit == 1);
    unsigned int m2 = __ballot_sync(0xffffffff, digit == 2);
    unsigned int m3 = __ballot_sync(0xffffffff, digit == 3);
    unsigned int m4 = __ballot_sync(0xffffffff, digit == 4);
    unsigned int m5 = __ballot_sync(0xffffffff, digit == 5);
    unsigned int m6 = __ballot_sync(0xffffffff, digit == 6);
    unsigned int m7 = __ballot_sync(0xffffffff, digit == 7);

    unsigned int c1 = __popc(m1);
    unsigned int c2 = __popc(m2);
    unsigned int c3 = __popc(m3);
    unsigned int c4 = __popc(m4);
    unsigned int c5 = __popc(m5);
    unsigned int c6 = __popc(m6);
    unsigned int c7 = __popc(m7);

    // 私有槽位，每个warp一个
    __shared__ unsigned int warp_c1[32];
    __shared__ unsigned int warp_c2[32];
    __shared__ unsigned int warp_c3[32];
    __shared__ unsigned int warp_c4[32];
    __shared__ unsigned int warp_c5[32];
    __shared__ unsigned int warp_c6[32];
    __shared__ unsigned int warp_c7[32];

    const unsigned int lane = threadIdx.x & 31;
    const unsigned int warpId = threadIdx.x >> 5;

    if (lane == 0) {
        warp_c1[warpId] = c1;
        warp_c2[warpId] = c2;
        warp_c3[warpId] = c3;
        warp_c4[warpId] = c4;
        warp_c5[warpId] = c5;
        warp_c6[warpId] = c6;
        warp_c7[warpId] = c7;
    }

    __syncthreads();

    if (threadIdx.x == 0){
        unsigned int t1=0, t2=0, t3=0, t4=0, t5=0, t6=0, t7=0;
        int numWarps = (blockDim.x + 31) >> 5;

        #pragma unroll
        for(unsigned int w=0; w<32; ++w){
            if (w < numWarps) {
                t1 += warp_c1[w];
                t2 += warp_c2[w];
                t3 += warp_c3[w];
                t4 += warp_c4[w];
                t5 += warp_c5[w];
                t6 += warp_c6[w];
                t7 += warp_c7[w];
            }
        }

        block_counts[blockIdx.x] = SECTION_SIZE - t1 - t2 - t3 - t4 - t5 - t6 - t7; // 需要保证 SECTION_SIZE == blockDim.x
        block_counts[gridDim.x+blockIdx.x] = t1;
        block_counts[gridDim.x*2+blockIdx.x] = t2;
        block_counts[gridDim.x*3+blockIdx.x] = t3;
        block_counts[gridDim.x*4+blockIdx.x] = t4;
        block_counts[gridDim.x*5+blockIdx.x] = t5;
        block_counts[gridDim.x*6+blockIdx.x] = t6;
        block_counts[gridDim.x*7+blockIdx.x] = t7;
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
    __shared__ unsigned int s_scan_4[SECTION_SIZE];
    __shared__ unsigned int s_scan_5[SECTION_SIZE];
    __shared__ unsigned int s_scan_6[SECTION_SIZE];
    __shared__ unsigned int s_scan_7[SECTION_SIZE];

    // 1. exclusive scan
    unsigned int i = threadIdx.x;
    unsigned int idx = blockDim.x*blockIdx.x + threadIdx.x;

    unsigned int key = input[idx];
    unsigned int bit = (key >> iter) & 7;

    s_scan_1[i] = (bit==1)?1:0;
    s_scan_2[i] = (bit==2)?1:0;
    s_scan_3[i] = (bit==3)?1:0;
    s_scan_4[i] = (bit==4)?1:0;
    s_scan_5[i] = (bit==5)?1:0;
    s_scan_6[i] = (bit==6)?1:0;
    s_scan_7[i] = (bit==7)?1:0;

    for(unsigned int stride = 1; stride<blockDim.x; stride *= 2){
        __syncthreads();
        unsigned int temp1,temp2,temp3,temp4,temp5,temp6,temp7;
        if(threadIdx.x >= stride){
            temp1 = s_scan_1[threadIdx.x] + s_scan_1[threadIdx.x-stride];
            temp2 = s_scan_2[threadIdx.x] + s_scan_2[threadIdx.x-stride];
            temp3 = s_scan_3[threadIdx.x] + s_scan_3[threadIdx.x-stride];
            temp4 = s_scan_4[threadIdx.x] + s_scan_4[threadIdx.x-stride];
            temp5 = s_scan_5[threadIdx.x] + s_scan_5[threadIdx.x-stride];
            temp6 = s_scan_6[threadIdx.x] + s_scan_6[threadIdx.x-stride];
            temp7 = s_scan_7[threadIdx.x] + s_scan_7[threadIdx.x-stride];
        }
        __syncthreads();
        if(threadIdx.x>=stride){
            s_scan_1[threadIdx.x] = temp1;
            s_scan_2[threadIdx.x] = temp2;
            s_scan_3[threadIdx.x] = temp3;
            s_scan_4[threadIdx.x] = temp4;
            s_scan_5[threadIdx.x] = temp5;
            s_scan_6[threadIdx.x] = temp6;
            s_scan_7[threadIdx.x] = temp7;
        }
    }

    __syncthreads();
    unsigned int numOnesTotal = s_scan_1[blockDim.x-1];
    unsigned int numTwosTotal = s_scan_2[blockDim.x-1];
    unsigned int numThreesTotal = s_scan_3[blockDim.x-1];
    unsigned int numFoursTotal = s_scan_4[blockDim.x-1];
    unsigned int numFivesTotal = s_scan_5[blockDim.x-1];
    unsigned int numSixsTotal = s_scan_6[blockDim.x-1];
    unsigned int numSevensTotal = s_scan_7[blockDim.x-1];

    __syncthreads();
    // 因为我们要做的是 exclusive scan
    s_scan_1[threadIdx.x] -= (bit==1)?1:0; 
    s_scan_2[threadIdx.x] -= (bit==2)?1:0;
    s_scan_3[threadIdx.x] -= (bit==3)?1:0;
    s_scan_4[threadIdx.x] -= (bit==4)?1:0; 
    s_scan_5[threadIdx.x] -= (bit==5)?1:0;
    s_scan_6[threadIdx.x] -= (bit==6)?1:0;
    s_scan_7[threadIdx.x] -= (bit==7)?1:0;

    // 2. 计算出每一个元素，在block内部（即shared memory上）移动到哪个位置
    __syncthreads();

    unsigned int numOnesBefore = s_scan_1[i];
    unsigned int numTwosBefore = s_scan_2[i];
    unsigned int numThreesBefore = s_scan_3[i];
    unsigned int numFoursBefore = s_scan_4[i];
    unsigned int numFivesBefore = s_scan_5[i];
    unsigned int numSixsBefore = s_scan_6[i];
    unsigned int numSevensBefore = s_scan_7[i];
    unsigned int dst;

    switch (bit) {
    case 0:
        dst = i- numOnesBefore - numTwosBefore - numThreesBefore - numFoursBefore - numFivesBefore - numSixsBefore - numSevensBefore;
        break;
    case 1:
        dst = SECTION_SIZE - numOnesTotal - numTwosTotal - numThreesTotal - numFoursTotal - numFivesTotal -numSixsTotal - numSevensTotal + numOnesBefore;
        break;
    case 2:
        dst = SECTION_SIZE - numTwosTotal - numThreesTotal - numFoursTotal - numFivesTotal -numSixsTotal - numSevensTotal + numTwosBefore;
        break;
    case 3:
        dst = SECTION_SIZE - numThreesTotal - numFoursTotal - numFivesTotal -numSixsTotal - numSevensTotal + numThreesBefore;
        break; 
    case 4:
        dst = SECTION_SIZE - numFoursTotal - numFivesTotal -numSixsTotal - numSevensTotal + numFoursBefore;
        break; 
    case 5:
        dst = SECTION_SIZE - numFivesTotal -numSixsTotal - numSevensTotal + numFivesBefore;
        break; 
    case 6:
        dst = SECTION_SIZE - numSixsTotal - numSevensTotal + numSixsBefore;
        break; 
    case 7:
        dst = SECTION_SIZE - numSevensTotal + numSevensBefore;
        break; 
    }

    s_data[dst] = key;
    __syncthreads(); 

    unsigned int total_zeros = blockDim.x - numOnesTotal - numTwosTotal - numThreesTotal - numFoursTotal - numFivesTotal - numSixsTotal - numSevensTotal;
    unsigned int total_zeros_ones = blockDim.x - numTwosTotal - numThreesTotal - numFoursTotal - numFivesTotal - numSixsTotal - numSevensTotal;
    unsigned int total_zeros_to_twos = blockDim.x - numThreesTotal - numFoursTotal - numFivesTotal - numSixsTotal - numSevensTotal;
    unsigned int total_zeros_to_threes = blockDim.x - numFoursTotal - numFivesTotal - numSixsTotal - numSevensTotal;
    unsigned int total_zeros_to_fours = blockDim.x - numFivesTotal - numSixsTotal - numSevensTotal; 
    unsigned int total_zeros_to_fives = blockDim.x - numSixsTotal - numSevensTotal; 
    unsigned int total_zeros_to_sixs = blockDim.x - numSevensTotal; 
    if (i<total_zeros){
        output[block_offsets[blockIdx.x]+i] = s_data[i];
    } 
    else  if (i<total_zeros_ones){
        output[block_offsets[blockIdx.x+gridDim.x]+i-total_zeros] = s_data[i];
    }
    else  if (i<total_zeros_to_twos){
        output[block_offsets[blockIdx.x+2*gridDim.x]+i-total_zeros_ones] = s_data[i];
    }
    else if (i<total_zeros_to_threes){
        output[block_offsets[blockIdx.x+3*gridDim.x]+i-total_zeros_to_twos] = s_data[i];
    }
    else  if (i<total_zeros_to_fours){
        output[block_offsets[blockIdx.x+4*gridDim.x]+i-total_zeros_to_threes] = s_data[i];
    }
    else if (i<total_zeros_to_fives){
        output[block_offsets[blockIdx.x+5*gridDim.x]+i-total_zeros_to_fours] = s_data[i];
    }
    else  if (i<total_zeros_to_sixs){
        output[block_offsets[blockIdx.x+6*gridDim.x]+i-total_zeros_to_fives] = s_data[i];
    }
    else {
        output[block_offsets[blockIdx.x+7*gridDim.x]+i-total_zeros_to_sixs] = s_data[i];
    }

}

#endif // SORT_KERNELS_7_CUH