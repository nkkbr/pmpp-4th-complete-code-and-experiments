#ifndef SORT_KERNELS_4_CUH
#define SORT_KERNELS_4_CUH

#include <cuda_runtime.h>

////////////////////////////////////////////////////////
/////// 13.5 Choice of radix value 
/////// radix = 8
////////////////////////////////////////////////////////

// .step 1
template <unsigned int SECTION_SIZE>
__global__ void radix_sort_count_block(
    const unsigned int* __restrict__ input,
    unsigned int* __restrict__ block_counts, // 长度是8*GridDim.x，前 GridDim.x 个数字是各个block中0的个数，第二个 GridDim.x 个数字是各个block中1的个数，然后是2,3,4,5,6,7的个数。
    unsigned int N,
    unsigned int iter // iter = 0,3,6,...
){
    __shared__ unsigned int s_block_ones;
    __shared__ unsigned int s_block_twos;
    __shared__ unsigned int s_block_threes;
    __shared__ unsigned int s_block_fours;
    __shared__ unsigned int s_block_fives;
    __shared__ unsigned int s_block_sixs;
    __shared__ unsigned int s_block_sevens;
    if (threadIdx.x == 0){
        s_block_ones = 0;
        s_block_twos = 0;
        s_block_threes = 0;
        s_block_fours = 0;
        s_block_fives = 0;
        s_block_sixs = 0;
        s_block_sevens = 0;
    }
    __syncthreads();

    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;

    unsigned int is_one = 0;
    unsigned int is_two = 0;
    unsigned int is_three = 0;
    unsigned int is_four = 0;
    unsigned int is_five = 0;
    unsigned int is_six = 0;
    unsigned int is_seven = 0;

    if (i<N){
        is_one = (((input[i] >> iter) & 7) == 1) ? 1:0;
        is_two = (((input[i] >> iter) & 7) == 2) ? 1:0;
        is_three = (((input[i] >> iter) & 7) == 3) ? 1:0;
        is_four = (((input[i] >> iter) & 7) == 4) ? 1:0;
        is_five = (((input[i] >> iter) & 7) == 5) ? 1:0;
        is_six = (((input[i] >> iter) & 7) == 6) ? 1:0;
        is_seven = (((input[i] >> iter) & 7) == 7) ? 1:0;
    }

    unsigned int mask = 0xffffffff;

    #pragma unroll
    for(int offset = 16; offset > 0; offset /= 2){
        is_one += __shfl_down_sync(mask, is_one, offset);
        is_two += __shfl_down_sync(mask, is_two, offset);
        is_three += __shfl_down_sync(mask, is_three, offset);
        is_four += __shfl_down_sync(mask, is_four, offset);
        is_five += __shfl_down_sync(mask, is_five, offset);
        is_six += __shfl_down_sync(mask, is_six, offset);
        is_seven += __shfl_down_sync(mask, is_seven, offset);
    }

    if ((threadIdx.x % 32) == 0) {
        atomicAdd(&s_block_ones, is_one);
        atomicAdd(&s_block_twos, is_two);
        atomicAdd(&s_block_threes, is_three);
        atomicAdd(&s_block_fours, is_four);
        atomicAdd(&s_block_fives, is_five);
        atomicAdd(&s_block_sixs, is_six);
        atomicAdd(&s_block_sevens, is_seven);
    }

    __syncthreads();

    if(threadIdx.x==0){
        unsigned int total_ones = s_block_ones;
        unsigned int total_twos = s_block_twos;
        unsigned int total_threes = s_block_threes;
        unsigned int total_fours = s_block_fours;
        unsigned int total_fives = s_block_fives;
        unsigned int total_sixs = s_block_sixs;
        unsigned int total_sevens = s_block_sevens;

        block_counts[blockIdx.x] = SECTION_SIZE - total_ones - total_twos - total_threes - total_fours - total_fives - total_sixs - total_sevens;
        block_counts[gridDim.x+blockIdx.x] = total_ones;
        block_counts[gridDim.x*2+blockIdx.x] = total_twos;
        block_counts[gridDim.x*3+blockIdx.x] = total_threes;
        block_counts[gridDim.x*4+blockIdx.x] = total_fours;
        block_counts[gridDim.x*5+blockIdx.x] = total_fives;
        block_counts[gridDim.x*6+blockIdx.x] = total_sixs;
        block_counts[gridDim.x*7+blockIdx.x] = total_sevens;
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

#endif // SORT_KERNELS_4_CUH