#ifndef SORT_KERNELS_6_CUH
#define SORT_KERNELS_6_CUH

#include <cuda_runtime.h>

////////////////////////////////////////////////////////
/////// 13.5 Choice of radix value (optimized version 2)
/////// radix = 4
////////////////////////////////////////////////////////

// .step 1
// 我们保证，BLOCK_SIZE 是 32的倍数
// 我们也保证，数据的总量 N 是 BLOCK_SIZE * COARSEN 的倍数。
// 实际测试的时候，这些数字，都是2的幂
// 我们保证 BLOCK_SIZE = blockDim.x， 为了明确这一点，我们在代码中不使用 blockDim.x，仅使用 BLOCK_SIZE 。
template <unsigned int BLOCK_SIZE, unsigned int COARSEN>
__global__ void radix_sort_count_block(
    const unsigned int* __restrict__ input,
    unsigned int* __restrict__ block_counts, // 长度是4*gridDim.x，前 gridDim.x 个数字是各个block所管辖的数据中0的个数，第二个 gridDim.x 个数字是各个block所管辖的数据中1的个数，然后是2和3的个数。
    unsigned int iter // iter = 0,2,4,...
){
    constexpr unsigned int nElem = BLOCK_SIZE * COARSEN;
    const unsigned int base = (COARSEN*BLOCK_SIZE)*blockIdx.x;
    const unsigned int tid = threadIdx.x;
    constexpr unsigned int warpNum = BLOCK_SIZE / 32;
    // 私有槽位，每个warp一个
    __shared__ unsigned int warp_c1[warpNum];
    __shared__ unsigned int warp_c2[warpNum];
    __shared__ unsigned int warp_c3[warpNum];

    const unsigned int lane = tid & 31;
    const unsigned int warpId = tid >> 5;

    unsigned int a=0,b=0,c=0;

    for(unsigned int j = 0;j<COARSEN; ++j){

        unsigned int digit = (input[base+tid+j*BLOCK_SIZE] >> iter) & 3;
        unsigned int m1 = __ballot_sync(0xffffffff, digit == 1);
        unsigned int m2 = __ballot_sync(0xffffffff, digit == 2);
        unsigned int m3 = __ballot_sync(0xffffffff, digit == 3);

        if (lane == 0) {
            a += __popc(m1);
            b += __popc(m2);
            c += __popc(m3);
        }
    }

    if (lane == 0) {
        warp_c1[warpId] = a;
        warp_c2[warpId] = b;
        warp_c3[warpId] = c;
    }
    __syncthreads();

    if (tid == 0){
        unsigned int t1=0, t2=0, t3=0;

        #pragma unroll
        for(unsigned int w=0; w<warpNum; ++w){
            t1 += warp_c1[w];
            t2 += warp_c2[w];
            t3 += warp_c3[w];
        }

        block_counts[blockIdx.x] = nElem - t1 - t2 - t3;
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
template <unsigned int BLOCK_SIZE, unsigned int COARSEN>
__global__ void radix_sort_coalesced_scatter(
    const unsigned int* __restrict__ input,
    unsigned int* __restrict__ output,
    const unsigned int* __restrict__ block_offsets,
    unsigned int iter
){
    constexpr unsigned int WARP_SIZE = 32;
    constexpr unsigned int NUM_WARPS = BLOCK_SIZE * COARSEN / WARP_SIZE;
    constexpr unsigned int PAD = BLOCK_SIZE * COARSEN / WARP_SIZE;
    constexpr unsigned int SDATA_SIZE = BLOCK_SIZE * COARSEN + PAD; // 每个warp都配1个padding

    // 这个用于存储在每个block里，排好了序的结果。这样，写入global memory的时候，它就是coalesced的写入了
    __shared__ unsigned int s_data[SDATA_SIZE];

    // 每个 warp 中，尾数为0，1，2，3的数字的总和
    __shared__ unsigned int s_warp_counts[NUM_WARPS][4];

    // 对于每个 warp，它之前的（不包含自己）warp 的尾数为0，1，2，3的数字的总和
    __shared__ unsigned int s_warp_prefix[NUM_WARPS][4];

    // 当前block中，在排列好之后，0，1，2，3 尾数的数字（我们称为桶），起始的编号是多少
    __shared__ unsigned int s_bucket_base[4];

    // rank_in_warp 是当前warp中，每个线程都独有的一个，它计算，在当前的warp中，自己前面的线程（不包含自己）有几个符合条件的
    // 因为我们这个是粗化的版本，所以这里的 rank_in_warp 是个数组，不在寄存器上，会造成性能的下降
    unsigned int rank_in_warp[COARSEN];

    for(unsigned int j=0;j<COARSEN;++j){

        const unsigned int base = (COARSEN*BLOCK_SIZE)*blockIdx.x;
        const unsigned int tid_curr = threadIdx.x + BLOCK_SIZE*j;

        const unsigned int lane = tid_curr & 31;
        const unsigned int warpId = tid_curr >> 5;
        const unsigned int lane_lt_mask = (1u << lane) - 1u;

        const unsigned int key = input[base+tid_curr];
        const unsigned int digit = (key >> iter) & 3u;

        const unsigned int FULL = 0xffffffffu;

        // 返回一个数字，32位的，每一位都表明，当前的thread上，digit == 0u/1u/2u/3u 是不是成立。其中digit是当前thread独有的，在寄存器上。 对于每一位，是的话返回1，不是的话返回0。
        const unsigned int m0 = __ballot_sync(FULL, digit == 0u);
        const unsigned int m1 = __ballot_sync(FULL, digit == 1u);
        const unsigned int m2 = __ballot_sync(FULL, digit == 2u);
        const unsigned int m3 = __ballot_sync(FULL, digit == 3u);

        // 求 32 个位的和，统计有几个符合要求。
        if (lane == 0) {
            s_warp_counts[warpId][0] = __popc(m0);
            s_warp_counts[warpId][1] = __popc(m1);
            s_warp_counts[warpId][2] = __popc(m2);
            s_warp_counts[warpId][3] = __popc(m3);
        }

        if (digit == 0u) rank_in_warp[j] = __popc(m0 & lane_lt_mask);
        else if (digit == 1u) rank_in_warp[j] = __popc(m1 & lane_lt_mask);
        else if (digit == 2u) rank_in_warp[j] = __popc(m2 & lane_lt_mask);
        else  rank_in_warp[j] = __popc(m3 & lane_lt_mask);
    }

    __syncthreads();

    if (threadIdx.x ==0) {
        unsigned int run0=0, run1=0, run2=0, run3=0;

        // 这里是很naive的，串行的加法，对于这个粗化的版本，这里的加法，次数其实很多。
        #pragma unroll
        for(int w=0; w<NUM_WARPS; ++w){
            s_warp_prefix[w][0] = run0;
            s_warp_prefix[w][1] = run1;
            s_warp_prefix[w][2] = run2;
            s_warp_prefix[w][3] = run3;
            run0 += s_warp_counts[w][0];
            run1 += s_warp_counts[w][1];
            run2 += s_warp_counts[w][2];
            run3 += s_warp_counts[w][3];
        }

        // 没事的，别怕，编译器会帮我们复用寄存器，这样的代码不会造成寄存器压力的。
        const unsigned int num0 = run0;
        const unsigned int num1 = run1;
        const unsigned int num2 = run2;

        // 这些变量，都是block级别的，而不是warp级别的
        s_bucket_base[0] = 0u;
        s_bucket_base[1] = num0;
        s_bucket_base[2] = num0 + num1;
        s_bucket_base[3] = num0 + num1 + num2;
    }

    __syncthreads();

    for(unsigned int j=0;j<COARSEN;++j){

        const unsigned int base = (COARSEN*BLOCK_SIZE)*blockIdx.x;
        const unsigned int tid_curr = threadIdx.x + BLOCK_SIZE*j;
        const unsigned int warpId = tid_curr >> 5;
        const unsigned int key = input[base+tid_curr];
        const unsigned int digit = (key >> iter) & 3u;

        const unsigned int base_d = s_bucket_base[digit];
        const unsigned int pref_d = s_warp_prefix[warpId][digit];
        const unsigned int dst = base_d + pref_d + rank_in_warp[j]; 

        const unsigned int dst_pad = dst + (dst >> 5); // 按照 dst 所处的位置添加pad，例如，如果原本在[0,32),那么要加0，原本在[32,64)，要加1。相当于每隔32个数字，都在后面增加了一个不用的padding。

        s_data[dst_pad] = key;
    }

    __syncthreads();

    for(unsigned int j=0;j<COARSEN;++j){
        const unsigned int tid_curr = threadIdx.x + BLOCK_SIZE*j;

        const unsigned int b1 = s_bucket_base[1];
        const unsigned int b2 = s_bucket_base[2];
        const unsigned int b3 = s_bucket_base[3];

        const unsigned int bin = (unsigned int)(tid_curr>=b1) + (unsigned int)(tid_curr>=b2) + (unsigned int)(tid_curr>=b3);

        const unsigned int bin_base = s_bucket_base[bin];
        const unsigned out_idx = block_offsets[blockIdx.x + bin*gridDim.x] + (tid_curr-bin_base);

        // 下面两行，终于把上面的努力都转换成了实际的性能上的提升。它既做到了 coalesced 的写入，又试图避免了 bank confilct。
        const unsigned int tid_curr_pad = tid_curr + (tid_curr >> 5);
        output[out_idx] = s_data[tid_curr_pad];
    }
}

#endif // SORT_KERNELS_6_CUH