#ifndef SORT_KERNELS_6_CUH
#define SORT_KERNELS_6_CUH

#include <cuda_runtime.h>

////////////////////////////////////////////////////////
/////// 13.5 Choice of radix value (optimized version 2)
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
    constexpr unsigned int WARP_SIZE = 32;
    constexpr unsigned int NUM_WARPS = SECTION_SIZE / WARP_SIZE;
    constexpr unsigned int PAD = SECTION_SIZE / WARP_SIZE;
    constexpr unsigned int SDATA_SIZE = SECTION_SIZE + PAD; // 每个warp都配1个padding

    // 这个用于存储在每个block里，排好了序的结果。这样，写入global memory的时候，它就是coalesced的写入了
    __shared__ unsigned int s_data[SDATA_SIZE];

    // 每个 warp 中，尾数为0，1，2，3的数字的总和
    __shared__ unsigned int s_warp_counts[NUM_WARPS][4];

    // 对于每个 warp，它之前的（不包含自己）warp 的尾数为0，1，2，3的数字的总和
    __shared__ unsigned int s_warp_prefix[NUM_WARPS][4];

    // 当前block中，在排列好之后，0，1，2，3 尾数的数字（我们称为桶），起始的编号是多少
    __shared__ unsigned int s_bucket_base[4];

    const unsigned int tid = threadIdx.x;
    const unsigned int idx = blockDim.x*blockIdx.x+threadIdx.x;

    const unsigned int lane = tid & 31;
    const unsigned int warpId = tid >> 5;
    const unsigned int lane_lt_mask = (1u << lane) - 1u;

    const unsigned int key = input[idx];
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

    // rank_in_warp 是当前warp中，每个线程都独有的一个，它计算，在当前的warp中，自己前面的线程（不包含自己）有几个符合条件的
    unsigned int rank_in_warp;

    if (digit == 0u) rank_in_warp = __popc(m0 & lane_lt_mask);
    else if (digit == 1u) rank_in_warp = __popc(m1 & lane_lt_mask);
    else if (digit == 2u) rank_in_warp = __popc(m2 & lane_lt_mask);
    else  rank_in_warp = __popc(m3 & lane_lt_mask);

    __syncthreads();

    if (tid ==0) {
        unsigned int run0=0, run1=0, run2=0, run3=0;

        // 这里是很naive的，串行的加法，而且这里的加法，次数其实很多。
        // 例如，SECTION_SIZE是1024的时候，要进行32次这个循环。
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

    const unsigned int base_d = s_bucket_base[digit];
    const unsigned int pref_d = s_warp_prefix[warpId][digit];
    const unsigned int dst = base_d + pref_d + rank_in_warp; 

    const unsigned int dst_pad = dst + (dst >> 5); // 按照 dst 所处的位置添加pad，例如，如果原本在[0,32),那么要加0，原本在[32,64)，要加1。相当于每隔32个数字，都在后面增加了一个不用的padding。

    s_data[dst_pad] = key;

    __syncthreads();

    const unsigned int b1 = s_bucket_base[1];
    const unsigned int b2 = s_bucket_base[2];
    const unsigned int b3 = s_bucket_base[3];

    const unsigned int bin = (unsigned int)(tid>=b1) + (unsigned int)(tid>=b2) + (unsigned int)(tid>=b3);

    const unsigned int bin_base = s_bucket_base[bin];
    const unsigned out_idx = block_offsets[blockIdx.x + bin*gridDim.x] + (tid-bin_base);

    // 下面两行，终于把上面的努力都转换成了实际的性能上的提升。它既做到了 coalesced 的写入，又试图避免了 bank confilct。
    const unsigned int tid_pad = tid + (tid >> 5);
    output[out_idx] = s_data[tid_pad];

}

#endif // SORT_KERNELS_6_CUH