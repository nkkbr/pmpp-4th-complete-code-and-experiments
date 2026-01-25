#ifndef SORT_KERNELS_1_CUH
#define SORT_KERNELS_1_CUH

#include <cuda_runtime.h>

////////////////////////////////////////////////////////
/////// 13.3 Parallel radix sort
////////////////////////////////////////////////////////

// .step 1
__global__ void radix_sort_extract(
    const unsigned int* __restrict__ input,
    unsigned int* __restrict__ bits,
    unsigned int N,
    unsigned int iter
){
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int key, bit;
    if(i<N){
        key = input[i];
        bit = (key >> iter) & 1;
        bits[i] = bit;
    }
}


// .step 2
// 是 exclusive scan，并且是一个 block 级别的 scan
// 因为需要进行全局的，而非block级别的scan，所以我们实际上不使用这个kernel
// 我们使用 thrust::exclusive_scan
template <unsigned int SECTION_SIZE>
__global__ void radix_sort_exclusive_scan(
    const unsigned int* __restrict__ bits,
    unsigned int* __restrict__ ones_before,
    unsigned int N
){
    __shared__ unsigned int XY[SECTION_SIZE];
    unsigned int i = blockDim.x*blockIdx.x + threadIdx.x;

    unsigned int val = 0;
    if (i<N){
        val = bits[i];
        XY[threadIdx.x] = val;
    } else {
        XY[threadIdx.x] = 0;
    }

    for(unsigned int stride = 1; stride<blockDim.x; stride *=2){
        __syncthreads();
        unsigned int temp;
        if(threadIdx.x >= stride){
            temp = XY[threadIdx.x]+ XY[threadIdx.x-stride];
        }
        __syncthreads();
        if(threadIdx.x >= stride){
            XY[threadIdx.x] = temp;
        }
    }
    if(i<N){
        ones_before[i] = XY[threadIdx.x]-val;
    }
}


// .step 3
__global__ void radix_sort_scatter(
    const unsigned int* __restrict__ input,
    unsigned int* __restrict__ output,
    const unsigned int* __restrict__ ones_before,
    unsigned int N,
    unsigned int iter
){
    unsigned int i = blockDim.x*blockIdx.x + threadIdx.x;
    unsigned int key, bit;

    unsigned int lastBit = (input[N-1] >> iter ) & 1;
    unsigned int numOnesTotal = ones_before[N-1] + lastBit; // 因为 exclusive scan 没有包含最后一个元素的信息，我们还是要手动加上
    
    if (i<N){
        key = input[i];
        bit = (key >> iter) &1;

        unsigned int numOnesBefore = ones_before[i];
        unsigned int dst = (bit == 0) ? (i-numOnesBefore):(N-numOnesTotal+numOnesBefore);
        output[dst] = key;
    }
}


#endif // SORT_KERNELS_1_CUH