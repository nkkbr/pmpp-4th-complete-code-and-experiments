#include <iostream>
#include <curand.h>
#include <chrono>
#include <ctime>
#include <iomanip>
#include <thrust/scan.h>
#include <thrust/device_ptr.h>
#include <curand.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <thrust/system/cuda/execution_policy.h>
#include "sort_kernels_6.coarsened.cuh"
#include <cstdlib>
#include <cstdio>
#define BENCHMARK_REPEAT 1000
#define M 31
#ifndef BS
#define BS 1024
#endif

#ifndef CO
#define CO 2
#endif

void generate_random_uniform(unsigned int* d_data, int N, unsigned long long seed = 1234) {
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_XORWOW);
    curandSetPseudoRandomGeneratorSeed(gen, seed);

    curandGenerate(gen, d_data, N);

    curandDestroyGenerator(gen);
}

void sort_kernels_6_coarsened(
    unsigned int *d_in, 
    unsigned int *d_out, 
    unsigned int *d_block_counts,
    unsigned int *d_block_counts_scanned,
    unsigned int grid_size,
    unsigned int block_size,
    double *time,
    cudaEvent_t start,
    cudaEvent_t stop
){
    float milliseconds = 0;

    thrust::device_ptr<unsigned int> d_block_counts_ptr(d_block_counts);
    thrust::device_ptr<unsigned int> d_block_counts_scanned_ptr(d_block_counts_scanned);

    for (int j=0;j<(M+1);j+=2){

        // 我们仅对 step1 和 step3 进行计时
        cudaEventRecord(start,0);
        // step1
        radix_sort_count_block<BS, CO><<<grid_size, block_size>>>(d_in, d_block_counts, j);

        cudaEventRecord(stop,0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&milliseconds, start, stop);
        *time += milliseconds;
        // step2
        // 我们使用 thrust 的kernel来做scan
        thrust::exclusive_scan(thrust::cuda::par.on(0), d_block_counts_ptr, d_block_counts_ptr+4*grid_size, d_block_counts_scanned_ptr);

        cudaEventRecord(start,0);
        // step3
        radix_sort_coalesced_scatter<BS, CO><<<grid_size, block_size>>>(d_in, d_out, d_block_counts_scanned, j);
        
        cudaEventRecord(stop,0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&milliseconds, start, stop);
        *time += milliseconds;
        std::swap(d_in, d_out);
    }
}

int main(int argc, char** argv){
    constexpr unsigned int N = 1 << 28;
    constexpr unsigned int grid_size = (N+(BS*CO)-1)/(BS*CO);

    auto now = std::chrono::system_clock::now();
    std::time_t now_time = std::chrono::system_clock::to_time_t(now);
    std::cout << "Benchmark started at: " 
            << std::put_time(std::localtime(&now_time), "%Y-%m-%d %H:%M:%S") 
            << std::endl;

    int dev=0;
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    printf("Currently using CUDA device %d: %s\n", dev, deviceProp.name);
    cudaSetDevice(dev);
    printf("Total Global Memory: %.2f GB\n", deviceProp.totalGlobalMem / (1024.0*1024.0*1024.0));
    printf("Benchmark Data Size: %u elements (%.2f GB per array)\n", N, (double)N * 4 / (1024.0*1024.0*1024.0));
    printf("------------------------------------------------------------------------------------------------------------\n\n");

    printf("radix_sort_coarsened\n\n");

    printf("BS = %d\n", BS);
    printf("CO = %d\n\n", CO);

    unsigned int *d_input, *d_output, *d_block_counts, *d_block_counts_scanned, *d_backup;
    cudaMalloc(&d_input, N*sizeof(unsigned int));
    cudaMalloc(&d_block_counts, 4*grid_size*sizeof(unsigned int));
    cudaMalloc(&d_block_counts_scanned, 4*grid_size*sizeof(unsigned int));
    cudaMalloc(&d_output, N*sizeof(unsigned int));
    cudaMalloc(&d_backup, N*sizeof(unsigned int));

    // 本测试中我们使用随机的生成
    generate_random_uniform(d_backup, N);

    unsigned int* d_in = d_input;
    unsigned int* d_out = d_output;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    double total_time = 0;
    // 1. warmup
    for(int i=0; i<10; ++i) {
        cudaMemcpy(d_in, d_backup, N*sizeof(unsigned int), cudaMemcpyDeviceToDevice); // 重置数据
        sort_kernels_6_coarsened(d_in,d_out,d_block_counts,d_block_counts_scanned,grid_size,BS,&total_time,start,stop);
    }
    cudaDeviceSynchronize();

    // 2. 正式 Benchmark
    total_time = 0;

    for(int i=0; i<BENCHMARK_REPEAT; ++i) {
        cudaMemcpy(d_in, d_backup, N*sizeof(unsigned int), cudaMemcpyDeviceToDevice);
        cudaMemset(d_block_counts, 0, 4*grid_size*sizeof(unsigned int));
        cudaMemset(d_block_counts_scanned, 0, 4*grid_size*sizeof(unsigned int));
        cudaMemset(d_output, 0, N*sizeof(unsigned int));

        sort_kernels_6_coarsened(d_in,d_out,d_block_counts,d_block_counts_scanned,grid_size,BS,&total_time,start,stop);

    }

    printf("Average Time: %f ms\n\n", total_time / BENCHMARK_REPEAT);

    cudaFree(d_input);
    cudaFree(d_block_counts);
    cudaFree(d_block_counts_scanned);
    cudaFree(d_output);
    cudaFree(d_backup);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    auto now_end = std::chrono::system_clock::now();
    std::time_t end_time = std::chrono::system_clock::to_time_t(now_end);
    std::cout << "Benchmark finished at: " 
              << std::put_time(std::localtime(&end_time), "%Y-%m-%d %H:%M:%S") 
              << std::endl;

    return 0;
}