#include <iostream>
#include <ctime>
#include <vector>
#include <chrono>
#include <iomanip>
#include <random>
#include <algorithm>
#include <thrust/scan.h>
#include <thrust/device_ptr.h>
#include "sort_kernels_4.cuh"
#define M 31

static void print_segment(
    const char* title,
    const unsigned int* cpu,
    const unsigned int* gpu,
    int start,
    int count
){
    printf("  %s\n", title);
    printf("    idx        CPU            GPU\n");
    printf("    -----------------------------------\n");
    for(int i = 0; i < count; ++i){
        int idx = start + i;
        printf("    [%8d]  %10u  %10u\n",
               idx, cpu[idx], gpu[idx]);
    }
    printf("\n");
}

bool verify(
    const unsigned int* cpu_res,
    const unsigned int* gpu_res,
    int n,
    const std::string& kernel_name
){
    for(int i = 0; i < n; ++i){
        if(cpu_res[i] != gpu_res[i]){
            printf("[\033[31mFAIL\033[0m] %s\n", kernel_name.c_str());
            printf("    First mismatch at index %d: CPU=%u, GPU=%u\n\n",
                   i, cpu_res[i], gpu_res[i]);

            // ===== 前 16 =====
            print_segment(
                "First 16 elements",
                cpu_res, gpu_res,
                0, 16
            );

            // ===== 中间 16 =====
            int mid_start = n / 2 - 8;
            print_segment(
                "Middle 16 elements",
                cpu_res, gpu_res,
                mid_start, 16
            );

            // ===== 末尾 16 =====
            int tail_start = n - 16;
            print_segment(
                "Last 16 elements",
                cpu_res, gpu_res,
                tail_start, 16
            );

            return false;
        }
    }

    printf("[\033[32mPASS\033[0m] %s (N=%d)\n",
           kernel_name.c_str(), n);
    return true;
}


int main() {

    auto now = std::chrono::system_clock::now();
    std::time_t now_time = std::chrono::system_clock::to_time_t(now);
    std::cout << "Verify started at: " 
            << std::put_time(std::localtime(&now_time), "%Y-%m-%d %H:%M:%S") 
            << std::endl;
        
    const int N = 1 << 28;

    std::vector<unsigned int> h_input(N);
    std::vector<unsigned int> h_gpu_output(N);

    std::mt19937 gen(1234);
    std::uniform_int_distribution<unsigned int> dis(0, 1u << M);
    for(int i=0;i<N;++i){
        h_input[i] = dis(gen);
    }
    
    int block_size = 1024;
    int grid_size = (N+block_size-1)/block_size;

    unsigned int *d_input, *d_output, *d_block_counts, *d_block_counts_scanned;
    cudaMalloc(&d_input, N*sizeof(unsigned int));
    cudaMalloc(&d_block_counts, 8*grid_size*sizeof(unsigned int));
    cudaMalloc(&d_block_counts_scanned, 8*grid_size*sizeof(unsigned int));
    cudaMalloc(&d_output, N*sizeof(unsigned int));
    // h_input 会使用标准库函数在原地做排序，排序之前先复制到device端
    cudaMemcpy(d_input, h_input.data(), N*sizeof(unsigned int), cudaMemcpyHostToDevice);

    printf("Computing CPU sort for comparison...\n");
    std::sort(h_input.begin(), h_input.end());
    printf("Done.\n\n");

    // Test 2
    printf("radix_sort_3.1\n\n");
    cudaMemset(d_block_counts, 0, 8*grid_size*sizeof(unsigned int));
    cudaMemset(d_block_counts_scanned, 0, 8*grid_size*sizeof(unsigned int));
    cudaMemset(d_output, 0, N*sizeof(unsigned int));
    
    unsigned int* d_in = d_input;
    unsigned int* d_out = d_output;
    for (int j=0;j<(M+1);j+=3){
        radix_sort_count_block<1024><<<grid_size,block_size>>>(d_in, d_block_counts, N, j);

        // 我们使用 thrust 的kernel来做scan
        thrust::device_ptr<unsigned int> d_block_counts_ptr(d_block_counts);
        thrust::device_ptr<unsigned int> d_block_counts_scanned_ptr(d_block_counts_scanned);
        thrust::exclusive_scan(d_block_counts_ptr, d_block_counts_ptr+8*grid_size, d_block_counts_scanned_ptr);

        radix_sort_coalesced_scatter<1024><<<grid_size,block_size>>>(d_in, d_out, d_block_counts_scanned, j);
        std::swap(d_in, d_out);
    }
    cudaDeviceSynchronize();
    cudaMemcpy(h_gpu_output.data(), d_in, N*sizeof(unsigned int), cudaMemcpyDeviceToHost);
    verify(h_input.data(), h_gpu_output.data(), N, "radix_sort_3.1");
    printf("-----------------------\n\n");

    cudaFree(d_input);
    cudaFree(d_block_counts);
    cudaFree(d_block_counts_scanned);
    cudaFree(d_output);

    auto now_end = std::chrono::system_clock::now();
    std::time_t end_time = std::chrono::system_clock::to_time_t(now_end);
    std::cout << "Verify Finished at: " 
              << std::put_time(std::localtime(&end_time), "%Y-%m-%d %H:%M:%S") 
              << std::endl;

    return 0;
}