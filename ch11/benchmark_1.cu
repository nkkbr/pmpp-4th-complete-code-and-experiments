#include <iostream>
#include <curand.h>
#include <chrono>
#include <ctime>
#include <iomanip>
#include "scan_kernels.cuh"
#define BENCHMARK_REPEAT 1000

// 这份文件比较`scan_kernels.cuh`中的12个kernel之间的性能
// 要完全客观地比较性能，比较难。难在任务的设计上。在本次测试中，我们将总的任务设定为很大的一个float的数组，在BLOCK_SIZE固定为1024的情况下，每个kernel去计算每个block所对应负责的区域的前缀和。
// 每个kernel计算出来的，不是全局的前缀和，是每个block所对应负责的区域的前缀和。
// 但每个block所对应负责的区域的大小，又不是一样的。这就造成了，其实有的kernel的实际工作量，多了一点。但这个多，又很难量化
// 所以，我们这个测试，并不是一个很严格的benchmarking。

const unsigned int TOTAL_N = (1ULL << 28)*3*5;
const int BLOCK_SIZE = 1024;


void initGaussian(float* d_input, size_t n, float mean, float stddev) {
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);
    curandGenerateNormal(gen, d_input, n, mean, stddev);
    curandDestroyGenerator(gen);
}

int main(){

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
    printf("Benchmark Data Size: %u elements (%.2f GB per array)\n", TOTAL_N, (double)TOTAL_N * 4 / (1024.0*1024.0*1024.0));
    printf("------------------------------------------------------------------------------------------------------------\n");

    float *d_input, *d_output;
    cudaMalloc(&d_input, (size_t)TOTAL_N*sizeof(float));
    cudaMalloc(&d_output, (size_t)TOTAL_N*sizeof(float));

    initGaussian(d_input, TOTAL_N, 0.0f, 1.0f);

    int items = 1024;
    int grid_size = (TOTAL_N+items-1)/items;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float milliseconds;

    dim3 block(1);
    dim3 grid(1);

    double total_bytes = (double)TOTAL_N * sizeof(float) * 2; 
    double gb_per_sec;

    //////////////////////////////////////////////////////////////////////////////////////
    //     1. Kogge_Stone_scan_kernel
    //////////////////////////////////////////////////////////////////////////////////////
    items = 1024;
    grid_size = (TOTAL_N+items-1)/items;

    block.x = BLOCK_SIZE;
    grid.x = grid_size;
    printf("grid:  %5d, %5d, %5d\n", grid.x, grid.y, grid.z);
    printf("block: %5d, %5d, %5d\n", block.x, block.y, block.z);

    // 2.1 Warmup
    for (int i=0;i<10;i++){
        Kogge_Stone_scan_kernel<1024><<<grid, block>>>(d_input, d_output, TOTAL_N);
    }
    cudaDeviceSynchronize();
    printf("Warmup complete for Kogge_Stone_scan_kernel\n");

    // 2.2 Benchmarking
    cudaEventRecord(start);
    for (int i=0;i<BENCHMARK_REPEAT;i++){
        Kogge_Stone_scan_kernel<1024><<<grid, block>>>(d_input, d_output, TOTAL_N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Execution time for Kogge_Stone_scan_kernel: %f ms\n\n\n",milliseconds/BENCHMARK_REPEAT);
    gb_per_sec = total_bytes / ((milliseconds / BENCHMARK_REPEAT) / 1000.0) / 1e9;
    printf("Effective Bandwidth: %.2f GB/s\n", gb_per_sec);
    printf("------------------------------------------------------------------------------------------------------------\n");


    //////////////////////////////////////////////////////////////////////////////////////
    //     2. Kogge_Stone_scan_kernel_double_buffering
    //////////////////////////////////////////////////////////////////////////////////////
    items = 1024;
    grid_size = (TOTAL_N+items-1)/items;

    block.x = BLOCK_SIZE;
    grid.x = grid_size;
    printf("grid:  %5d, %5d, %5d\n", grid.x, grid.y, grid.z);
    printf("block: %5d, %5d, %5d\n", block.x, block.y, block.z);

    // 2.1 Warmup
    for (int i=0;i<10;i++){
        Kogge_Stone_scan_kernel_double_buffering<1024><<<grid, block>>>(d_input, d_output, TOTAL_N);
    }
    cudaDeviceSynchronize();
    printf("Warmup complete for Kogge_Stone_scan_kernel_double_buffering\n");

    // 2.2 Benchmarking
    cudaEventRecord(start);
    for (int i=0;i<BENCHMARK_REPEAT;i++){
        Kogge_Stone_scan_kernel_double_buffering<1024><<<grid, block>>>(d_input, d_output, TOTAL_N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Execution time for Kogge_Stone_scan_kernel_double_buffering: %f ms\n\n\n",milliseconds/BENCHMARK_REPEAT);
    gb_per_sec = total_bytes / ((milliseconds / BENCHMARK_REPEAT) / 1000.0) / 1e9;
    printf("Effective Bandwidth: %.2f GB/s\n", gb_per_sec);
    printf("------------------------------------------------------------------------------------------------------------\n");


    //////////////////////////////////////////////////////////////////////////////////////
    //     3. Kogge_Stone_scan_kernel_circular_buffer
    //////////////////////////////////////////////////////////////////////////////////////
    items = 1024;
    grid_size = (TOTAL_N+items-1)/items;

    block.x = BLOCK_SIZE;
    grid.x = grid_size;
    printf("grid:  %5d, %5d, %5d\n", grid.x, grid.y, grid.z);
    printf("block: %5d, %5d, %5d\n", block.x, block.y, block.z);

    // 2.1 Warmup
    for (int i=0;i<10;i++){
        Kogge_Stone_scan_kernel_circular_buffer<1024><<<grid, block>>>(d_input, d_output, TOTAL_N);
    }
    cudaDeviceSynchronize();
    printf("Warmup complete for Kogge_Stone_scan_kernel_circular_buffer\n");

    // 2.2 Benchmarking
    cudaEventRecord(start);
    for (int i=0;i<BENCHMARK_REPEAT;i++){
        Kogge_Stone_scan_kernel_circular_buffer<1024><<<grid, block>>>(d_input, d_output, TOTAL_N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Execution time for Kogge_Stone_scan_kernel_circular_buffer: %f ms\n\n\n",milliseconds/BENCHMARK_REPEAT);
    gb_per_sec = total_bytes / ((milliseconds / BENCHMARK_REPEAT) / 1000.0) / 1e9;
    printf("Effective Bandwidth: %.2f GB/s\n", gb_per_sec);
    printf("------------------------------------------------------------------------------------------------------------\n");


    //////////////////////////////////////////////////////////////////////////////////////
    //     3.5 Kogge_Stone_scan_kernel_circular_buffer_v2
    //////////////////////////////////////////////////////////////////////////////////////
    items = 1024;
    grid_size = (TOTAL_N+items-1)/items;

    block.x = BLOCK_SIZE;
    grid.x = grid_size;
    printf("grid:  %5d, %5d, %5d\n", grid.x, grid.y, grid.z);
    printf("block: %5d, %5d, %5d\n", block.x, block.y, block.z);

    // 2.1 Warmup
    for (int i=0;i<10;i++){
        Kogge_Stone_scan_kernel_circular_buffer_v2<1024><<<grid, block>>>(d_input, d_output, TOTAL_N);
    }
    cudaDeviceSynchronize();
    printf("Warmup complete for Kogge_Stone_scan_kernel_circular_buffer_v2\n");

    // 2.2 Benchmarking
    cudaEventRecord(start);
    for (int i=0;i<BENCHMARK_REPEAT;i++){
        Kogge_Stone_scan_kernel_circular_buffer_v2<1024><<<grid, block>>>(d_input, d_output, TOTAL_N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Execution time for Kogge_Stone_scan_kernel_circular_buffer_v2: %f ms\n\n\n",milliseconds/BENCHMARK_REPEAT);
    gb_per_sec = total_bytes / ((milliseconds / BENCHMARK_REPEAT) / 1000.0) / 1e9;
    printf("Effective Bandwidth: %.2f GB/s\n", gb_per_sec);
    printf("------------------------------------------------------------------------------------------------------------\n");
    

    //////////////////////////////////////////////////////////////////////////////////////
    //     4. Kogge_Stone_scan_kernel_shfl_up_sync_version
    //////////////////////////////////////////////////////////////////////////////////////
    items = 1024;
    grid_size = (TOTAL_N+items-1)/items;

    block.x = BLOCK_SIZE;
    grid.x = grid_size;
    printf("grid:  %5d, %5d, %5d\n", grid.x, grid.y, grid.z);
    printf("block: %5d, %5d, %5d\n", block.x, block.y, block.z);

    // 2.1 Warmup
    for (int i=0;i<10;i++){
        Kogge_Stone_scan_kernel_shfl_up_sync_version<<<grid, block>>>(d_input, d_output, TOTAL_N);
    }
    cudaDeviceSynchronize();
    printf("Warmup complete for Kogge_Stone_scan_kernel_shfl_up_sync_version\n");

    // 2.2 Benchmarking
    cudaEventRecord(start);
    for (int i=0;i<BENCHMARK_REPEAT;i++){
        Kogge_Stone_scan_kernel_shfl_up_sync_version<<<grid, block>>>(d_input, d_output, TOTAL_N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Execution time for Kogge_Stone_scan_kernel_shfl_up_sync_version: %f ms\n\n\n",milliseconds/BENCHMARK_REPEAT);
    gb_per_sec = total_bytes / ((milliseconds / BENCHMARK_REPEAT) / 1000.0) / 1e9;
    printf("Effective Bandwidth: %.2f GB/s\n", gb_per_sec);
    printf("------------------------------------------------------------------------------------------------------------\n");


    //////////////////////////////////////////////////////////////////////////////////////
    //     5. Kogge_Stone_scan_kernel_shfl_up_sync_version_coarsening_degree_4
    //////////////////////////////////////////////////////////////////////////////////////
    items = 1024*4;
    grid_size = (TOTAL_N+items-1)/items;

    block.x = BLOCK_SIZE;
    grid.x = grid_size;
    printf("grid:  %5d, %5d, %5d\n", grid.x, grid.y, grid.z);
    printf("block: %5d, %5d, %5d\n", block.x, block.y, block.z);

    // 2.1 Warmup
    for (int i=0;i<10;i++){
        Kogge_Stone_scan_kernel_shfl_up_sync_version_coarsening_degree_4<<<grid, block>>>(d_input, d_output, TOTAL_N);
    }
    cudaDeviceSynchronize();
    printf("Warmup complete for Kogge_Stone_scan_kernel_shfl_up_sync_version_coarsening_degree_4\n");

    // 2.2 Benchmarking
    cudaEventRecord(start);
    for (int i=0;i<BENCHMARK_REPEAT;i++){
        Kogge_Stone_scan_kernel_shfl_up_sync_version_coarsening_degree_4<<<grid, block>>>(d_input, d_output, TOTAL_N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Execution time for Kogge_Stone_scan_kernel_shfl_up_sync_version_coarsening_degree_4: %f ms\n\n\n",milliseconds/BENCHMARK_REPEAT);
    gb_per_sec = total_bytes / ((milliseconds / BENCHMARK_REPEAT) / 1000.0) / 1e9;
    printf("Effective Bandwidth: %.2f GB/s\n", gb_per_sec);
    printf("------------------------------------------------------------------------------------------------------------\n");


    //////////////////////////////////////////////////////////////////////////////////////
    //     6. Kogge_Stone_scan_kernel_shfl_up_sync_version_coarsening_degree_8
    //////////////////////////////////////////////////////////////////////////////////////
    items = 1024*8;
    grid_size = (TOTAL_N+items-1)/items;

    block.x = BLOCK_SIZE;
    grid.x = grid_size;
    printf("grid:  %5d, %5d, %5d\n", grid.x, grid.y, grid.z);
    printf("block: %5d, %5d, %5d\n", block.x, block.y, block.z);

    // 2.1 Warmup
    for (int i=0;i<10;i++){
        Kogge_Stone_scan_kernel_shfl_up_sync_version_coarsening_degree_8<<<grid, block>>>(d_input, d_output, TOTAL_N);
    }
    cudaDeviceSynchronize();
    printf("Warmup complete for Kogge_Stone_scan_kernel_shfl_up_sync_version_coarsening_degree_8\n");

    // 2.2 Benchmarking
    cudaEventRecord(start);
    for (int i=0;i<BENCHMARK_REPEAT;i++){
        Kogge_Stone_scan_kernel_shfl_up_sync_version_coarsening_degree_8<<<grid, block>>>(d_input, d_output, TOTAL_N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Execution time for Kogge_Stone_scan_kernel_shfl_up_sync_version_coarsening_degree_8: %f ms\n\n\n",milliseconds/BENCHMARK_REPEAT);
    gb_per_sec = total_bytes / ((milliseconds / BENCHMARK_REPEAT) / 1000.0) / 1e9;
    printf("Effective Bandwidth: %.2f GB/s\n", gb_per_sec);
    printf("------------------------------------------------------------------------------------------------------------\n");


    //////////////////////////////////////////////////////////////////////////////////////
    //     7. Kogge_Stone_scan_kernel_shfl_up_sync_version_coarsening_degree_12
    //////////////////////////////////////////////////////////////////////////////////////
    items = 1024*12;
    grid_size = (TOTAL_N+items-1)/items;

    block.x = BLOCK_SIZE;
    grid.x = grid_size;
    printf("grid:  %5d, %5d, %5d\n", grid.x, grid.y, grid.z);
    printf("block: %5d, %5d, %5d\n", block.x, block.y, block.z);

    // 2.1 Warmup
    for (int i=0;i<10;i++){
        Kogge_Stone_scan_kernel_shfl_up_sync_version_coarsening_degree_12<<<grid, block>>>(d_input, d_output, TOTAL_N);
    }
    cudaDeviceSynchronize();
    printf("Warmup complete for Kogge_Stone_scan_kernel_shfl_up_sync_version_coarsening_degree_12\n");

    // 2.2 Benchmarking
    cudaEventRecord(start);
    for (int i=0;i<BENCHMARK_REPEAT;i++){
        Kogge_Stone_scan_kernel_shfl_up_sync_version_coarsening_degree_12<<<grid, block>>>(d_input, d_output, TOTAL_N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Execution time for Kogge_Stone_scan_kernel_shfl_up_sync_version_coarsening_degree_12: %f ms\n\n\n",milliseconds/BENCHMARK_REPEAT);
    gb_per_sec = total_bytes / ((milliseconds / BENCHMARK_REPEAT) / 1000.0) / 1e9;
    printf("Effective Bandwidth: %.2f GB/s\n", gb_per_sec);
    printf("------------------------------------------------------------------------------------------------------------\n");


    //////////////////////////////////////////////////////////////////////////////////////
    //     8. Kogge_Stone_scan_kernel_shfl_up_sync_version_coarsening_degree_16
    //////////////////////////////////////////////////////////////////////////////////////
    items = 1024*16;
    grid_size = (TOTAL_N+items-1)/items;

    block.x = BLOCK_SIZE;
    grid.x = grid_size;
    printf("grid:  %5d, %5d, %5d\n", grid.x, grid.y, grid.z);
    printf("block: %5d, %5d, %5d\n", block.x, block.y, block.z);

    // 2.1 Warmup
    for (int i=0;i<10;i++){
        Kogge_Stone_scan_kernel_shfl_up_sync_version_coarsening_degree_16<<<grid, block>>>(d_input, d_output, TOTAL_N);
    }
    cudaDeviceSynchronize();
    printf("Warmup complete for Kogge_Stone_scan_kernel_shfl_up_sync_version_coarsening_degree_16\n");

    // 2.2 Benchmarking
    cudaEventRecord(start);
    for (int i=0;i<BENCHMARK_REPEAT;i++){
        Kogge_Stone_scan_kernel_shfl_up_sync_version_coarsening_degree_16<<<grid, block>>>(d_input, d_output, TOTAL_N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Execution time for Kogge_Stone_scan_kernel_shfl_up_sync_version_coarsening_degree_16: %f ms\n\n\n",milliseconds/BENCHMARK_REPEAT);
    gb_per_sec = total_bytes / ((milliseconds / BENCHMARK_REPEAT) / 1000.0) / 1e9;
    printf("Effective Bandwidth: %.2f GB/s\n", gb_per_sec);
    printf("------------------------------------------------------------------------------------------------------------\n");


    //////////////////////////////////////////////////////////////////////////////////////
    //     9. Kogge_Stone_scan_kernel_shfl_up_sync_version_coarsening_degree_20
    //////////////////////////////////////////////////////////////////////////////////////
    items = 1024*20;
    grid_size = (TOTAL_N+items-1)/items;

    block.x = BLOCK_SIZE;
    grid.x = grid_size;
    printf("grid:  %5d, %5d, %5d\n", grid.x, grid.y, grid.z);
    printf("block: %5d, %5d, %5d\n", block.x, block.y, block.z);

    // 2.1 Warmup
    for (int i=0;i<10;i++){
        Kogge_Stone_scan_kernel_shfl_up_sync_version_coarsening_degree_20<<<grid, block>>>(d_input, d_output, TOTAL_N);
    }
    cudaDeviceSynchronize();
    printf("Warmup complete for Kogge_Stone_scan_kernel_shfl_up_sync_version_coarsening_degree_20\n");

    // 2.2 Benchmarking
    cudaEventRecord(start);
    for (int i=0;i<BENCHMARK_REPEAT;i++){
        Kogge_Stone_scan_kernel_shfl_up_sync_version_coarsening_degree_20<<<grid, block>>>(d_input, d_output, TOTAL_N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Execution time for Kogge_Stone_scan_kernel_shfl_up_sync_version_coarsening_degree_20: %f ms\n\n\n",milliseconds/BENCHMARK_REPEAT);
    gb_per_sec = total_bytes / ((milliseconds / BENCHMARK_REPEAT) / 1000.0) / 1e9;
    printf("Effective Bandwidth: %.2f GB/s\n", gb_per_sec);
    printf("------------------------------------------------------------------------------------------------------------\n");


    //////////////////////////////////////////////////////////////////////////////////////
    //     10. Brent_Kung_scan_kernel
    //////////////////////////////////////////////////////////////////////////////////////
    items = 1024*2;
    grid_size = (TOTAL_N+items-1)/items;

    block.x = BLOCK_SIZE;
    grid.x = grid_size;
    printf("grid:  %5d, %5d, %5d\n", grid.x, grid.y, grid.z);
    printf("block: %5d, %5d, %5d\n", block.x, block.y, block.z);

    // 2.1 Warmup
    for (int i=0;i<10;i++){
        Brent_Kung_scan_kernel<2048><<<grid, block>>>(d_input, d_output, TOTAL_N);
    }
    cudaDeviceSynchronize();
    printf("Warmup complete for Brent_Kung_scan_kernel\n");

    // 2.2 Benchmarking
    cudaEventRecord(start);
    for (int i=0;i<BENCHMARK_REPEAT;i++){
        Brent_Kung_scan_kernel<2048><<<grid, block>>>(d_input, d_output, TOTAL_N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Execution time for Brent_Kung_scan_kernel: %f ms\n\n\n",milliseconds/BENCHMARK_REPEAT);
    gb_per_sec = total_bytes / ((milliseconds / BENCHMARK_REPEAT) / 1000.0) / 1e9;
    printf("Effective Bandwidth: %.2f GB/s\n", gb_per_sec);
    printf("------------------------------------------------------------------------------------------------------------\n");


    //////////////////////////////////////////////////////////////////////////////////////
    //     11. Brent_Kung_scan_kernel_optimized_padding
    //////////////////////////////////////////////////////////////////////////////////////
    items = 1024*2;
    grid_size = (TOTAL_N+items-1)/items;

    block.x = BLOCK_SIZE;
    grid.x = grid_size;
    printf("grid:  %5d, %5d, %5d\n", grid.x, grid.y, grid.z);
    printf("block: %5d, %5d, %5d\n", block.x, block.y, block.z);

    // 2.1 Warmup
    for (int i=0;i<10;i++){
        Brent_Kung_scan_kernel_optimized_padding<2048><<<grid, block>>>(d_input, d_output, TOTAL_N);
    }
    cudaDeviceSynchronize();
    printf("Warmup complete for Brent_Kung_scan_kernel_optimized_padding\n");

    // 2.2 Benchmarking
    cudaEventRecord(start);
    for (int i=0;i<BENCHMARK_REPEAT;i++){
        Brent_Kung_scan_kernel_optimized_padding<2048><<<grid, block>>>(d_input, d_output, TOTAL_N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Execution time for Brent_Kung_scan_kernel_optimized_padding: %f ms\n\n\n",milliseconds/BENCHMARK_REPEAT);
    gb_per_sec = total_bytes / ((milliseconds / BENCHMARK_REPEAT) / 1000.0) / 1e9;
    printf("Effective Bandwidth: %.2f GB/s\n", gb_per_sec);
    printf("------------------------------------------------------------------------------------------------------------\n");


    //////////////////////////////////////////////////////////////////////////////////////
    //     12. Brent_Kung_scan_kernel_optimized_padding_plus_vectorized_memory_access
    //////////////////////////////////////////////////////////////////////////////////////
    items = 1024*2;
    grid_size = (TOTAL_N+items-1)/items;

    block.x = BLOCK_SIZE;
    grid.x = grid_size;
    printf("grid:  %5d, %5d, %5d\n", grid.x, grid.y, grid.z);
    printf("block: %5d, %5d, %5d\n", block.x, block.y, block.z);

    // 2.1 Warmup
    for (int i=0;i<10;i++){
        Brent_Kung_scan_kernel_optimized_padding_plus_vectorized_memory_access<2048><<<grid, block>>>(d_input, d_output, TOTAL_N);
    }
    cudaDeviceSynchronize();
    printf("Warmup complete for Brent_Kung_scan_kernel_optimized_padding_plus_vectorized_memory_access\n");

    // 2.2 Benchmarking
    cudaEventRecord(start);
    for (int i=0;i<BENCHMARK_REPEAT;i++){
        Brent_Kung_scan_kernel_optimized_padding_plus_vectorized_memory_access<2048><<<grid, block>>>(d_input, d_output, TOTAL_N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Execution time for Brent_Kung_scan_kernel_optimized_padding_plus_vectorized_memory_access: %f ms\n\n\n",milliseconds/BENCHMARK_REPEAT);
    gb_per_sec = total_bytes / ((milliseconds / BENCHMARK_REPEAT) / 1000.0) / 1e9;
    printf("Effective Bandwidth: %.2f GB/s\n", gb_per_sec);
    printf("------------------------------------------------------------------------------------------------------------\n");

    auto now_end = std::chrono::system_clock::now();
    std::time_t end_time = std::chrono::system_clock::to_time_t(now_end);
    std::cout << "Benchmark Finished at: " 
              << std::put_time(std::localtime(&end_time), "%Y-%m-%d %H:%M:%S") 
              << std::endl;


    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaFree(d_input);
    cudaFree(d_output);
    return 0;
}