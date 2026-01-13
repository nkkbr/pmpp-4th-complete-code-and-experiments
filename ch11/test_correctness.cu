#include <iostream>
#include <vector>
#include <random>
#include <ctime>
#include <chrono>
#include <iomanip>
#include "scan_kernels.cuh"

#define CHECK(call)                                                            \
{                                                                              \
    const cudaError_t error = call;                                            \
    if (error != cudaSuccess)                                                  \
    {                                                                          \
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);                 \
        fprintf(stderr, "code: %d, reason: %s\n", error,                       \
                cudaGetErrorString(error));                                    \
        exit(1);                                                               \
    }                                                                          \
}

void cpu_scan(const std::vector<float>& input, std::vector<float>& output) {
    if (input.empty()) return;
    output[0] = input[0];
    for(size_t i=1;i<input.size();++i){
        output[i] = output[i-1]+input[i];
    }
}

bool verify(
    const float* cpu_res, 
    const float* gpu_res, 
    int n, 
    const std::string& kernel_name
){
    const float epsilon = 5e-3;

    for(int i=0;i<n;++i){
        float diff = std::abs(cpu_res[i]-gpu_res[i]);
        if (diff > epsilon && diff / (std::abs(cpu_res[i])+1e-6) > epsilon) {
            printf("[\033[31mFAIL\033[0m] %s\n",kernel_name.c_str());
            printf("    Mismatch at index %d: CPU=%f, GPU=%f, Diff=%f\n", i, cpu_res[i], gpu_res[i], diff);
            return false;
        }
    }
    printf("[\033[32mPASS\033[0m] %s (N=%d)\n", kernel_name.c_str(), n);
    return true;
}

int main() {

    auto now = std::chrono::system_clock::now();
    std::time_t now_time = std::chrono::system_clock::to_time_t(now);
    std::cout << "Verify started at: " 
            << std::put_time(std::localtime(&now_time), "%Y-%m-%d %H:%M:%S") 
            << std::endl;

    const int MAX_N = 20480;

    std::vector<float> h_input(MAX_N);
    std::vector<float> h_cpu_output(MAX_N);
    std::vector<float> h_gpu_output(MAX_N);

    std::mt19937 gen(1234);
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);
    for(int i=0;i<MAX_N;++i){
        h_input[i] = dis(gen);
    }

    printf("Computing CPU Naive Scan for comparison...\n");
    cpu_scan(h_input, h_cpu_output);
    printf("Done.\n\n");

    float *d_input, *d_output;
    cudaMalloc(&d_input, MAX_N*sizeof(float));
    cudaMalloc(&d_output, MAX_N*sizeof(float));
    cudaMemcpy(d_input, h_input.data(), MAX_N*sizeof(float), cudaMemcpyHostToDevice);

    unsigned int n;
    // Test1
    n=1024;
    printf("Kogge_Stone_scan_kernel\n\n");
    cudaMemset(d_output, 0, MAX_N*sizeof(float));
    Kogge_Stone_scan_kernel<1024><<<1, 1024>>>(d_input, d_output, n);
    cudaDeviceSynchronize();
    cudaMemcpy(h_gpu_output.data(), d_output, n*sizeof(float),cudaMemcpyDeviceToHost);
    verify(h_cpu_output.data(), h_gpu_output.data(), n, "Kogge_Stone_scan_kernel");
    printf("-----------------------\n\n");

    // Test2
    n=1024;
    printf("Kogge_Stone_scan_kernel_double_buffering\n\n");
    cudaMemset(d_output, 0, MAX_N*sizeof(float));
    Kogge_Stone_scan_kernel_double_buffering<1024><<<1, 1024>>>(d_input, d_output, n);
    cudaDeviceSynchronize();
    cudaMemcpy(h_gpu_output.data(), d_output, n*sizeof(float),cudaMemcpyDeviceToHost);
    verify(h_cpu_output.data(), h_gpu_output.data(), n, "Kogge_Stone_scan_kernel_double_buffering");
    printf("-----------------------\n\n");

    // Test3
    n=1024;
    printf("Kogge_Stone_scan_kernel_circular_buffer\n\n");
    cudaMemset(d_output, 0, MAX_N*sizeof(float));
    Kogge_Stone_scan_kernel_circular_buffer<1024><<<1, 1024>>>(d_input, d_output, n);
    cudaDeviceSynchronize();
    cudaMemcpy(h_gpu_output.data(), d_output, n*sizeof(float),cudaMemcpyDeviceToHost);
    verify(h_cpu_output.data(), h_gpu_output.data(), n, "Kogge_Stone_scan_kernel_circular_buffer");
    printf("-----------------------\n\n");

    // Test4
    n=1024;
    printf("Kogge_Stone_scan_kernel_shfl_up_sync_version\n\n");
    cudaMemset(d_output, 0, MAX_N*sizeof(float));
    Kogge_Stone_scan_kernel_shfl_up_sync_version<<<1, 1024>>>(d_input, d_output, n);
    cudaDeviceSynchronize();
    cudaMemcpy(h_gpu_output.data(), d_output, n*sizeof(float),cudaMemcpyDeviceToHost);
    verify(h_cpu_output.data(), h_gpu_output.data(), n, "Kogge_Stone_scan_kernel_shfl_up_sync_version");
    printf("-----------------------\n\n");

    // Test5
    n=4096;
    printf("Kogge_Stone_scan_kernel_shfl_up_sync_version_coarsening_degree_4\n\n");
    cudaMemset(d_output, 0, MAX_N*sizeof(float));
    Kogge_Stone_scan_kernel_shfl_up_sync_version_coarsening_degree_4<<<1, 1024>>>(d_input, d_output, n);
    cudaDeviceSynchronize();
    cudaMemcpy(h_gpu_output.data(), d_output, n*sizeof(float),cudaMemcpyDeviceToHost);
    verify(h_cpu_output.data(), h_gpu_output.data(), n, "Kogge_Stone_scan_kernel_shfl_up_sync_version_coarsening_degree_4");
    printf("-----------------------\n\n");

    // Test6
    n=8192;
    printf("Kogge_Stone_scan_kernel_shfl_up_sync_version_coarsening_degree_8\n\n");
    cudaMemset(d_output, 0, MAX_N*sizeof(float));
    Kogge_Stone_scan_kernel_shfl_up_sync_version_coarsening_degree_8<<<1, 1024>>>(d_input, d_output, n);
    cudaDeviceSynchronize();
    cudaMemcpy(h_gpu_output.data(), d_output, n*sizeof(float),cudaMemcpyDeviceToHost);
    verify(h_cpu_output.data(), h_gpu_output.data(), n, "Kogge_Stone_scan_kernel_shfl_up_sync_version_coarsening_degree_8");
    printf("-----------------------\n\n");

    // Test7
    n=12288;
    printf("Kogge_Stone_scan_kernel_shfl_up_sync_version_coarsening_degree_12\n\n");
    cudaMemset(d_output, 0, MAX_N*sizeof(float));
    Kogge_Stone_scan_kernel_shfl_up_sync_version_coarsening_degree_12<<<1, 1024>>>(d_input, d_output, n);
    cudaDeviceSynchronize();
    cudaMemcpy(h_gpu_output.data(), d_output, n*sizeof(float),cudaMemcpyDeviceToHost);
    verify(h_cpu_output.data(), h_gpu_output.data(), n, "Kogge_Stone_scan_kernel_shfl_up_sync_version_coarsening_degree_12");
    printf("-----------------------\n\n");

    // Test8
    n=16384;
    printf("Kogge_Stone_scan_kernel_shfl_up_sync_version_coarsening_degree_16\n\n");
    cudaMemset(d_output, 0, MAX_N*sizeof(float));
    Kogge_Stone_scan_kernel_shfl_up_sync_version_coarsening_degree_16<<<1, 1024>>>(d_input, d_output, n);
    cudaDeviceSynchronize();
    cudaMemcpy(h_gpu_output.data(), d_output, n*sizeof(float),cudaMemcpyDeviceToHost);
    verify(h_cpu_output.data(), h_gpu_output.data(), n, "Kogge_Stone_scan_kernel_shfl_up_sync_version_coarsening_degree_16");
    printf("-----------------------\n\n");

    // Test9
    n=20480;
    printf("Kogge_Stone_scan_kernel_shfl_up_sync_version_coarsening_degree_20\n\n");
    cudaMemset(d_output, 0, MAX_N*sizeof(float));
    Kogge_Stone_scan_kernel_shfl_up_sync_version_coarsening_degree_20<<<1, 1024>>>(d_input, d_output, n);
    cudaDeviceSynchronize();
    cudaMemcpy(h_gpu_output.data(), d_output, n*sizeof(float),cudaMemcpyDeviceToHost);
    verify(h_cpu_output.data(), h_gpu_output.data(), n, "Kogge_Stone_scan_kernel_shfl_up_sync_version_coarsening_degree_20");
    printf("-----------------------\n\n");

    // Test10
    n=2048;
    printf("Brent_Kung_scan_kernel\n\n");
    cudaMemset(d_output, 0, MAX_N*sizeof(float));
    Brent_Kung_scan_kernel<2048><<<1, 1024>>>(d_input, d_output, n);
    cudaDeviceSynchronize();
    cudaMemcpy(h_gpu_output.data(), d_output, n*sizeof(float),cudaMemcpyDeviceToHost);
    verify(h_cpu_output.data(), h_gpu_output.data(), n, "Brent_Kung_scan_kernel");
    printf("-----------------------\n\n");

    // Test11
    n=2048;
    printf("Brent_Kung_scan_kernel_optimized_padding\n\n");
    cudaMemset(d_output, 0, MAX_N*sizeof(float));
    Brent_Kung_scan_kernel_optimized_padding<2048><<<1, 1024>>>(d_input, d_output, n);
    cudaDeviceSynchronize();
    cudaMemcpy(h_gpu_output.data(), d_output, n*sizeof(float),cudaMemcpyDeviceToHost);
    verify(h_cpu_output.data(), h_gpu_output.data(), n, "Brent_Kung_scan_kernel_optimized_padding");
    printf("-----------------------\n\n");

    // Test12
    n=2048;
    printf("Brent_Kung_scan_kernel_optimized_padding_plus_vectorized_memory_access\n\n");
    cudaMemset(d_output, 0, MAX_N*sizeof(float));
    Brent_Kung_scan_kernel_optimized_padding_plus_vectorized_memory_access<2048><<<1, 1024>>>(d_input, d_output, n);
    cudaDeviceSynchronize();
    cudaMemcpy(h_gpu_output.data(), d_output, n*sizeof(float),cudaMemcpyDeviceToHost);
    verify(h_cpu_output.data(), h_gpu_output.data(), n, "Brent_Kung_scan_kernel_optimized_padding_plus_vectorized_memory_access");
    printf("-----------------------\n\n");
    
    cudaFree(d_input);
    cudaFree(d_output);

    auto now_end = std::chrono::system_clock::now();
    std::time_t end_time = std::chrono::system_clock::to_time_t(now_end);
    std::cout << "Verify Finished at: " 
              << std::put_time(std::localtime(&end_time), "%Y-%m-%d %H:%M:%S") 
              << std::endl;

    return 0;
}