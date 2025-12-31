#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <random>
#include <omp.h>

#define BENCHMARK_REPEAT 1000

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

void reduction_cpu(float* input, float* output, unsigned long long N){
    double sum = 0.0f;

    #pragma omp parallel for reduction(+:sum)
    for (unsigned long long i=0;i<N;++i){
        sum += (double)input[i];
    }
    *output = (float)sum;
}

template<unsigned int BLOCK_DIM>
__global__ void CoarsenedShuffleSumReductionKernel(float* input, float* output, unsigned long long N, unsigned long long coarse_factor){
    __shared__ float input_s[BLOCK_DIM];
    unsigned long long segment = coarse_factor*2*blockDim.x*blockIdx.x;
    unsigned long long i = segment + threadIdx.x;
    unsigned int t = threadIdx.x;

    float sum = 0.0f;

    if (i<N){
        sum = input[i];
    }
    for(unsigned int tile=1;tile<coarse_factor*2;++tile){
        unsigned long long idx = i+(unsigned long long)tile*BLOCK_DIM;
        if (idx<N){
            sum += input[idx];
        }
    }
    input_s[t] = sum;

    for (unsigned int stride = blockDim.x/2; stride>= 32; stride/=2){
        __syncthreads();
        if (t<stride) {
            input_s[t] += input_s[t+stride];
        }
    }

    if (t<32) {
        float val = input_s[t];
        unsigned int mask = 0xffffffff;
        val += __shfl_down_sync(mask, val, 16); 
        val += __shfl_down_sync(mask, val, 8);
        val += __shfl_down_sync(mask, val, 4);
        val += __shfl_down_sync(mask, val, 2);
        val += __shfl_down_sync(mask, val, 1);

        if (t==0) {
            atomicAdd(output, val);
        }
    }
}

void launch_kernel(float* input, float* output ,unsigned int block_dim, unsigned int grid_dim, unsigned long long N, unsigned long long coarse_factor){
    // Timing setup
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float milliseconds;

    dim3 block(block_dim);
    dim3 grid(grid_dim);
    printf("grid:  %5d, %5d, %5d\n", grid.x, grid.y, grid.z);
    printf("block: %5d, %5d, %5d\n", block.x, block.y, block.z);
    printf("Total Elements:  %20llu    COARSE_FACTOR:   %20llu\n\n",N,coarse_factor);

    // 2.1 Warmup
    for (int i=0;i<5;i++){
        switch (block_dim) {
            case 32:
                CoarsenedShuffleSumReductionKernel<32><<<grid, block>>>(input,output,N,coarse_factor);
                break;
            case 64:
                CoarsenedShuffleSumReductionKernel<64><<<grid, block>>>(input,output,N,coarse_factor);
                break;
            case 128:
                CoarsenedShuffleSumReductionKernel<128><<<grid, block>>>(input,output,N,coarse_factor);
                break;
            case 256:
                CoarsenedShuffleSumReductionKernel<256><<<grid, block>>>(input,output,N,coarse_factor);
                break;
            case 512:
                CoarsenedShuffleSumReductionKernel<512><<<grid, block>>>(input,output,N,coarse_factor);
                break;
            case 1024:
                CoarsenedShuffleSumReductionKernel<1024><<<grid, block>>>(input,output,N,coarse_factor);
                break;
            default:
                printf("unsupported block size\n");
        }
        cudaDeviceSynchronize();
    }
    printf("Warmup complete for CoarsenedShuffleSumReductionKernel\n");

    // 2.2 Benchmarking
    cudaEventRecord(start);
    for (int i=0;i<BENCHMARK_REPEAT;i++){
        switch (block_dim) {
            case 32:
                CoarsenedShuffleSumReductionKernel<32><<<grid, block>>>(input,output,N,coarse_factor);
                break;
            case 64:
                CoarsenedShuffleSumReductionKernel<64><<<grid, block>>>(input,output,N,coarse_factor);
                break;
            case 128:
                CoarsenedShuffleSumReductionKernel<128><<<grid, block>>>(input,output,N,coarse_factor);
                break;
            case 256:
                CoarsenedShuffleSumReductionKernel<256><<<grid, block>>>(input,output,N,coarse_factor);
                break;
            case 512:
                CoarsenedShuffleSumReductionKernel<512><<<grid, block>>>(input,output,N,coarse_factor);
                break;
            case 1024:
                CoarsenedShuffleSumReductionKernel<1024><<<grid, block>>>(input,output,N,coarse_factor);
                break;
            default:
                printf("unsupported block size\n");
        }
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Execution time for CoarsenedShuffleSumReductionKernel: %f ms\n\n\n",milliseconds/BENCHMARK_REPEAT);

}

int main(int argc, char **argv){
    // Device Information
    int dev=3;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("Currently using CUDA device %d: %s\n", dev, deviceProp.name);
    CHECK(cudaSetDevice(dev));

    constexpr unsigned long long N = 1ULL<<35;
    constexpr unsigned int BLOCK_DIM = 512;
    constexpr unsigned int GRID_DIM = 132*128;
    constexpr unsigned long long DIVISOR = (unsigned long long)BLOCK_DIM * GRID_DIM * 2;
    constexpr unsigned long long COARSE_FACTOR = (N + DIVISOR - 1) / DIVISOR;

    // 1. Verify Correctness

    float avg = 0.0;
    float std = 1.0;

    float* h_data = (float*) malloc(N*sizeof(float));

    #pragma omp parallel 
        {
            std::mt19937 local_gen(1234 + omp_get_thread_num()); 
            std::normal_distribution<float> local_d(avg, std);

            #pragma omp for
            for (unsigned long long i = 0; i < N; ++i) {
                h_data[i] = local_d(local_gen);
            }
        }

    float h_result_cpu = 0.0f;
    float h_result_gpu = 0.0f;

    reduction_cpu(h_data, &h_result_cpu, N);

    float* d_data = NULL;
    float* d_result_gpu = NULL;
    cudaMalloc((void **)&d_data, N*sizeof(float));
    cudaMalloc((void **)&d_result_gpu, sizeof(float));

    cudaMemcpy(d_data, h_data, N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_result_gpu,0,sizeof(float));

    dim3 block(BLOCK_DIM);
    dim3 grid(GRID_DIM);
    CoarsenedShuffleSumReductionKernel<BLOCK_DIM><<<grid, block>>>(d_data, d_result_gpu, N,COARSE_FACTOR);

    cudaMemcpy(&h_result_gpu, d_result_gpu, sizeof(float), cudaMemcpyDeviceToHost);


    printf("Total elements N: %llu\n", N);
    printf("grid:  %5d, %5d, %5d\n", grid.x, grid.y, grid.z);
    printf("block: %5d, %5d, %5d\n\n", block.x, block.y, block.z);
    printf("h_result_cpu: %f\n", h_result_cpu);
    printf("h_result_gpu: %f\n", h_result_gpu);

    // 2. Benchmarking
    // Strictly speaking, d_result_gpu should be reset to 0 for each iteration. 
    // However, since this is purely for benchmarking performance rather than verifying calculation results, 
    // we skip the reset to avoid overhead.
    unsigned long long n_list[] = {1ULL<<26,1ULL<<27,1ULL<<28,1ULL<<29,1ULL<<30,1ULL<<31,1ULL<<32,1ULL<<33,1ULL<<34,1ULL<<35};
    unsigned int block_sizes[] = {32, 64, 128, 256, 512, 1024};
    unsigned int grid_multipliers[] = {1, 2, 4, 8, 16, 32, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048};
    // Hardcoded SM count for NVIDIA H200; bypassing the deviceProp.multiProcessorCount query.
    unsigned int numSMs = 132;
    
    for (unsigned long long n : n_list) {
        printf("\n################################################################\n");
        printf("########## Starting Benchmark for N: %20llu\n", n);
        printf("################################################################\n");
        for (unsigned int b : block_sizes) {

            printf("\n################################################################\n");
            printf("Starting Benchmark for Block Size: %5d\n", b);
            printf("################################################################\n");
            // bool is_coarse_factor_already_one = false; 
            // With fixed N and BLOCK_SIZE, increasing grid_multipliers may reduce coarse_factor to 1.
            // Once it reaches 1, we stop testing larger grid_multipliers.
            
            for ( unsigned int m : grid_multipliers) {
                // if (is_coarse_factor_already_one) break;
                unsigned long long DIVISOR = (unsigned long long)b * m * numSMs * 2;
                unsigned long long coarse_factor = (n + DIVISOR - 1) / DIVISOR;
                // if (coarse_factor == 1) is_coarse_factor_already_one = true;
                launch_kernel(d_data, d_result_gpu, b, m*numSMs,n,coarse_factor);
                printf("-------------------------------\n");
            }
        }
        printf("\n\n\n\n\n\n\n\n\n\n");
    }

    free(h_data);
    cudaFree(d_data);
    cudaFree(d_result_gpu);
    cudaDeviceReset();

    return EXIT_SUCCESS;
}