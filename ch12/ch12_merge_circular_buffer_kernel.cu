// 这份文件是直接复制了`ch12_merge_tiled_kernel.cu`，再经过修改得来的，所以有很多的重复的部分
// 虽然不是一个好的做法，不过这里只是为了快速地进行实验

#include <cuda_runtime.h>
#include <iostream>
#include <random>
#include <cstdlib>
#include <omp.h>
#include <math.h>
#include <chrono>
#include <ctime>
#include <iomanip>

#define BENCHMARK_REPEAT 50

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

void fill_sorted_efficiently(float* data, long long count) {
    
    std::mt19937_64 rng(std::random_device{}());
    std::uniform_real_distribution<float> dist(0.0f, 0.000001f);

    double current_val = -1073.741824f + dist(rng)*1000;
    
    for (long long i = 0; i < count; ++i) {
        current_val += dist(rng);
        data[i] = (float)current_val;
    }
}

void check_equal(float *A, float *B, long long size){
    // 虽说float一般来说不应该使用 != 来判断是不是不想等
    // 但本代码中，都是搬运数据，不涉及数据的加减，所以，不会有精度上的问题。
    // 可以，也应该直接使用 != 来进行判断
    bool is_equal = true;
    for(long long i =0;i<size;++i){
        if (A[i] != B[i]){
            printf("EQUAL CHECK FAILED! i = %llu ,CPU=%f, GPU=%f\n\n",i, A[i], B[i]);
            is_equal = false;
            break;
        }
    }
    if (is_equal){
        printf("EQUAL CHECK SUCCESS!\n\n");
    }
}

__host__ __device__ void merge_sequential(float *A, long long m, float *B, long long n, float *C){
    long long i = 0;
    long long j = 0;
    long long k = 0;

    while((i<m)&&(j<n)){
        if (A[i]<=B[j]){
            C[k++] = A[i++];
        } else {
            C[k++] = B[j++];
        }
    }

    if (i==m) {
        while(j<n) {
            C[k++] = B[j++];
        }
    } else {
        while(i<m){
            C[k++] = A[i++];
        }
    }
}

__device__ void merge_sequential_circular(
    float *A, 
    long long m, 
    float *B, 
    long long n, 
    float *C, 
    long long A_S_start, 
    long long B_S_start, 
    long long tile_size
) {
    long long i = 0;
    long long j = 0;
    long long k = 0;

    while ((i<m) && (j<n)) {
        long long i_cir = (A_S_start+i)%tile_size;
        long long j_cir = (B_S_start+j)%tile_size;
        if (A[i_cir] <= B[j_cir]) {
            C[k++] = A[i_cir];
            ++i;
        } else {
            C[k++] = B[j_cir];
            ++j;
        }
    }
    if (i==m) {
        while(j<n){
            long long j_cir = (B_S_start+j)%tile_size;
            C[k++] = B[j_cir];
            ++j;
        }
    } else {
        while (i<m){
            long long i_cir = (A_S_start+i)%tile_size;
            C[k++] = A[i_cir];
            ++i;
        }
    }
}

__host__ __device__ long long co_rank(long long k, float* A, long long m, float* B, long long n){
    long long i = k < m ? k : m;
    long long j = k - i;
    long long i_low = 0 > (k-n) ? 0 : k-n;
    long long j_low = 0 > (k-m) ? 0 : k-m;
    long long delta;
    bool active = true;
    while(active) {
        if (i>0 && j<n && A[i-1] > B[j]) {
            delta = ((i-i_low+1)>>1);
            j_low=j;
            j += delta;
            i -= delta;
        } else if (j>0 && i<m && B[j-1] >= A[i]) {
            delta = ((j-j_low+1)>>1);
            i_low=i;
            i += delta;
            j -= delta;
        } else {
            active = false;
        }
    }
    return i;
}

__device__ long long co_rank_circular(
    long long k,
    float* A,
    long long m,
    float* B,
    long long n,
    long long A_S_start, 
    long long B_S_start, 
    long long tile_size
){
    long long i = k < m ? k : m;
    long long j = k - i;
    long long i_low = 0 > (k-n) ? 0 : k-n;
    long long j_low = 0 > (k-m) ? 0 : k-m;
    long long delta;
    bool active = true;
    while(active) {
        long long i_cir = (A_S_start+i)%tile_size;
        long long i_m_1_cir = (A_S_start+i-1)%tile_size;
        long long j_cir = (B_S_start+j)%tile_size;
        long long j_m_1_cir = (B_S_start+j-1)%tile_size;
        if (i>0 && j<n && A[i_m_1_cir] > B[j_cir]){
            delta = ((i-i_low+1)>>1);
            j_low=j;
            j += delta;
            i -= delta;
        } else if (j>0 && i<m && B[j_m_1_cir] >= A[i_cir]) {
            delta = ((j-j_low+1)>>1);
            i_low=i;
            i += delta;
            j -= delta;
        } else {
            active = false;
        }
    }
    return i;
}

__global__ void merge_circular_buffer_kernel(float* A, long long m, float* B, long long n, float* C, unsigned int tile_size){
    extern __shared__ float sharedAB[];
    float * A_S = &sharedAB[0];
    float * B_S = &sharedAB[tile_size];
    
    // int C_curr = blockIdx.x * ceil((m+n)/gridDim.x);
    long long total_work = m + n;
    long long grid_dim = gridDim.x; 
    long long elements_per_block = (total_work + grid_dim - 1) / grid_dim;

    long long C_curr = blockIdx.x*elements_per_block;
    long long C_next = min((blockIdx.x + 1) * elements_per_block, (m+n));

    if (threadIdx.x == 0) {
        long long* index_ptr = (long long*)A_S;
        index_ptr[0] = co_rank(C_curr,A,m,B,n);
        index_ptr[1] = co_rank(C_next,A,m,B,n);
    }
    __syncthreads();

    long long* index_ptr = (long long*)A_S;
    long long A_curr = index_ptr[0];
    long long A_next = index_ptr[1];
    long long B_curr = C_curr - A_curr;
    long long B_next = C_next - A_next;
    __syncthreads();

    long long counter = 0;
    long long C_length = C_next-C_curr;
    long long A_length = A_next-A_curr;
    long long B_length = B_next-B_curr;
    long long total_iteration = (C_length + tile_size -1)/tile_size;
    long long C_completed = 0;
    long long A_consumed = 0;
    long long B_consumed = 0;

    long long A_S_start = 0;
    long long B_S_start = 0;
    long long A_S_consumed = tile_size;
    long long B_S_consumed = tile_size;

    while(counter<total_iteration){
        for(unsigned int i = 0;i<A_S_consumed;i+=blockDim.x){
            if((i+threadIdx.x)<(A_length-A_consumed) && (i+threadIdx.x)<A_S_consumed) {
                A_S[(A_S_start+tile_size-A_S_consumed+i+threadIdx.x)%tile_size] = A[A_curr+A_consumed+(tile_size - A_S_consumed)+i+threadIdx.x];
            }
        }
        for(unsigned int i = 0;i<B_S_consumed;i+=blockDim.x){
            if((i+threadIdx.x)<(B_length-B_consumed) && (i+threadIdx.x)<B_S_consumed) {
                B_S[(B_S_start+tile_size-B_S_consumed+i+threadIdx.x)%tile_size] = B[B_curr+B_consumed+(tile_size - B_S_consumed)+i+threadIdx.x];
            }
        }
        __syncthreads();

        long long c_curr = threadIdx.x*(tile_size/blockDim.x);
        long long c_next = (threadIdx.x+1)*(tile_size/blockDim.x);
        c_curr = (c_curr<=C_length-C_completed) ? c_curr : C_length - C_completed;
        c_next = (c_next<=C_length-C_completed) ? c_next : C_length - C_completed;

        long long a_curr = co_rank_circular(c_curr, A_S, min((long long)tile_size, A_length-A_consumed), B_S, min((long long)tile_size, B_length-B_consumed), A_S_start, B_S_start, tile_size);
        long long b_curr = c_curr - a_curr;
        long long a_next = co_rank_circular(c_next, A_S, min((long long)tile_size, A_length-A_consumed), B_S, min((long long)tile_size, B_length-B_consumed), A_S_start, B_S_start, tile_size);
        long long b_next = c_next - a_next;

        merge_sequential_circular(A_S, a_next-a_curr, B_S, b_next-b_curr, C+C_curr+C_completed+c_curr, A_S_start+a_curr, B_S_start+b_curr, tile_size);
        counter++;

        long long current_tile_A_valid = min((long long)tile_size, A_length - A_consumed);
        long long current_tile_B_valid = min((long long)tile_size, B_length - B_consumed);

        // 原本书中的源代码，是这样的
        // A_consumed += co_rank(tile_size, A_S, tile_size, B_S, tile_size);
        // 但这样的代码，假定了A_S和B_S都有tile_size个有效的值。而如果A_S和B_S没有tile_size个有效的值，程序还会去考虑有效值之后的那些垃圾值，那么co_rank就算错了！
        A_S_consumed = co_rank_circular(
            min((long long)tile_size, C_length-C_completed), 
            A_S, current_tile_A_valid, 
            B_S, current_tile_B_valid,
            A_S_start, B_S_start, tile_size);
        B_S_consumed = min((long long)tile_size, C_length-C_completed) - A_S_consumed;

        A_consumed += A_S_consumed;
        C_completed += min((long long)tile_size, C_length-C_completed);
        B_consumed = C_completed - A_consumed;

        A_S_start = (A_S_start+A_S_consumed) % tile_size;
        B_S_start = (B_S_start+B_S_consumed) % tile_size;
        __syncthreads();
    }
}

void launch_kernel(unsigned int tile_size, unsigned int block_size, unsigned int grid_size, float* d_data_A, float* d_data_B, float* d_data_C, long long A, long long B, long long total_iteration){
    // Timing setup
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float milliseconds;

    dim3 block(block_size);
    dim3 grid(grid_size);
    printf("grid:  %5d, %5d, %5d\n", grid.x, grid.y, grid.z);
    printf("block: %5d, %5d, %5d\n", block.x, block.y, block.z);
    printf("total_iteration  %20llu\n\n",total_iteration);

    size_t shared_mem_bytes = 2 * tile_size * sizeof(float);

    // 2.1 Warmup
    for (int i=0;i<5;i++){
        merge_circular_buffer_kernel<<<grid, block,shared_mem_bytes>>>(d_data_A, A, d_data_B, B, d_data_C, tile_size);
    }
    cudaDeviceSynchronize();
    printf("Warmup complete for merge_circular_buffer_kernel\n");

    // 2.2 Benchmarking
    cudaEventRecord(start);
    for (int i=0;i<BENCHMARK_REPEAT;i++){
        merge_circular_buffer_kernel<<<grid, block,shared_mem_bytes>>>(d_data_A, A, d_data_B, B, d_data_C, tile_size);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Execution time for merge_circular_buffer_kernel: %f ms\n\n\n",milliseconds/BENCHMARK_REPEAT);


    // 打印一行 CSV 格式 的数据
    // 直接 grep "RESULT" 就能拿到所有数据，方便画图
    printf("RESULT, %d, %d, %d, %lld, %f\n", tile_size, block_size, grid_size, total_iteration, milliseconds/BENCHMARK_REPEAT);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main(int argc, char **argv){
    // Device Information
    int dev=0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("Currently using CUDA device %d: %s\n", dev, deviceProp.name);
    CHECK(cudaSetDevice(dev));

    long long N = (1ULL<<34) + 1733;
    long long A = (1ULL<<33) + 17;
    long long B = N - A;

    float* h_data_A = (float*) malloc(A*sizeof(float));
    float* h_data_B = (float*) malloc(B*sizeof(float));
    float* h_data_C = (float*) malloc(N*sizeof(float));
    float* h_data_C_gpu = (float*) malloc(N*sizeof(float));

    #pragma omp parallel sections
        {
            #pragma omp section
            {
                printf("Filling A...\n");
                fill_sorted_efficiently(h_data_A, A);
                printf("A filled.\n");
            }
            #pragma omp section
            {
                printf("Filling B...\n");
                fill_sorted_efficiently(h_data_B, B);
                printf("B filled.\n");
            }
        }

    printf("A[0]-A[4]:                       % 12.6f, % 12.6f, % 12.6f, % 12.6f, % 12.6f\n", h_data_A[0], h_data_A[1], h_data_A[2], h_data_A[3], h_data_A[4]);
    printf("A[1ULL<<31]-A[(1ULL<<31)+4]:     % 12.6f, % 12.6f, % 12.6f, % 12.6f, % 12.6f\n", h_data_A[(1ULL<<31)], h_data_A[(1ULL<<31)+1], h_data_A[(1ULL<<31)+2], h_data_A[(1ULL<<31)+3], h_data_A[(1ULL<<31)+4]);
    printf("A[last-4]-A[last]:               % 12.6f, % 12.6f, % 12.6f, % 12.6f, % 12.6f\n", h_data_A[A-5], h_data_A[A-4], h_data_A[A-3], h_data_A[A-2], h_data_A[A-1]);
    printf("B[0]-B[4]:                       % 12.6f, % 12.6f, % 12.6f, % 12.6f, % 12.6f\n", h_data_B[0], h_data_B[1], h_data_B[2], h_data_B[3], h_data_B[4]);
    printf("B[1ULL<<31]-B[(1ULL<<31)+4]:     % 12.6f, % 12.6f, % 12.6f, % 12.6f, % 12.6f\n", h_data_B[(1ULL<<31)], h_data_B[(1ULL<<31)+1], h_data_B[(1ULL<<31)+2], h_data_B[(1ULL<<31)+3], h_data_B[(1ULL<<31)+4]);
    printf("B[last-4]-B[last]:               % 12.6f, % 12.6f, % 12.6f, % 12.6f, % 12.6f\n", h_data_B[B-5], h_data_B[B-4], h_data_B[B-3], h_data_B[B-2], h_data_B[B-1]);

    merge_sequential(h_data_A, A, h_data_B, B, h_data_C);

    printf("C[0]-C[4]:                       % 12.6f, % 12.6f, % 12.6f, % 12.6f, % 12.6f\n", h_data_C[0], h_data_C[1], h_data_C[2], h_data_C[3], h_data_C[4]);
    printf("C[1ULL<<32]-C[(1ULL<<32)+4]:     % 12.6f, % 12.6f, % 12.6f, % 12.6f, % 12.6f\n", h_data_C[(1ULL<<32)], h_data_C[(1ULL<<32)+1], h_data_C[(1ULL<<32)+2], h_data_C[(1ULL<<32)+3], h_data_C[(1ULL<<32)+4]);
    printf("C[last-4]-C[last]:               % 12.6f, % 12.6f, % 12.6f, % 12.6f, % 12.6f\n", h_data_C[N-5], h_data_C[N-4], h_data_C[N-3], h_data_C[N-2], h_data_C[N-1]);

    float* d_data_A = NULL;
    float* d_data_B = NULL;
    float* d_data_C = NULL;

    cudaMalloc((void **)&d_data_A, A*sizeof(float));
    cudaMalloc((void **)&d_data_B, B*sizeof(float));
    cudaMalloc((void **)&d_data_C, N*sizeof(float));

    cudaMemcpy(d_data_A, h_data_A, A*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_data_B, h_data_B, B*sizeof(float), cudaMemcpyHostToDevice);

    unsigned int tile_size = 1024;
    unsigned int block_x = 512;
    unsigned int grid_x = 1000000;

    dim3 block(block_x);
    dim3 grid(grid_x);

    size_t shared_mem_bytes = 2 * tile_size * sizeof(float);
    merge_circular_buffer_kernel<<<grid, block,shared_mem_bytes>>>(d_data_A, A, d_data_B, B, d_data_C, tile_size);

    cudaMemcpy(h_data_C_gpu, d_data_C, N*sizeof(float), cudaMemcpyDeviceToHost);
    printf("C_gpu[0]-C_gpu[4]:                       % 12.6f, % 12.6f, % 12.6f, % 12.6f, % 12.6f\n", h_data_C_gpu[0], h_data_C_gpu[1], h_data_C_gpu[2], h_data_C_gpu[3], h_data_C_gpu[4]);
    printf("C_gpu[1ULL<<32]-C_gpu[(1ULL<<32)+4]:     % 12.6f, % 12.6f, % 12.6f, % 12.6f, % 12.6f\n", h_data_C_gpu[(1ULL<<32)], h_data_C_gpu[(1ULL<<32)+1], h_data_C_gpu[(1ULL<<32)+2], h_data_C_gpu[(1ULL<<32)+3], h_data_C_gpu[(1ULL<<32)+4]);
    printf("C_gpu[last-4]-C_gpu[last]:               % 12.6f, % 12.6f, % 12.6f, % 12.6f, % 12.6f\n", h_data_C_gpu[N-5], h_data_C_gpu[N-4], h_data_C_gpu[N-3], h_data_C_gpu[N-2], h_data_C_gpu[N-1]);
    check_equal(h_data_C,h_data_C_gpu,N);

    // 2. Benchmarking
    unsigned int tile_sizes[] = {512, 1024, 2048, 4096};
    unsigned int block_sizes[] = {32, 64, 128, 256, 512, 1024};
    unsigned int grid_multipliers[] = {1, 2, 4, 8, 16, 32, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 4096, 8192, 16384, 32768};
    // Hardcoded SM count for NVIDIA H200; bypassing the deviceProp.multiProcessorCount query.
    unsigned int numSMs = 132;

    auto now = std::chrono::system_clock::now();
    std::time_t now_time = std::chrono::system_clock::to_time_t(now);

    std::cout << "Benchmark started at: " 
            << std::put_time(std::localtime(&now_time), "%Y-%m-%d %H:%M:%S") 
            << std::endl;

    for (unsigned int ts : tile_sizes) {
        printf("\n################################################################\n");
        printf("########## Starting Benchmark for tile_sizes: %5d\n", ts);
        printf("################################################################\n");
        for (unsigned int b : block_sizes) {
            if (ts<b) continue;
            printf("\n################################################################\n");
            printf("Starting Benchmark for Block Size: %5d\n", b);
            printf("################################################################\n");
            
            for (unsigned int m : grid_multipliers) {

                // 需要向上取整
                long long current_grid_size = (long long)m * numSMs;
                long long max_elements_per_block = (N + current_grid_size - 1) / current_grid_size;
                long long real_total_iteration = (max_elements_per_block + ts - 1) / ts;

                launch_kernel(ts, b, m*numSMs, d_data_A, d_data_B, d_data_C, A, B, real_total_iteration);
                printf("-------------------------------\n");
            }
        }
        printf("\n\n\n\n\n\n\n\n\n\n");
    }

    auto now_end = std::chrono::system_clock::now();
    std::time_t end_time = std::chrono::system_clock::to_time_t(now_end);

    std::cout << "Benchmark Finished at: " 
              << std::put_time(std::localtime(&end_time), "%Y-%m-%d %H:%M:%S") 
              << std::endl;

    free(h_data_A);
    free(h_data_B);
    free(h_data_C);
    free(h_data_C_gpu);
    cudaFree(d_data_A);
    cudaFree(d_data_B);
    cudaFree(d_data_C);

    return EXIT_SUCCESS;
}