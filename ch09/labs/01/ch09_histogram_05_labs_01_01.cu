#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <cstdio>
#include <cstring>
#include <cstdlib>

#define NUM_BINS 7
#define INPUT_REPETITIONS 1000
#define BENCHMARK_REPEAT 1000
#define BLOCK_X 256
#define GRID_X (132*32)

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


void histogram_sequential(const char *data, unsigned int length, unsigned int *histo) {
    for (unsigned int i =0;i<length;++i) {
        int alphabet_position = data[i]-'a';
        if (alphabet_position>=0 && alphabet_position<26)
            histo[alphabet_position/4]++;
    }
}

void repeat(std::string& str, int n){
    if (n<=1) return;
    std::string pattern = str;
    str.reserve(str.size()*n);
    for(unsigned int i =1;i<n;++i){
        str += pattern;
    }
}

void print_histo(const char* name, unsigned int *histo) {
    printf("[%s]  ", name);
    printf(
        "a-d: %8u  e-h: %8u  i-l: %8u  m-p: %8u  q-t: %8u  u-x: %8u  y-z: %8u\n",
        histo[0]/INPUT_REPETITIONS,histo[1]/INPUT_REPETITIONS,histo[2]/INPUT_REPETITIONS,histo[3]/INPUT_REPETITIONS,histo[4]/INPUT_REPETITIONS,histo[5]/INPUT_REPETITIONS,histo[6]/INPUT_REPETITIONS
    );
}

__global__ void histo_private_shared_memory_coarsening_interleaved(char *data, unsigned int length, unsigned int *histo){
    __shared__ unsigned int histo_s[NUM_BINS];

    // Initialize shared memory
    for(unsigned int bin=threadIdx.x;bin<NUM_BINS;bin+=blockDim.x){
        histo_s[bin]=0u;
    }
    __syncthreads();

    unsigned int tid = blockIdx.x*blockDim.x+threadIdx.x;

    // Grid-Stride Loop: Allows processing data arrays larger than the total number of threads
    for(unsigned int i=tid;i<length;i += blockDim.x*gridDim.x){
        int alphabet_position =data[i]-'a';
        if(alphabet_position>=0 && alphabet_position<26){
            atomicAdd(&(histo_s[alphabet_position/4]),1);
        }
    }
    __syncthreads();

    for(unsigned int bin=threadIdx.x;bin<NUM_BINS;bin+=blockDim.x){
        unsigned int binValue = histo_s[bin];
        if (binValue>0){
            atomicAdd(&(histo[bin]),binValue);
        }
    }
}


void lauch_kernel(int grid_size, int block_size, char *d_text_data, unsigned int length, unsigned int *d_histo_gpu){

    // Timing setup
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float milliseconds;

    dim3 block(block_size);
    dim3 grid(grid_size);
    printf("grid:  %5d, %5d, %5d\n", grid.x, grid.y, grid.z);
    printf("block: %5d, %5d, %5d\n\n", block.x, block.y, block.z);
    printf("Grid-Stride Loop Iterations: %5d\n", (length+block.x*grid.x-1)/(block.x*grid.x));

    // 2.1 Warmup
    for (int i=0;i<5;i++){
        histo_private_shared_memory_coarsening_interleaved<<<grid, block>>>(d_text_data, length, d_histo_gpu);
        cudaDeviceSynchronize();
    }
    printf("Warmup complete for histo_private_shared_memory_coarsening_interleaved\n");

    // 2.2 Benchmarking
    cudaEventRecord(start);
    for (int i=0;i<BENCHMARK_REPEAT;i++){
        histo_private_shared_memory_coarsening_interleaved<<<grid, block>>>(d_text_data, length, d_histo_gpu);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Execution time for histo_private_shared_memory_coarsening_interleaved: %f ms\n\n\n",milliseconds/BENCHMARK_REPEAT);

}


int main(int argc, char **argv){
    // Device Information
    int dev=0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("Currently using CUDA device %d: %s\n", dev, deviceProp.name);
    printf("Number of SMs: %d\n", deviceProp.multiProcessorCount);
    CHECK(cudaSetDevice(dev));

    // File Input
    std::string content;
    {
        std::ifstream inputFile("../../pg64317.txt");
        std::stringstream buffer;
        buffer << inputFile.rdbuf();
        content = buffer.str();
    } // The buffer is destructed here, releasing memory (RAII)

    // The original text might be too short for effective benchmarking (instant completion).
    // Therefore, we repeat the content INPUT_REPETITIONS times.
    repeat(content, INPUT_REPETITIONS);

    const char* text_data = content.c_str();
    unsigned int length = strlen(text_data);
    printf("Text length to process: %u characters\n", length);

    // Kernel Dimension Settings
    dim3 block(BLOCK_X);
    dim3 grid(GRID_X);
    printf("grid:  %5d, %5d, %5d\n", grid.x, grid.y, grid.z);
    printf("block: %5d, %5d, %5d\n\n", block.x, block.y, block.z);

    unsigned int *h_histo_cpu = (unsigned int *) malloc(NUM_BINS*sizeof(unsigned int));
    unsigned int *h_histo_gpu = (unsigned int *) malloc(NUM_BINS*sizeof(unsigned int)); 
    memset(h_histo_cpu, 0, NUM_BINS*sizeof(unsigned int));
    // h_histo_gpu does not need initialization as it will be overwritten by data from the device

    // GPU Memory Allocation
    char *d_text_data = NULL;
    unsigned int *d_histo_gpu = NULL;

    size_t text_data_Bytes = content.length();
    size_t histo_Bytes = NUM_BINS*sizeof(unsigned int);

    // Strictly speaking, the trailing '\0' is not copied to the device.
    // This is acceptable here because the kernel iterates based on the explicit 'length'.
    cudaMalloc((void **)&d_text_data, text_data_Bytes); 
    cudaMalloc((void **)&d_histo_gpu, histo_Bytes);
    
    cudaMemcpy(d_text_data, text_data, text_data_Bytes, cudaMemcpyHostToDevice);
    cudaMemset(d_histo_gpu, 0, histo_Bytes);


    // 1. Verify Correctness
    // CPU execution/Host
    histogram_sequential(text_data, length, h_histo_cpu); 

    // GPU execution/Device
    histo_private_shared_memory_coarsening_interleaved<<<grid, block>>>(d_text_data, length, d_histo_gpu);
    cudaDeviceSynchronize();
    cudaMemcpy(h_histo_gpu, d_histo_gpu, histo_Bytes, cudaMemcpyDeviceToHost);
    
    // Print Comparison
    print_histo("histo_cpu", h_histo_cpu);
    print_histo("histo_gpu", h_histo_gpu);

    /* 
       Normally, we must re-initialize d_histo_gpu to 0 before every calculation:
       cudaMemset(d_histo_gpu, 0, histo_Bytes);
       
       However, below we are benchmarking for speed only and do not care about the accumulated result values.
       Therefore, we skip re-initialization to measure pure kernel execution time.
    */
    
    int block_sizes[] = {32, 64, 128, 256, 512, 1024};
    int grid_multipliers[] = {1, 2, 4, 8, 16, 32, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048};
    // Hardcoded SM count for NVIDIA H200; bypassing the deviceProp.multiProcessorCount query.
    int numSMs = 132;

    for (int b : block_sizes) {

        printf("\n################################################################\n");
        printf("Starting Benchmark for Block Size: %5d\n", b);
        printf("################################################################\n");

        for (int m : grid_multipliers) {
            lauch_kernel(m*numSMs,b,d_text_data, length, d_histo_gpu);
            printf("-------------------------------\n");
        }
    }

    free(h_histo_cpu);
    free(h_histo_gpu);
    cudaFree(d_text_data);
    cudaFree(d_histo_gpu);
    cudaDeviceReset();

    return EXIT_SUCCESS;
}