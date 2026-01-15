// scan_kernels.cuh
#ifndef SCAN_KERNELS_CUH
#define SCAN_KERNELS_CUH

#include <cuda_runtime.h>

// 这里的 SECTION_SIZE 最多只能是 1024，它应该与 blockDim.x 一致
template <unsigned int SECTION_SIZE>
__global__ void Kogge_Stone_scan_kernel(float *X, float *Y, unsigned int N){
    __shared__ float XY[SECTION_SIZE];
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i<N){
        XY[threadIdx.x] = X[i];
    } else {
        XY[threadIdx.x] = 0.0f; // 似乎不设置也没关系
    }

    for(unsigned int stride = 1; stride<blockDim.x; stride *=2){
        __syncthreads();
        float temp;
        if(threadIdx.x >= stride){
            temp = XY[threadIdx.x]+ XY[threadIdx.x-stride];
        }
        __syncthreads();
        if(threadIdx.x >= stride){
            XY[threadIdx.x] = temp;
        }
    }
    if(i<N){
        Y[i] = XY[threadIdx.x];
    }
}

template <unsigned int SECTION_SIZE>
__global__ void Kogge_Stone_scan_kernel_double_buffering(float *X, float *Y, unsigned int N){
    __shared__ float buffer_a[SECTION_SIZE];
    __shared__ float buffer_b[SECTION_SIZE];

    float *i_data = buffer_a;
    float *o_data = buffer_b;

    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i<N){
        i_data[threadIdx.x] = X[i];
    } else {
        i_data[threadIdx.x] = 0.0f;
    }

    for(unsigned int stride = 1; stride<blockDim.x; stride *=2){
        __syncthreads();
        if(threadIdx.x >= stride){
            o_data[threadIdx.x] = i_data[threadIdx.x] + i_data[threadIdx.x-stride];
        } else {
            o_data[threadIdx.x] = i_data[threadIdx.x];
        }

        float *tmp = i_data;
        i_data = o_data;
        o_data = tmp;
    }
    if(i<N){
        Y[i] = i_data[threadIdx.x];
    }
}

// 与 Kogge_Stone_scan_kernel_double_buffering 一样，少了一次 __syncthreads(); 性能相比 Kogge_Stone_scan_kernel 会有所提升
// 但事实上，取模运算 (%) 是性能毒药，所以这个 kernel 的性能预计会比 Kogge_Stone_scan_kernel_double_buffering 要差。
template <unsigned int SECTION_SIZE>
__global__ void Kogge_Stone_scan_kernel_circular_buffer(float *X, float *Y, unsigned int N){
    __shared__ float XY[SECTION_SIZE*2];
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i<N){
        XY[threadIdx.x] = X[i];
    } else {
        XY[threadIdx.x] = 0.0f;
    }

    unsigned int tmp_index = 0;
    for(unsigned int stride = 1; stride<blockDim.x; stride *=2){
        __syncthreads();
        if(threadIdx.x >= stride){
            XY[(threadIdx.x+tmp_index+SECTION_SIZE)%(SECTION_SIZE*2)]= XY[(threadIdx.x+tmp_index)%(SECTION_SIZE*2)]+ XY[(threadIdx.x-stride+tmp_index)%(SECTION_SIZE*2)];
        } else {
            XY[(threadIdx.x+tmp_index+SECTION_SIZE)%(SECTION_SIZE*2)]= XY[(threadIdx.x+tmp_index)%(SECTION_SIZE*2)];
        }
        tmp_index = (tmp_index+SECTION_SIZE)%(2*SECTION_SIZE);
    }
    if(i<N){
        Y[i] = XY[(threadIdx.x+tmp_index)%(SECTION_SIZE*2)];
    }
}


// 这个版本的区别，仅仅在于，并不是取模，而是用了位运算
// 这个版本有一个限制就是 2*SECTION_SIZE 必须是2的幂才行
// 这个版本到底能不能真的加速呢？很难说。其实，取模的版本，可能编译器本身就已经优化成了位运算了
// 有可能最终，还是指针交换胜出
template <unsigned int SECTION_SIZE>
__global__ void Kogge_Stone_scan_kernel_circular_buffer_v2(float *X, float *Y, unsigned int N){
    const unsigned int MASK = (SECTION_SIZE * 2) - 1;

    __shared__ float XY[SECTION_SIZE*2];
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i<N){
        XY[threadIdx.x] = X[i];
    } else {
        XY[threadIdx.x] = 0.0f;
    }

    unsigned int tmp_index = 0;
    for(unsigned int stride = 1; stride < blockDim.x; stride *= 2){
        __syncthreads();
        unsigned int current_read_base = (threadIdx.x + tmp_index) & MASK;
        unsigned int current_write_base = (threadIdx.x + tmp_index + SECTION_SIZE) & MASK;
        
        if(threadIdx.x >= stride){
            unsigned int neighbor_index = (threadIdx.x - stride + tmp_index) & MASK;
            XY[current_write_base] = XY[current_read_base] + XY[neighbor_index];
        } else {
            XY[current_write_base] = XY[current_read_base];
        }
        tmp_index = (tmp_index + SECTION_SIZE) & MASK;
    }
    if(i < N){
        Y[i] = XY[(threadIdx.x + tmp_index) & MASK];
    }
}

// 使用 shuffle instructions within warps 
// 仅仅用于性能比较用。测试本kernel，请在block_size=1024的设定下进行
__global__ void Kogge_Stone_scan_kernel_shfl_up_sync_version(float *X, float *Y, unsigned int N){
    // 只需要 32 个 float 的 Shared Memory 来做 Warp 间通信
    // 相比旧代码的 1024 个 float，省了 30 倍的 Shared Memory！
    // 所以，事实上有提升occupancy的可能性
    __shared__ float warp_sums[32];

    unsigned int tid = threadIdx.x;
    unsigned int lane_id = tid % 32;
    unsigned int warp_id = tid / 32;

    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // 将数据拿到每个thread的寄存器上
    float val = (idx < N) ? X[idx] : 0.0f;

    float neighbor;

    neighbor = __shfl_up_sync(0xffffffff, val, 1);
    if (lane_id>=1) val += neighbor;

    neighbor = __shfl_up_sync(0xffffffff, val, 2);
    if (lane_id>=2) val += neighbor;

    neighbor = __shfl_up_sync(0xffffffff, val, 4);
    if (lane_id>=4) val += neighbor;

    neighbor = __shfl_up_sync(0xffffffff, val, 8);
    if (lane_id>=8) val += neighbor;

    neighbor = __shfl_up_sync(0xffffffff, val, 16);
    if (lane_id>=16) val += neighbor;

    // 此时，每个warp内部各自的前缀和，已经计算好了

    if(lane_id==31){
        warp_sums[warp_id] = val;
    }

    __syncthreads();

    float warp_base = 0.0f;

    if(warp_id==0){
        float w_val = warp_sums[lane_id];

        float w_neighbor;

        w_neighbor = __shfl_up_sync(0xffffffff, w_val, 1);
        if (lane_id>=1) w_val += w_neighbor;

        w_neighbor = __shfl_up_sync(0xffffffff, w_val, 2);
        if (lane_id>=2) w_val += w_neighbor;

        w_neighbor = __shfl_up_sync(0xffffffff, w_val, 4);
        if (lane_id>=4) w_val += w_neighbor;

        w_neighbor = __shfl_up_sync(0xffffffff, w_val, 8);
        if (lane_id>=8) w_val += w_neighbor;
        
        w_neighbor = __shfl_up_sync(0xffffffff, w_val, 16);
        if (lane_id>=16) w_val += w_neighbor;

        warp_sums[lane_id] = w_val;
    }

    __syncthreads();

    if (warp_id>0) {
        warp_base = warp_sums[warp_id-1];
    }

    val += warp_base;

    if (idx<N) {
        Y[idx] = val;
    }
}


// 11.5 提到了线程粗化
// 我们来通过线程粗化，改进 Kogge_Stone_scan_kernel_shfl_up_sync_version
// 同样地，本kernel仅仅用于性能比较用。测试本kernel，请在block_size=1024的设定下进行
// 这个kernel，粗化的程度是4，所以，事实上它可以管理4096个数字
__global__ void Kogge_Stone_scan_kernel_shfl_up_sync_version_coarsening_degree_4(float *X, float *Y, unsigned int N){

    // shared memory是属于block的。一个block有1024个线程，那么就是有32个warp。
    __shared__ float warp_sums[32];

    unsigned int tid = threadIdx.x;
    unsigned int lane_id = tid % (32);
    unsigned int warp_id = tid / (32);

    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // 将数据拿到每个thread的寄存器上
    float4* X_vec = (float4*)X;
    float4* Y_vec = (float4*)Y;
    float4 val = (idx < N/4) ? X_vec[idx] : make_float4(0.0f,0.0f,0.0f,0.0f);

    // Phase 1：寄存器内的串行扫描 (Serial Scan)
    val.y += val.x;
    val.z += val.y;
    val.w += val.z;

    // Phase 2: 计算一个warp内部的前缀和

    float neighbor;
    float warp_accumulated_offset = 0.0f;

    neighbor = __shfl_up_sync(0xffffffff, val.w, 1);
    if (lane_id>=1) {
        val.w += neighbor;
        warp_accumulated_offset += neighbor;
    }

    neighbor = __shfl_up_sync(0xffffffff, val.w, 2);
    if (lane_id>=2) {
        val.w += neighbor;
        warp_accumulated_offset += neighbor;
    }

    neighbor = __shfl_up_sync(0xffffffff, val.w, 4);
    if (lane_id>=4) {
        val.w += neighbor;
        warp_accumulated_offset += neighbor;
    }

    neighbor = __shfl_up_sync(0xffffffff, val.w, 8);
    if (lane_id>=8) {
        val.w += neighbor;
        warp_accumulated_offset += neighbor;
    }

    neighbor = __shfl_up_sync(0xffffffff, val.w, 16);
    if (lane_id>=16) {
        val.w += neighbor;
        warp_accumulated_offset += neighbor;
    }

    val.x += warp_accumulated_offset;
    val.y += warp_accumulated_offset;
    val.z += warp_accumulated_offset;

    // Phase 3:计算warp之间，block之内的前缀和。与 Kogge_Stone_scan_kernel_shfl_up_sync_version 的对应部分类似
    // 取出本warp的最后一个数字
    if(lane_id==31){
        warp_sums[warp_id] = val.w;
    }

    __syncthreads();

    float warp_base = 0.0f;

    if(warp_id==0){
        float w_val = warp_sums[lane_id];

        float w_neighbor;

        w_neighbor = __shfl_up_sync(0xffffffff, w_val, 1);
        if (lane_id>=1) w_val += w_neighbor;

        w_neighbor = __shfl_up_sync(0xffffffff, w_val, 2);
        if (lane_id>=2) w_val += w_neighbor;

        w_neighbor = __shfl_up_sync(0xffffffff, w_val, 4);
        if (lane_id>=4) w_val += w_neighbor;

        w_neighbor = __shfl_up_sync(0xffffffff, w_val, 8);
        if (lane_id>=8) w_val += w_neighbor;
        
        w_neighbor = __shfl_up_sync(0xffffffff, w_val, 16);
        if (lane_id>=16) w_val += w_neighbor;

        warp_sums[lane_id] = w_val;
    }

    __syncthreads();

    if (warp_id>0) {
        warp_base = warp_sums[warp_id-1];
    }

    val.x += warp_base;
    val.y += warp_base;
    val.z += warp_base;
    val.w += warp_base;

    if (idx<N/4) {
        Y_vec[idx] = val;
    }
}


// 在 Kogge_Stone_scan_kernel_shfl_up_sync_version_coarsening_degree_4 的基础上，增加粗化的程度
__global__ void Kogge_Stone_scan_kernel_shfl_up_sync_version_coarsening_degree_8(float *X, float *Y, unsigned int N){

    // shared memory是属于block的。一个block有1024个线程，那么就是有32个warp。
    __shared__ float warp_sums[32];

    unsigned int tid = threadIdx.x;
    unsigned int lane_id = tid % (32);
    unsigned int warp_id = tid / (32);

    unsigned int idx = 2*(blockIdx.x * blockDim.x + threadIdx.x);
    // 将数据拿到每个thread的寄存器上
    float4* X_vec = (float4*)X;
    float4* Y_vec = (float4*)Y;

    float4 val1 = (idx < N/4) ? X_vec[idx] : make_float4(0.0f,0.0f,0.0f,0.0f);
    float4 val2 = (idx+1 < N/4) ? X_vec[idx+1] : make_float4(0.0f,0.0f,0.0f,0.0f);

    // Phase 1：寄存器内的串行扫描 (Serial Scan)
    val1.y += val1.x;
    val1.z += val1.y;
    val1.w += val1.z;

    val2.x += val1.w;
    val2.y += val2.x;
    val2.z += val2.y;
    val2.w += val2.z;

    // Phase 2: 计算一个warp内部的前缀和

    float neighbor;
    float warp_accumulated_offset = 0.0f;

    neighbor = __shfl_up_sync(0xffffffff, val2.w, 1);
    if (lane_id>=1) {
        val2.w += neighbor;
        warp_accumulated_offset += neighbor;
    }

    neighbor = __shfl_up_sync(0xffffffff, val2.w, 2);
    if (lane_id>=2) {
        val2.w += neighbor;
        warp_accumulated_offset += neighbor;
    }

    neighbor = __shfl_up_sync(0xffffffff, val2.w, 4);
    if (lane_id>=4) {
        val2.w += neighbor;
        warp_accumulated_offset += neighbor;
    }

    neighbor = __shfl_up_sync(0xffffffff, val2.w, 8);
    if (lane_id>=8) {
        val2.w += neighbor;
        warp_accumulated_offset += neighbor;
    }

    neighbor = __shfl_up_sync(0xffffffff, val2.w, 16);
    if (lane_id>=16) {
        val2.w += neighbor;
        warp_accumulated_offset += neighbor;
    }

    val1.x += warp_accumulated_offset;
    val1.y += warp_accumulated_offset;
    val1.z += warp_accumulated_offset;
    val1.w += warp_accumulated_offset;
    val2.x += warp_accumulated_offset;
    val2.y += warp_accumulated_offset;
    val2.z += warp_accumulated_offset;

    // Phase 3:计算warp之间，block之内的前缀和。与 Kogge_Stone_scan_kernel_shfl_up_sync_version 的对应部分类似
    // 取出本warp的最后一个数字
    if(lane_id==31){
        warp_sums[warp_id] = val2.w;
    }

    __syncthreads();

    float warp_base = 0.0f;

    if(warp_id==0){
        float w_val = warp_sums[lane_id];

        float w_neighbor;

        w_neighbor = __shfl_up_sync(0xffffffff, w_val, 1);
        if (lane_id>=1) w_val += w_neighbor;

        w_neighbor = __shfl_up_sync(0xffffffff, w_val, 2);
        if (lane_id>=2) w_val += w_neighbor;

        w_neighbor = __shfl_up_sync(0xffffffff, w_val, 4);
        if (lane_id>=4) w_val += w_neighbor;

        w_neighbor = __shfl_up_sync(0xffffffff, w_val, 8);
        if (lane_id>=8) w_val += w_neighbor;
        
        w_neighbor = __shfl_up_sync(0xffffffff, w_val, 16);
        if (lane_id>=16) w_val += w_neighbor;

        warp_sums[lane_id] = w_val;
    }

    __syncthreads();

    if (warp_id>0) {
        warp_base = warp_sums[warp_id-1];
    }

    val1.x += warp_base;
    val1.y += warp_base;
    val1.z += warp_base;
    val1.w += warp_base;
    val2.x += warp_base;
    val2.y += warp_base;
    val2.z += warp_base;
    val2.w += warp_base;

    if (idx<N/4) Y_vec[idx] = val1;
    if (idx+1<N/4) Y_vec[idx+1] = val2;
}



// 在 Kogge_Stone_scan_kernel_shfl_up_sync_version_coarsening_degree_4 的基础上，增加粗化的程度
__global__ void Kogge_Stone_scan_kernel_shfl_up_sync_version_coarsening_degree_12(float *X, float *Y, unsigned int N){

    // shared memory是属于block的。一个block有1024个线程，那么就是有32个warp。
    __shared__ float warp_sums[32];

    unsigned int tid = threadIdx.x;
    unsigned int lane_id = tid % (32);
    unsigned int warp_id = tid / (32);

    unsigned int idx = 3*(blockIdx.x * blockDim.x + threadIdx.x);
    // 将数据拿到每个thread的寄存器上
    float4* X_vec = (float4*)X;
    float4* Y_vec = (float4*)Y;

    float4 val1 = (idx < N/4) ? X_vec[idx] : make_float4(0.0f,0.0f,0.0f,0.0f);
    float4 val2 = (idx+1 < N/4) ? X_vec[idx+1] : make_float4(0.0f,0.0f,0.0f,0.0f);
    float4 val3 = (idx+2 < N/4) ? X_vec[idx+2] : make_float4(0.0f,0.0f,0.0f,0.0f);

    // Phase 1：寄存器内的串行扫描 (Serial Scan)
    val1.y += val1.x;
    val1.z += val1.y;
    val1.w += val1.z;

    val2.x += val1.w;
    val2.y += val2.x;
    val2.z += val2.y;
    val2.w += val2.z;

    val3.x += val2.w;
    val3.y += val3.x;
    val3.z += val3.y;
    val3.w += val3.z;

    // Phase 2: 计算一个warp内部的前缀和

    float neighbor;
    float warp_accumulated_offset = 0.0f;

    neighbor = __shfl_up_sync(0xffffffff, val3.w, 1);
    if (lane_id>=1) {
        val3.w += neighbor;
        warp_accumulated_offset += neighbor;
    }

    neighbor = __shfl_up_sync(0xffffffff, val3.w, 2);
    if (lane_id>=2) {
        val3.w += neighbor;
        warp_accumulated_offset += neighbor;
    }

    neighbor = __shfl_up_sync(0xffffffff, val3.w, 4);
    if (lane_id>=4) {
        val3.w += neighbor;
        warp_accumulated_offset += neighbor;
    }

    neighbor = __shfl_up_sync(0xffffffff, val3.w, 8);
    if (lane_id>=8) {
        val3.w += neighbor;
        warp_accumulated_offset += neighbor;
    }

    neighbor = __shfl_up_sync(0xffffffff, val3.w, 16);
    if (lane_id>=16) {
        val3.w += neighbor;
        warp_accumulated_offset += neighbor;
    }

    val1.x += warp_accumulated_offset;
    val1.y += warp_accumulated_offset;
    val1.z += warp_accumulated_offset;
    val1.w += warp_accumulated_offset;
    val2.x += warp_accumulated_offset;
    val2.y += warp_accumulated_offset;
    val2.z += warp_accumulated_offset;
    val2.w += warp_accumulated_offset;
    val3.x += warp_accumulated_offset;
    val3.y += warp_accumulated_offset;
    val3.z += warp_accumulated_offset;

    // Phase 3:计算warp之间，block之内的前缀和。与 Kogge_Stone_scan_kernel_shfl_up_sync_version 的对应部分类似
    // 取出本warp的最后一个数字
    if(lane_id==31){
        warp_sums[warp_id] = val3.w;
    }

    __syncthreads();

    float warp_base = 0.0f;

    if(warp_id==0){
        float w_val = warp_sums[lane_id];

        float w_neighbor;

        w_neighbor = __shfl_up_sync(0xffffffff, w_val, 1);
        if (lane_id>=1) w_val += w_neighbor;

        w_neighbor = __shfl_up_sync(0xffffffff, w_val, 2);
        if (lane_id>=2) w_val += w_neighbor;

        w_neighbor = __shfl_up_sync(0xffffffff, w_val, 4);
        if (lane_id>=4) w_val += w_neighbor;

        w_neighbor = __shfl_up_sync(0xffffffff, w_val, 8);
        if (lane_id>=8) w_val += w_neighbor;
        
        w_neighbor = __shfl_up_sync(0xffffffff, w_val, 16);
        if (lane_id>=16) w_val += w_neighbor;

        warp_sums[lane_id] = w_val;
    }

    __syncthreads();

    if (warp_id>0) {
        warp_base = warp_sums[warp_id-1];
    }

    val1.x += warp_base;
    val1.y += warp_base;
    val1.z += warp_base;
    val1.w += warp_base;
    val2.x += warp_base;
    val2.y += warp_base;
    val2.z += warp_base;
    val2.w += warp_base;
    val3.x += warp_base;
    val3.y += warp_base;
    val3.z += warp_base;
    val3.w += warp_base;

    if (idx<N/4) Y_vec[idx] = val1;
    if (idx+1<N/4) Y_vec[idx+1] = val2;
    if (idx+2<N/4) Y_vec[idx+2] = val3;
}


// 在 Kogge_Stone_scan_kernel_shfl_up_sync_version_coarsening_degree_4 的基础上，增加粗化的程度
__global__ void Kogge_Stone_scan_kernel_shfl_up_sync_version_coarsening_degree_16(float *X, float *Y, unsigned int N){

    // shared memory是属于block的。一个block有1024个线程，那么就是有32个warp。
    __shared__ float warp_sums[32];

    unsigned int tid = threadIdx.x;
    unsigned int lane_id = tid % (32);
    unsigned int warp_id = tid / (32);

    unsigned int idx = 4*(blockIdx.x * blockDim.x + threadIdx.x);
    // 将数据拿到每个thread的寄存器上
    float4* X_vec = (float4*)X;
    float4* Y_vec = (float4*)Y;

    float4 val1 = (idx < N/4) ? X_vec[idx] : make_float4(0.0f,0.0f,0.0f,0.0f);
    float4 val2 = (idx+1 < N/4) ? X_vec[idx+1] : make_float4(0.0f,0.0f,0.0f,0.0f);
    float4 val3 = (idx+2 < N/4) ? X_vec[idx+2] : make_float4(0.0f,0.0f,0.0f,0.0f);
    float4 val4 = (idx+3 < N/4) ? X_vec[idx+3] : make_float4(0.0f,0.0f,0.0f,0.0f);

    // Phase 1：寄存器内的串行扫描 (Serial Scan)
    val1.y += val1.x;
    val1.z += val1.y;
    val1.w += val1.z;

    val2.x += val1.w;
    val2.y += val2.x;
    val2.z += val2.y;
    val2.w += val2.z;

    val3.x += val2.w;
    val3.y += val3.x;
    val3.z += val3.y;
    val3.w += val3.z;

    val4.x += val3.w;
    val4.y += val4.x;
    val4.z += val4.y;
    val4.w += val4.z;

    // Phase 2: 计算一个warp内部的前缀和

    float neighbor;
    float warp_accumulated_offset = 0.0f;

    neighbor = __shfl_up_sync(0xffffffff, val4.w, 1);
    if (lane_id>=1) {
        val4.w += neighbor;
        warp_accumulated_offset += neighbor;
    }

    neighbor = __shfl_up_sync(0xffffffff, val4.w, 2);
    if (lane_id>=2) {
        val4.w += neighbor;
        warp_accumulated_offset += neighbor;
    }

    neighbor = __shfl_up_sync(0xffffffff, val4.w, 4);
    if (lane_id>=4) {
        val4.w += neighbor;
        warp_accumulated_offset += neighbor;
    }

    neighbor = __shfl_up_sync(0xffffffff, val4.w, 8);
    if (lane_id>=8) {
        val4.w += neighbor;
        warp_accumulated_offset += neighbor;
    }

    neighbor = __shfl_up_sync(0xffffffff, val4.w, 16);
    if (lane_id>=16) {
        val4.w += neighbor;
        warp_accumulated_offset += neighbor;
    }

    val1.x += warp_accumulated_offset;
    val1.y += warp_accumulated_offset;
    val1.z += warp_accumulated_offset;
    val1.w += warp_accumulated_offset;
    val2.x += warp_accumulated_offset;
    val2.y += warp_accumulated_offset;
    val2.z += warp_accumulated_offset;
    val2.w += warp_accumulated_offset;
    val3.x += warp_accumulated_offset;
    val3.y += warp_accumulated_offset;
    val3.z += warp_accumulated_offset;
    val3.w += warp_accumulated_offset;
    val4.x += warp_accumulated_offset;
    val4.y += warp_accumulated_offset;
    val4.z += warp_accumulated_offset;

    // Phase 3:计算warp之间，block之内的前缀和。与 Kogge_Stone_scan_kernel_shfl_up_sync_version 的对应部分类似
    // 取出本warp的最后一个数字
    if(lane_id==31){
        warp_sums[warp_id] = val4.w;
    }

    __syncthreads();

    float warp_base = 0.0f;

    if(warp_id==0){
        float w_val = warp_sums[lane_id];

        float w_neighbor;

        w_neighbor = __shfl_up_sync(0xffffffff, w_val, 1);
        if (lane_id>=1) w_val += w_neighbor;

        w_neighbor = __shfl_up_sync(0xffffffff, w_val, 2);
        if (lane_id>=2) w_val += w_neighbor;

        w_neighbor = __shfl_up_sync(0xffffffff, w_val, 4);
        if (lane_id>=4) w_val += w_neighbor;

        w_neighbor = __shfl_up_sync(0xffffffff, w_val, 8);
        if (lane_id>=8) w_val += w_neighbor;
        
        w_neighbor = __shfl_up_sync(0xffffffff, w_val, 16);
        if (lane_id>=16) w_val += w_neighbor;

        warp_sums[lane_id] = w_val;
    }

    __syncthreads();

    if (warp_id>0) {
        warp_base = warp_sums[warp_id-1];
    }

    val1.x += warp_base;
    val1.y += warp_base;
    val1.z += warp_base;
    val1.w += warp_base;
    val2.x += warp_base;
    val2.y += warp_base;
    val2.z += warp_base;
    val2.w += warp_base;
    val3.x += warp_base;
    val3.y += warp_base;
    val3.z += warp_base;
    val3.w += warp_base;
    val4.x += warp_base;
    val4.y += warp_base;
    val4.z += warp_base;
    val4.w += warp_base;

    if (idx<N/4) Y_vec[idx] = val1;
    if (idx+1<N/4) Y_vec[idx+1] = val2;
    if (idx+2<N/4) Y_vec[idx+2] = val3;
    if (idx+3<N/4) Y_vec[idx+3] = val4;
}


// 在 Kogge_Stone_scan_kernel_shfl_up_sync_version_coarsening_degree_4 的基础上，增加粗化的程度
__global__ void Kogge_Stone_scan_kernel_shfl_up_sync_version_coarsening_degree_20(float *X, float *Y, unsigned int N){

    // shared memory是属于block的。一个block有1024个线程，那么就是有32个warp。
    __shared__ float warp_sums[32];

    unsigned int tid = threadIdx.x;
    unsigned int lane_id = tid % (32);
    unsigned int warp_id = tid / (32);

    unsigned int idx = 5*(blockIdx.x * blockDim.x + threadIdx.x);
    // 将数据拿到每个thread的寄存器上
    float4* X_vec = (float4*)X;
    float4* Y_vec = (float4*)Y;

    float4 val1 = (idx < N/4) ? X_vec[idx] : make_float4(0.0f,0.0f,0.0f,0.0f);
    float4 val2 = (idx+1 < N/4) ? X_vec[idx+1] : make_float4(0.0f,0.0f,0.0f,0.0f);
    float4 val3 = (idx+2 < N/4) ? X_vec[idx+2] : make_float4(0.0f,0.0f,0.0f,0.0f);
    float4 val4 = (idx+3 < N/4) ? X_vec[idx+3] : make_float4(0.0f,0.0f,0.0f,0.0f);
    float4 val5 = (idx+4 < N/4) ? X_vec[idx+4] : make_float4(0.0f,0.0f,0.0f,0.0f);

    // Phase 1：寄存器内的串行扫描 (Serial Scan)
    val1.y += val1.x;
    val1.z += val1.y;
    val1.w += val1.z;

    val2.x += val1.w;
    val2.y += val2.x;
    val2.z += val2.y;
    val2.w += val2.z;

    val3.x += val2.w;
    val3.y += val3.x;
    val3.z += val3.y;
    val3.w += val3.z;

    val4.x += val3.w;
    val4.y += val4.x;
    val4.z += val4.y;
    val4.w += val4.z;

    val5.x += val4.w;
    val5.y += val5.x;
    val5.z += val5.y;
    val5.w += val5.z;

    // Phase 2: 计算一个warp内部的前缀和

    float neighbor;
    float warp_accumulated_offset = 0.0f;

    neighbor = __shfl_up_sync(0xffffffff, val5.w, 1);
    if (lane_id>=1) {
        val5.w += neighbor;
        warp_accumulated_offset += neighbor;
    }

    neighbor = __shfl_up_sync(0xffffffff, val5.w, 2);
    if (lane_id>=2) {
        val5.w += neighbor;
        warp_accumulated_offset += neighbor;
    }

    neighbor = __shfl_up_sync(0xffffffff, val5.w, 4);
    if (lane_id>=4) {
        val5.w += neighbor;
        warp_accumulated_offset += neighbor;
    }

    neighbor = __shfl_up_sync(0xffffffff, val5.w, 8);
    if (lane_id>=8) {
        val5.w += neighbor;
        warp_accumulated_offset += neighbor;
    }

    neighbor = __shfl_up_sync(0xffffffff, val5.w, 16);
    if (lane_id>=16) {
        val5.w += neighbor;
        warp_accumulated_offset += neighbor;
    }

    val1.x += warp_accumulated_offset;
    val1.y += warp_accumulated_offset;
    val1.z += warp_accumulated_offset;
    val1.w += warp_accumulated_offset;
    val2.x += warp_accumulated_offset;
    val2.y += warp_accumulated_offset;
    val2.z += warp_accumulated_offset;
    val2.w += warp_accumulated_offset;
    val3.x += warp_accumulated_offset;
    val3.y += warp_accumulated_offset;
    val3.z += warp_accumulated_offset;
    val3.w += warp_accumulated_offset;
    val4.x += warp_accumulated_offset;
    val4.y += warp_accumulated_offset;
    val4.z += warp_accumulated_offset;
    val4.w += warp_accumulated_offset;
    val5.x += warp_accumulated_offset;
    val5.y += warp_accumulated_offset;
    val5.z += warp_accumulated_offset;

    // Phase 3:计算warp之间，block之内的前缀和。与 Kogge_Stone_scan_kernel_shfl_up_sync_version 的对应部分类似
    // 取出本warp的最后一个数字
    if(lane_id==31){
        warp_sums[warp_id] = val5.w;
    }

    __syncthreads();

    float warp_base = 0.0f;

    if(warp_id==0){
        float w_val = warp_sums[lane_id];

        float w_neighbor;

        w_neighbor = __shfl_up_sync(0xffffffff, w_val, 1);
        if (lane_id>=1) w_val += w_neighbor;

        w_neighbor = __shfl_up_sync(0xffffffff, w_val, 2);
        if (lane_id>=2) w_val += w_neighbor;

        w_neighbor = __shfl_up_sync(0xffffffff, w_val, 4);
        if (lane_id>=4) w_val += w_neighbor;

        w_neighbor = __shfl_up_sync(0xffffffff, w_val, 8);
        if (lane_id>=8) w_val += w_neighbor;
        
        w_neighbor = __shfl_up_sync(0xffffffff, w_val, 16);
        if (lane_id>=16) w_val += w_neighbor;

        warp_sums[lane_id] = w_val;
    }

    __syncthreads();

    if (warp_id>0) {
        warp_base = warp_sums[warp_id-1];
    }

    val1.x += warp_base;
    val1.y += warp_base;
    val1.z += warp_base;
    val1.w += warp_base;
    val2.x += warp_base;
    val2.y += warp_base;
    val2.z += warp_base;
    val2.w += warp_base;
    val3.x += warp_base;
    val3.y += warp_base;
    val3.z += warp_base;
    val3.w += warp_base;
    val4.x += warp_base;
    val4.y += warp_base;
    val4.z += warp_base;
    val4.w += warp_base;
    val5.x += warp_base;
    val5.y += warp_base;
    val5.z += warp_base;
    val5.w += warp_base;

    if (idx<N/4) Y_vec[idx] = val1;
    if (idx+1<N/4) Y_vec[idx+1] = val2;
    if (idx+2<N/4) Y_vec[idx+2] = val3;
    if (idx+3<N/4) Y_vec[idx+3] = val4;
    if (idx+4<N/4) Y_vec[idx+4] = val5;
}


// 这里的 SECTION_SIZE 最多只能是 2048，它应该与 blockDim.x 应该等于 SECTION_SIZE/2
// 测试的时候，请使用2的幂的blocksize，例如，256,512,1024,2048。
template <unsigned int SECTION_SIZE>
__global__ void Brent_Kung_scan_kernel(float *X, float *Y, unsigned int N){
    __shared__ float XY[SECTION_SIZE];

    // 每一个block，管理着2倍于block大小的一个连续的区间
    // 每一个线程，每次操作的那个数字的位置，是会发生变化的。有一个映射管理着这件事
    // 但是，任何时候，需要的线程的数量，都不大于SECTION_SIZE的1/2
    unsigned int i = 2*blockIdx.x*blockDim.x + threadIdx.x;

    // 所以要分两次把数据拉进 shared memory
    if (i < N) {
        XY[threadIdx.x] = X[i];
    } else {
        XY[threadIdx.x] = 0.0f;
    }
    if (i + blockDim.x < N) {
        XY[threadIdx.x + blockDim.x] = X[i + blockDim.x];
    } else {
        XY[threadIdx.x + blockDim.x] = 0.0f;
    }

    for(unsigned int stride = 1; stride <= blockDim.x; stride *=2){
        __syncthreads();
        unsigned int index = (threadIdx.x+1)*2*stride-1;
        if(index<SECTION_SIZE){
            XY[index] += XY[index-stride];
        }
    }

    for(unsigned int stride = SECTION_SIZE/4; stride > 0; stride /=2){
        __syncthreads();
        unsigned int index = (threadIdx.x +1)*stride*2-1;
        if(index+stride<SECTION_SIZE){
            XY[index+stride] += XY[index];
        }
    }
    __syncthreads();
    if (i<N) Y[i] = XY[threadIdx.x];
    if (i+blockDim.x<N) Y[i+blockDim.x] = XY[threadIdx.x+blockDim.x];
}


#define LOG_NUM_BANKS 5
#define PADDED_INDEX(n) ( (n) + ((n) >> LOG_NUM_BANKS) )
template <unsigned int SECTION_SIZE>
__global__ void Brent_Kung_scan_kernel_optimized_padding(float* X, float* Y, unsigned int N){
    __shared__ float XY[SECTION_SIZE + (SECTION_SIZE >> LOG_NUM_BANKS)];

    unsigned int i = 2*blockIdx.x*blockDim.x + threadIdx.x;

    int s_idx_1 = threadIdx.x;
    int s_idx_2 = threadIdx.x + blockDim.x;

    int p_idx_1 = PADDED_INDEX(s_idx_1);
    int p_idx_2 = PADDED_INDEX(s_idx_2);

    if (i < N) {
        XY[p_idx_1] = X[i];
    } else {
        XY[p_idx_1] = 0.0f;
    }
    if (i + blockDim.x < N) {
        XY[p_idx_2] = X[i + blockDim.x];
    } else {
        XY[p_idx_2] = 0.0f;
    }

    for(unsigned int stride = 1; stride <= blockDim.x; stride *=2){
        __syncthreads();
        unsigned int index = (threadIdx.x+1)*2*stride-1;
        if(index<SECTION_SIZE){
            XY[PADDED_INDEX(index)] += XY[PADDED_INDEX(index-stride)];
        }
    }

    for(unsigned int stride = SECTION_SIZE/4; stride > 0; stride /=2){
        __syncthreads();
        unsigned int index = (threadIdx.x +1)*stride*2-1;
        if(index+stride<SECTION_SIZE){
            XY[PADDED_INDEX(index+stride)] += XY[PADDED_INDEX(index)];
        }
    }
    __syncthreads();
    if (i<N) Y[i] = XY[p_idx_1];
    if (i+blockDim.x<N) Y[i+blockDim.x] = XY[p_idx_2];
}


// 我们假设 N 是32的倍数。这个假设，并不过分。
#define LOG_NUM_BANKS 5
#define PADDED_INDEX(n) ( (n) + ((n) >> LOG_NUM_BANKS) )
template <unsigned int SECTION_SIZE>
__global__ void Brent_Kung_scan_kernel_optimized_padding_plus_vectorized_memory_access(float* X, float* Y, unsigned int N){
    __shared__ float XY[SECTION_SIZE + (SECTION_SIZE >> LOG_NUM_BANKS)];

    float2* X_vec = (float2*)X;
    float2* Y_vec = (float2*)Y;

    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;

    float2 val = X_vec[idx]; // 这里直接读取，没有防护。

    int s_idx_1 = 2*tid;
    int s_idx_2 = 2*tid + 1;

    int p_idx_1 = PADDED_INDEX(s_idx_1);
    int p_idx_2 = PADDED_INDEX(s_idx_2);

    if (2*idx < N) {
        XY[p_idx_1] = val.x;
    } else {
        XY[p_idx_1] = 0.0f;
    }
    if (2*idx+1 < N) {
        XY[p_idx_2] = val.y;
    } else {
        XY[p_idx_2] = 0.0f;
    }

    for(unsigned int stride = 1; stride <= blockDim.x; stride *=2){
        __syncthreads();
        unsigned int index = (threadIdx.x+1)*2*stride-1;
        if(index<SECTION_SIZE){
            XY[PADDED_INDEX(index)] += XY[PADDED_INDEX(index-stride)];
        }
    }

    for(unsigned int stride = SECTION_SIZE/4; stride > 0; stride /=2){
        __syncthreads();
        unsigned int index = (threadIdx.x +1)*stride*2-1;
        if(index+stride<SECTION_SIZE){
            XY[PADDED_INDEX(index+stride)] += XY[PADDED_INDEX(index)];
        }
    }
    __syncthreads();

    val.x = XY[p_idx_1];
    val.y = XY[p_idx_2];

    if (2*idx < N) {
        Y_vec[idx] = val;
    }
}

#endif // SCAN_KERNELS_CUH